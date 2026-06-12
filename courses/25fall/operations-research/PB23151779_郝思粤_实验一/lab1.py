import numpy as np
import time
import matplotlib.pyplot as plt

class SimplexSolver:
    def __init__(self, c, A, b):
        """
        初始化线性规划问题: 
        求解最小化问题: min c^T x, s.t. Ax = b, x >= 0
        """
        self.c = np.array(c, dtype=float) 
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.m, self.n = self.A.shape
        
        # 状态标记
        self.status = "Not Started"
        self.solution = None
        self.optimal_val = None

    def _preprocess(self):
        """
        模块0：预处理
        返回 True 表示继续，Returns False 表示已确定状态（如无解）
        """
        # 1. 检查全为0的行
        zero_rows = np.all(self.A == 0, axis=1)
        if np.any(zero_rows):
            # 如果对应行的b不为0，则出现 0 = non_zero 矛盾
            if np.any(self.b[zero_rows] != 0):
                self.status = "LP has no feasible point"
                return False
            else:
                # 移除全0行和对应的b（0=0约束是冗余的）
                self.A = self.A[~zero_rows]
                self.b = self.b[~zero_rows]
                self.m = self.A.shape[0]

        # 2. 修复：确保 RHS 非负（等式约束行整体乘 -1 不改变可行域）
        neg_rhs = self.b < 0
        if np.any(neg_rhs):
            self.A[neg_rhs, :] *= -1.0
            self.b[neg_rhs] *= -1.0

        return True

    def _check_full_rank_qr(self):
        """
        模块1：检查A是否满秩，使用QR分解
        如果秩 < m：
        1. 检查是否存在矛盾 (Infeasible)
        2. 如果无矛盾，移除多余的约束行
        返回: True(继续), False(无解)
        """
        if self.m == 0:
            return True

        # 鲁棒性检查 
        rank_A = np.linalg.matrix_rank(self.A)
        Augmented = np.hstack([self.A, self.b.reshape(-1, 1)])
        rank_Augmented = np.linalg.matrix_rank(Augmented)

        if rank_A < rank_Augmented:
            self.status = "LP has no feasible point"
            return False

        # 修复：使用带列主元的 QR，按要求删冗余约束
        B = self.A.T  # n x m
        tol = 1e-10

        try:
            from scipy.linalg import qr as scipy_qr
            Q, R, piv = scipy_qr(B, pivoting=True, mode="economic")
            diag_R = np.abs(np.diag(R)) if R.size else np.array([])
            r = int(np.sum(diag_R > tol))

            if r < self.m:
                keep_rows = np.sort(piv[:r])
                self.A = self.A[keep_rows, :]
                self.b = self.b[keep_rows]
                self.m = self.A.shape[0]

        except Exception:
            Q, R = np.linalg.qr(B)
            diag_R = np.abs(np.diag(R))
            if rank_A < self.m:
                keep_idx = diag_R > tol
                if np.sum(keep_idx) != rank_A:
                    keep_idx = np.zeros(self.m, dtype=bool)
                    keep_idx[:rank_A] = True
                self.A = self.A[keep_idx]
                self.b = self.b[keep_idx]
                self.m = self.A.shape[0]

        return True

    def solve(self):
        # 步骤0：预处理 
        if not self._preprocess():
            return self.status

        # 步骤1：秩检查 
        if not self._check_full_rank_qr():
            return self.status

        # 步骤2：大M法初始化
        M = 100000.0 
        
        # 扩展 c: [c_1, ..., c_n, M, ..., M]
        c_extended = np.concatenate([self.c, np.full(self.m, M)])
        
        # 扩展 A: [A | I]
        A_extended = np.hstack([self.A, np.eye(self.m)])
        
        # 初始基变量索引 (人工变量)
        basis_indices = list(range(self.n, self.n + self.m)) 
        
        # 初始化单纯形表
        tableau = np.zeros((self.m + 1, self.n + self.m + 1))
        
        # 填入约束部分
        tableau[1:, 0] = self.b
        tableau[1:, 1:] = A_extended
        
        # 计算初始检验数 (Reduced Costs)
        for j in range(self.n + self.m):
            col = A_extended[:, j]
            z_j = np.dot(np.full(self.m, M), col)
            tableau[0, j+1] = c_extended[j] - z_j
            
        # 初始目标函数值
        tableau[0, 0] = -np.dot(np.full(self.m, M), self.b)

        # 步骤3：单纯形法迭代
        iteration = 0
        max_iter = 10000
        
        while iteration < max_iter:
            # 1. 入基
            reduced_costs = tableau[0, 1:]
            candidates = np.where(reduced_costs < -1e-9)[0]
            
            if len(candidates) == 0:
                break
            
            # Bland's Rule
            entering_idx = candidates[0] 
            
            # 2. 出基 (最小比率)
            pivot_col = tableau[1:, entering_idx + 1]
            rhs = tableau[1:, 0]
            
            ratios = []
            row_indices = []
            
            for i in range(self.m):
                if pivot_col[i] > 1e-9:
                    ratios.append(rhs[i] / pivot_col[i])
                    row_indices.append(i)
            
            if not ratios:
                self.status = "unbounded LP"
                return self.status
            
            min_ratio = min(ratios)
            min_ratio_indices = [row_indices[i] for i, r in enumerate(ratios) if abs(r - min_ratio) < 1e-9]
            
            if len(min_ratio_indices) > 1:
                leaving_row_local = -1
                min_basis_idx = float('inf')
                for r_idx in min_ratio_indices:
                    current_basis_var = basis_indices[r_idx]
                    if current_basis_var < min_basis_idx:
                        min_basis_idx = current_basis_var
                        leaving_row_local = r_idx
            else:
                leaving_row_local = min_ratio_indices[0]
                
            basis_indices[leaving_row_local] = entering_idx
            
            # 3. Pivot
            pivot_row_idx_in_tableau = leaving_row_local + 1
            pivot_element = tableau[pivot_row_idx_in_tableau, entering_idx + 1]
            tableau[pivot_row_idx_in_tableau, :] /= pivot_element
            for i in range(self.m + 1):
                if i != pivot_row_idx_in_tableau:
                    factor = tableau[i, entering_idx + 1]
                    tableau[i, :] -= factor * tableau[pivot_row_idx_in_tableau, :]
            
            iteration += 1
            
        if iteration == max_iter:
            self.status = "Not Converged (Cycle?)"
            return self.status

        # 迭代结束，检查人工变量
        has_artificial = False
        for i in range(self.m):
            if basis_indices[i] >= self.n: 
                if tableau[i+1, 0] > 1e-6: 
                    has_artificial = True
                    break
        
        if has_artificial:
            self.status = "LP has no feasible point"
        else:
            self.status = "Optimal"
            self.optimal_val = -tableau[0, 0] 
            x_sol = np.zeros(self.n)
            for i in range(self.m):
                idx = basis_indices[i]
                if idx < self.n:
                    x_sol[idx] = tableau[i+1, 0]
            self.solution = x_sol
            
        return self.status

# --- 测试与统计模块 ---

def generate_problem(m, n):
    """随机生成线性规划问题 (确保大概率有解)"""
    x0 = np.random.rand(n)
    A = np.random.rand(m, n) * 10 - 5
    b = A.dot(x0) 
    c = np.random.rand(n) * 10 - 5
    return c, A, b

def run_performance_test():
    scales = [(m, 2*m) for m in range(15, 181, 15)] # (m, n)
    results_mean = []
    results_std = []
    
    print("\n=== 开始性能测试 ===")
    
    for m, n in scales:
        times = []
        valid_count = 0
        print(f"正在测试规模 m={m}, n={n} ...")
        
        for _ in range(20): 
            c, A, b = generate_problem(m, n)
            solver = SimplexSolver(c, A, b)
            
            start_time = time.time()
            status = solver.solve()
            end_time = time.time()
            
            # 仅统计找到最优解的情况
            if status == "Optimal":
                times.append(end_time - start_time)
                valid_count += 1
        
        if times:
            avg_time = np.mean(times)
            std_time = np.std(times)
            results_mean.append(avg_time)
            results_std.append(std_time)
            print(f"  成功样本: {valid_count}/20, 平均时间: {avg_time:.4f}s, 方差: {std_time**2:.6f}")
        else:
            results_mean.append(0)
            results_std.append(0)

    # 绘图
    plt.figure(figsize=(10, 6))
    x_labels = [f"{m}x{n}" for m, n in scales]
    plt.errorbar(x_labels, results_mean, yerr=results_std, fmt='-o', capsize=5)
    plt.title("Simplex Algorithm Performance vs Scale")
    plt.xlabel("Problem Scale (Constraints x Variables)")
    plt.ylabel("Average Time (seconds)")
    plt.grid(True)
    plt.savefig('efficiency_plot.png')
    print("性能统计图已保存为 efficiency_plot.png")

def simple_test():
    print("\n" + "="*20 + " 算法正确性验证 (Minimization) " + "="*20)
    
    # ---------------------------------------------------------
    # 案例 1: 正常有最优解
    # ---------------------------------------------------------
    print("\n>>> 测试 1: 标准有解 (Min -3x1 - 2x2)")
    c1 = [-3, -2, 0, 0, 0] 
    A1 = [[2, 1, 1, 0, 0], [1, 1, 0, 1, 0], [1, 0, 0, 0, 1]]
    b1 = [100, 80, 40]
    
    solver1 = SimplexSolver(c1, A1, b1)
    res1 = solver1.solve()
    print("[原始问题]")
    print(f"   A:{A1}")
    print(f"   b:{b1}")
    print(f"   c:{c1}")
    print(f"  [结果] 状态: {res1}")
    print(f"  [目标值] Optimal Z: {solver1.optimal_val}")
    print(f"  [解向量] x: {solver1.solution}")

    # ---------------------------------------------------------
    # 案例 2: 冗余约束
    # ---------------------------------------------------------
    print("\n>>> 测试 2: 冗余约束测试 (QR分解应移除多余行)")
    c2 = [1, 1, 0]
    A2 = [[1, 1, 1], [2, 2, 2]]
    b2 = [10, 20]
    
    solver2 = SimplexSolver(c2, A2, b2)
    print("[原始问题]")
    print(f"   A:{A2}")
    print(f"   b:{b2}")
    print(f"   c:{c2}")
    print(f"  [原始维度] m={solver2.m}")
    res2 = solver2.solve()
    print(f"  [处理后维度] m={solver2.m} (若减少说明成功移除冗余)")
    print(f"  [结果] 状态: {res2}")
    print(f"  [目标值] Optimal Z: {solver2.optimal_val}")
    print(f"  [解向量] x: {solver2.solution}") 

    # ---------------------------------------------------------
    # 案例 3: 无可行解
    # ---------------------------------------------------------
    print("\n>>> 测试 3: 无可行解 (应输出: LP has no feasible point)")
    c3 = [2, 1]
    A3 = [[1, 1], [1, 1]]
    b3 = [10, 20] 
    solver3 = SimplexSolver(c3, A3, b3)
    res3 = solver3.solve()
    print("[原始问题]")
    print(f"   A:{A3}")
    print(f"   b:{b3}")
    print(f"   c:{c3}")
    print(f"  [结果] 状态: {res3}") 

    # ---------------------------------------------------------
    # 案例 4: 无界解
    # ---------------------------------------------------------
    print("\n>>> 测试 4: 无界解 (应输出: unbounded LP)")
    c4 = [-1, 0, 0] 
    A4 = [[1, -1, 1]]
    b4 = [10]
    solver4 = SimplexSolver(c4, A4, b4)
    res4 = solver4.solve()
    print("[原始问题]")
    print(f"   A:{A4}")
    print(f"   b:{b4}")
    print(f"   c:{c4}")
    print(f"  [结果] 状态: {res4}")

    print("\n=== 书上循环基解例子===")
    c_cycle = [-4, -1, 0, 0, 0]  # [x1,x2,s1,s2,s3]
    A_cycle = [
        [-1,  2, 1, 0, 0],
        [ 2,  3, 0, 1, 0],
        [ 1, -1, 0, 0, 1],
    ]
    b_cycle = [4, 12, 3]

    solver_cycle = SimplexSolver(c_cycle, A_cycle, b_cycle)
    t0 = time.time()
    status_cycle = solver_cycle.solve()
    t1 = time.time()
    print(f"状态: {status_cycle}")
    print(f"用时: {t1 - t0:.6f}s")
    print(f"目标值: {solver_cycle.optimal_val}")
    print(f"解向量: {solver_cycle.solution}")

if __name__ == "__main__":
    simple_test()
    run_performance_test()
