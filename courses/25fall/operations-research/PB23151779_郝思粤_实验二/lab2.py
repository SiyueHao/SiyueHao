import heapq
import networkx as nx
import numpy as np
import time
import matplotlib.pyplot as plt
import math

# ==========================================
# 模块 1: Dijkstra 算法 (使用 heapq)
# ==========================================

def dijkstra_heap(G, source, target):
    """
    使用 heapq 实现的 Dijkstra 最短路径算法
    输入：
        G      : networkx Graph 对象 
        source : 起点
        target : 终点
    输出：
        最短路长度, 路径列表
    """
    distances = {node: float('inf') for node in G.nodes()}
    distances[source] = 0.0

    predecessors = {node: None for node in G.nodes()}

    # 最小堆：存 (当前距离, 节点)
    priority_queue = [(0.0, source)]

    visited = set()

    while priority_queue:
        current_dist, u = heapq.heappop(priority_queue)

        # 懒惰删除：如果该点已被最终确定过最短距离，则跳过
        if u in visited:
            continue
        visited.add(u)

        # 单对最短路：到达 target 可提前结束
        if u == target:
            break

        # 遍历邻居并松弛
        for v, edge_data in G[u].items():
            weight = edge_data.get('weight', 1.0)
            nd = current_dist + weight
            if nd < distances[v]:
                distances[v] = nd
                predecessors[v] = u
                heapq.heappush(priority_queue, (nd, v))

    return distances[target], get_path(predecessors, source, target)


def get_path(predecessors, source, target):
    """根据前驱字典还原路径；若不可达返回 []"""
    path = []
    curr = target
    while curr is not None:
        path.append(curr)
        if curr == source:
            break
        curr = predecessors[curr]
    return path[::-1] if path and path[-1] == source else []


# ==========================================
# 模块 2: 图的生成与检查
# ==========================================

def check_graph_validity(G):
    """
    评分点：判断图是否连通，是否有负权重边
    """
    # 1) 负权重检查（Dijkstra 不允许负权边）
    for u, v, data in G.edges(data=True):
        if data.get('weight', 1) < 0:
            raise ValueError(f"检测到负权重边: ({u}, {v}) weight={data['weight']}。Dijkstra 无法处理。")

    # 2) 连通性检查
    if G.number_of_nodes() > 0 and (not nx.is_connected(G)):
        raise ValueError("图不连通，无法保证存在最短路径。")

    return True


def generate_connected_er_graph(n):
    """
    生成随机连通图 (Erdos-Renyi)
    依据理论：若 p > (1+epsilon) * ln(n) / n，则图几乎必然连通
    这里取 epsilon=0.5，即 p = 1.5 * ln(n)/n，并用循环确保连通
    """
    if n <= 1:
        G = nx.Graph()
        G.add_nodes_from(range(n))
        return G

    p = 1.5 * math.log(n) / n
    p = min(p, 1.0)

    while True:
        G = nx.erdos_renyi_graph(n, p)
        if nx.is_connected(G):
            # 随机生成正权重 (1~10)，避免负权
            for (u, v) in G.edges():
                G.edges[u, v]['weight'] = int(np.random.randint(1, 11))
            return G


def print_graph_edges_with_weights(G, max_edges_show=60):
    """
    打印图的边及其权重（用于正确性验证的可观察输出）
    """
    edges = list(G.edges(data=True))
    # 统一排序，便于复现实验报告里的“截图/记录”
    edges_sorted = sorted(edges, key=lambda x: (min(x[0], x[1]), max(x[0], x[1])))

    show_cnt = min(len(edges_sorted), max_edges_show)
    print(f"边及权重（显示 {show_cnt}/{len(edges_sorted)} 条，格式: u-v(w)）：")
    for i in range(show_cnt):
        u, v, data = edges_sorted[i]
        w = data.get("weight", 1)
        print(f"  {u}-{v} (w={w})")
    if len(edges_sorted) > show_cnt:
        print("  ...（其余边略）")


# ==========================================
# 模块 3: COPT 求解最短路的 LP 模型
# ==========================================

def solve_shortest_path_lp_copt(G, source, target, env=None, timelimit=30.0, verbose=False):
    """
    用 COPT 求解最短路的 LP（流量守恒）模型
    """
    import coptpy as cp
    from coptpy import COPT

    # 1) 创建/复用 COPT 环境（复用可以减少重复初始化开销）
    if env is None:
        env = cp.Envr()

    # 2) 创建模型
    model = env.createModel("shortest_path_lp")

    # 3) 关闭求解器日志，减少与实验结果无关的输出
    if not verbose:
        try:
            model.setParam(COPT.Param.Logging, 0)
        except Exception:
            pass

    # 4) 时间限制（防止极端情况下卡住）
    if timelimit is not None:
        model.setParam(COPT.Param.TimeLimit, float(timelimit))

    # 5) 建立变量：每条边一个 x_uv
    edges = list(G.edges(data=True))
    x = {}
    for (u, v, data) in edges:
        x[(u, v)] = model.addVar(lb=0.0, ub=1.0, name=f"x_{u}_{v}")

    # 6) 目标函数：min ∑ w_uv x_uv
    obj = cp.quicksum(data.get("weight", 1.0) * x[(u, v)] for (u, v, data) in edges)
    model.setObjective(obj, sense=COPT.MINIMIZE)

    # 7) 流量守恒约束
    for i in G.nodes():
        b_i = 0.0
        if i == source:
            b_i = 1.0
        elif i == target:
            b_i = -1.0

        out_edges = list(G.out_edges(i))
        in_edges = list(G.in_edges(i))

        expr_out = cp.quicksum(x[(i, j)] for (i, j) in out_edges) if out_edges else 0.0
        expr_in = cp.quicksum(x[(k, i)] for (k, i) in in_edges) if in_edges else 0.0

        model.addConstr(expr_out - expr_in == b_i, name=f"flow_{i}")

    # 8) 求解
    model.solve()

    # 9) 读取结果
    if model.status != COPT.OPTIMAL:
        raise RuntimeError(f"COPT 未返回最优解，status={model.status}")

    return float(model.objval)


# ==========================================
# 模块 4: 正确性验证模块
# ==========================================

def verify_correctness(max_edges_show=60):
    """
    正确性验证（用于实验报告展示）：
    1) 生成一个小规模随机连通 ER 图
    2) 在输出节点数等性质之前，先打印该图的边及其权重
    3) 再做：连通性/负权/正权检查 + Dijkstra 与 COPT-LP 一致性检查
    """
    np.random.seed(42)  # 便于复现

    n = 12
    source, target = 0, n - 1

    # 复用 COPT 环境（减少初始化开销）
    try:
        import coptpy as cp
        env = cp.Envr()
    except Exception:
        env = None

    # 1) 生成图
    G = generate_connected_er_graph(n)

    # 2) 打印边及其权重
    print("=== 正确性验证：随机 ER 连通图实例 ===")
    print_graph_edges_with_weights(G, max_edges_show=max_edges_show)

    # 3) 输出图的基本信息（节点数/边数等）
    is_conn = nx.is_connected(G) if G.number_of_nodes() > 0 else True
    weights = [data.get("weight", 1) for (_, _, data) in G.edges(data=True)]
    min_w = min(weights) if weights else None
    max_w = max(weights) if weights else None
    all_positive = (min_w is None) or (min_w > 0)

    # 4) 调用题目要求的检查（连通性 + 负权）
    validity_pass = True
    validity_err = ""
    try:
        check_graph_validity(G)
    except Exception as e:
        validity_pass = False
        validity_err = str(e)

    print("\n--- 图性质检查 ---")
    print(f"节点数 n={n}, source={source}, target={target}")
    print(f"边数 E={G.number_of_edges()}, 是否连通: {is_conn}")
    print(f"边权范围: [{min_w}, {max_w}], 是否全部为正: {all_positive}")
    print(f"check_graph_validity(G): {'通过' if validity_pass else '未通过'}")
    if not validity_pass:
        print(f"  失败原因: {validity_err}")
        # 既然图不合法，后续验证没有意义，直接返回
        print()
        return

    # 5) Dijkstra 结果检查：路径结构 + 路径权重一致性
    dist_d, path = dijkstra_heap(G, source, target)
    ok_path = (len(path) > 0 and path[0] == source and path[-1] == target)

    path_weight = 0.0
    if ok_path:
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            path_weight += float(G.edges[u, v].get("weight", 1.0))
    ok_weight = ok_path and (abs(path_weight - dist_d) <= 1e-9)

    print("\n--- Dijkstra 输出检查 ---")
    print(f"Dijkstra 距离: {dist_d:.6f}")
    print(f"Dijkstra 路径: {path}")
    print(f"路径连通性(首尾)检查: {'OK' if ok_path else 'FAIL'}")
    print(f"路径权重和: {path_weight:.6f}, 与距离一致性: {'OK' if ok_weight else 'FAIL'}")

    # 6) 与 COPT-LP 对比（无向图转有向）
    print("\n--- 与 COPT-LP 对比 ---")
    Gd = G.to_directed()
    dist_lp = solve_shortest_path_lp_copt(Gd, source, target, env=env, timelimit=30.0, verbose=False)
    ok_match = (abs(dist_d - dist_lp) <= 1e-6)
    print(f"COPT-LP 目标值: {dist_lp:.6f}")
    print(f"与 Dijkstra 一致性: {'OK' if ok_match else 'FAIL'}\n")


# ==========================================
# 模块 5: 主实验与性能对比
# ==========================================

def run_experiment():
    # 先做一次正确性验证（只输出一次）
    verify_correctness(max_edges_show=60)

    print("=== 开始 Project 2 实验：Dijkstra vs LP(COPT) ===")

    node_counts = [50, 100, 200, 300, 500]
    trials_per_n = 10

    dijk_mean, dijk_std = [], []
    lp_mean, lp_std = [], []

    # 复用 COPT 环境（减少初始化开销）
    try:
        import coptpy as cp
        env = cp.Envr()
    except Exception:
        env = None

    print(f"{'Nodes':<8} | {'E(avg)':<10} | {'Dijkstra mean±std (s)':<26} | {'COPT-LP mean±std (s)':<26} | {'Check':<8}")
    print("-" * 90)

    for n in node_counts:
        d_times = []
        lp_times = []
        e_list = []
        diff_found = False

        source, target = 0, n - 1

        for _ in range(trials_per_n):
            # 1) 生成随机连通图并检查
            G = generate_connected_er_graph(n)
            check_graph_validity(G)
            e_list.append(G.number_of_edges())

            # 2) Dijkstra 计时
            st = time.time()
            dist_d, _ = dijkstra_heap(G, source, target)
            td = time.time() - st
            d_times.append(td)

            # 3) COPT-LP 计时（无向图转有向）
            Gd = G.to_directed()
            st = time.time()
            dist_lp = solve_shortest_path_lp_copt(Gd, source, target, env=env, timelimit=30.0, verbose=False)
            tlp = time.time() - st
            lp_times.append(tlp)

            # 4) 正确性检查（只记录是否出现过不一致）
            if abs(dist_d - dist_lp) > 1e-6:
                diff_found = True

        # 汇总统计
        d_mean = float(np.mean(d_times))
        d_std = float(np.std(d_times))
        l_mean = float(np.mean(lp_times))
        l_std = float(np.std(lp_times))
        e_avg = float(np.mean(e_list))

        dijk_mean.append(d_mean)
        dijk_std.append(d_std)
        lp_mean.append(l_mean)
        lp_std.append(l_std)

        check_str = "OK" if not diff_found else "DIFF"
        print(f"{n:<8} | {e_avg:<10.1f} | {d_mean:.6f}±{d_std:.6f}{'':<6} | {l_mean:.6f}±{l_std:.6f}{'':<6} | {check_str:<8}")

    # 绘图：均值±标准差
    plt.figure(figsize=(10, 6))
    plt.errorbar(node_counts, dijk_mean, yerr=dijk_std, fmt='-o', capsize=4, label='Dijkstra (heapq)')
    plt.errorbar(node_counts, lp_mean, yerr=lp_std, fmt='-s', capsize=4, label='LP (COPT)')

    plt.title('Performance: Dijkstra vs LP(COPT) on Random Connected ER Graphs')
    plt.xlabel('Number of Nodes (n)')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig('dijkstra_vs_copt_lp.png', dpi=200)

    print("\n对比图已保存为 dijkstra_vs_copt_lp.png（dpi=200）")


if __name__ == "__main__":
    run_experiment()
