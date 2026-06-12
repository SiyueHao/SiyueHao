# ============================
# 环境与工具
# ============================
import math
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

sim = AerSimulator()

# 小工具：运行电路并返回计数
def run_counts(circ, shots=4096):
    c = transpile(circ, sim)
    result = sim.run(c, shots=shots).result()
    return result.get_counts()

# ============================
# 实验 1：H -> Rz(theta) -> H
# ============================
def circuit_H_Rz_H(theta: float) -> QuantumCircuit:
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.rz(theta, 0)
    qc.h(0)
    qc.measure(0, 0)
    return qc


if __name__ == "__main__":
    # 演示：扫描一组theta，记录P(0)、P(1)
    thetas = np.linspace(0, 2 * np.pi, 13)
    p0_list, p1_list = [], []

    for t in thetas:
        qc = circuit_H_Rz_H(float(t))
        counts = run_counts(qc, shots=4096)
        p0_list.append(counts.get('0', 0) / 4096)
        p1_list.append(counts.get('1', 0) / 4096)

    # 画概率曲线
    plt.figure(figsize=(7, 4), dpi=150)
    plt.plot(thetas, p0_list, marker='o', label='P(0)')
    plt.plot(thetas, p1_list, marker='s', label='P(1)')
    plt.xlabel('theta (rad)')
    plt.ylabel('Probability')
    plt.title('H → Rz(θ) → H measurement probabilities')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("lab2_theta_curve.png")

    # 具体θ=π/2的直方图
    qc_demo = circuit_H_Rz_H(np.pi / 2)
    counts_demo = run_counts(qc_demo)
    fig = plot_histogram(counts_demo, title='θ=π/2 histogram')
    fig.savefig("lab2_theta_pi_over_2.png")

    print("理论期望：P(0)=cos^2(θ/2), P(1)=sin^2(θ/2)。")

    # ============================
    # 实验 2：Deutsch 算法
    # ============================
    def oracle_f0(qc, q0, q1):
        return

    def oracle_f1(qc, q0, q1):
        qc.x(q1)

    def oracle_fid(qc, q0, q1):
        qc.cx(q0, q1)

    def oracle_fnot(qc, q0, q1):
        qc.x(q0)
        qc.cx(q0, q1)
        qc.x(q0)

    def deutsch_circuit(oracle_func, shots=1024):
        q0q1 = QuantumRegister(2, 'q')
        c = ClassicalRegister(1, 'c')
        qc = QuantumCircuit(q0q1, c)

        qc.x(q0q1[1])
        qc.h(q0q1[0])
        qc.h(q0q1[1])

        oracle_func(qc, q0q1[0], q0q1[1])

        qc.h(q0q1[0])
        qc.measure(q0q1[0], c[0])

        return qc, run_counts(qc, shots=shots)


    cases = {
        'constant_f0': oracle_f0,
        'constant_f1': oracle_f1,
        'balanced_fid': oracle_fid,
        'balanced_fnot': oracle_fnot,
    }

    for name, f in cases.items():
        qc, counts = deutsch_circuit(f, shots=2048)
        fig = plot_histogram(counts, title=name)
        fig.savefig(f"lab2_{name}.png")
        print(name, counts)

    print("\n判别规则：测到 0 → 常数；测到 1 → 平衡。")
