from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity
from math import pi

# 1. 构造 3 比特 QFT 电路
def qft_3():
    qc = QuantumCircuit(3, name="QFT3")
    # 注意：这里假设比特顺序为 q0 是最低位
    qc.h(0)
    qc.cp(pi/2, 1, 0)
    qc.cp(pi/4, 2, 0)

    qc.h(1)
    qc.cp(pi/2, 2, 1)

    qc.h(2)

    # 交换比特，修正比特顺序（QFT 标准做法）
    qc.swap(0, 2)
    return qc

# 2. 逆 QFT：直接取电路的逆
def iqft_3():
    return qft_3().inverse()

if __name__ == "__main__":
    for x in range(8):  # 遍历 000~111
        print(f"\n=== 输入基态 |{x:03b}> ===")
        # 准备输入电路
        prep = QuantumCircuit(3)
        for i in range(3):      # 第 i 个比特是否置 1
            if (x >> i) & 1:
                prep.x(i)

        # 初始态向量
        state_in = Statevector.from_instruction(prep)

        # 加上 QFT
        qc_qft = prep.compose(qft_3())
        state_after_qft = Statevector.from_instruction(qc_qft)

        # 再加逆 QFT
        qc_full = qc_qft.compose(iqft_3())
        state_after_iqft = Statevector.from_instruction(qc_full)

        # 计算保真度
        fid = state_fidelity(state_in, state_after_iqft)

        print("初始态向量：", state_in.data)
        print("QFT 后态向量：", state_after_qft.data)
        print("逆 QFT 后态向量：", state_after_iqft.data)
        print("恢复保真度：", fid)
