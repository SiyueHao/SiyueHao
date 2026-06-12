# test_qiskit.py
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

# 1. 创建一个 1 量子比特 + 1 经典比特的量子电路
qc = QuantumCircuit(1, 1)

# 2. 对量子比特 q0 施加一个 Hadamard 门（H 门）
qc.h(0)

# 3. 测量：把量子比特 q0 的结果测量到经典比特 c0 上
qc.measure(0, 0)

# 打印电路结构看看（认识一下长什么样）
print("量子电路结构：")
print(qc)

# 4. 用本地模拟器执行这个电路
sim = AerSimulator()
job = sim.run(qc, shots=1000)  # 重复运行 1000 次
result = job.result()
counts = result.get_counts()

print("测量统计结果：")
print(counts)
