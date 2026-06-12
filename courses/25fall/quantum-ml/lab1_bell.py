# bell.py

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# 1. 创建 2 个量子比特 + 2 个经典比特的量子电路
qc = QuantumCircuit(2, 2)

# 2. 在 q0 上施加 H 门
qc.h(0)

# 3. 施加 CNOT 门，控制位是 q0，目标位是 q1
qc.cx(0, 1)  # 等价于 CNOT(q0, q1)

# 4. 测量：把 q0 测到 c0，把 q1 测到 c1
qc.measure(0, 0)
qc.measure(1, 1)

# 打印电路结构，看一下是不是符合“先 H on q0，再 CNOT(q0, q1)，再测量”
print("Bell 态量子电路：")
print(qc)

# 5. 用本地模拟器运行
sim = AerSimulator()
job = sim.run(qc, shots=1000)  # 重复实验 1000 次
result = job.result()
counts = result.get_counts()

print("测量统计结果 (1000 次)：")
print(counts)
