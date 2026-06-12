import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# 1. 创建量子设备：1 个量子比特
dev = qml.device("default.qubit", wires=1)

# 2. 构造简单二分类数据集
# X: 输入特征，Y: 标签
X = np.array([[0.1], [1.0]])   # 形状为 (2, 1)
Y = np.array([0, 1])           # 标签 0 和 1

# 3. 定义参数化量子电路（PQC）
@qml.qnode(dev)
def circuit(x, theta):
    """
    x     : 标量输入（这里是一维）
    theta : 长度为 3 的可训练参数向量
    """
    # 数据嵌入
    qml.RX(x, wires=0)
    # 参数化量子线路（Ansatz）
    qml.Rot(theta[0], theta[1], theta[2], wires=0)
    # 测量 Pauli-Z 的期望值，范围为 [-1, 1]
    return qml.expval(qml.PauliZ(0))

# 4. 定义损失函数（MSE）
def loss(theta):
    # 对每个样本分别调用量子电路，得到预测值
    preds = [(circuit(x, theta) + 1) / 2 for x in X]  # 映射到 [0, 1]
    preds = np.array(preds)
    # 均方误差
    return np.mean((preds - Y) ** 2)

# 5. 初始化参数和优化器
np.random.seed(42)
theta = np.random.randn(3)  # 随机初始化 3 个参数
opt = qml.GradientDescentOptimizer(stepsize=0.3)

# 训练轮数
num_epochs = 50

# 用于记录每一轮的损失值
loss_history = []

# 6. 训练循环
for epoch in range(num_epochs):
    theta = opt.step(loss, theta)           # 更新参数
    current_loss = loss(theta)             # 当前损失
    loss_history.append(current_loss)      # 记录损失

    if epoch % 10 == 0:
        print(f"Epoch {epoch:2d}: loss = {current_loss:.4f}")

print("\n训练完成，最终参数 theta =", theta)

# 7. 查看最终预测效果
preds = [(circuit(x, theta) + 1) / 2 for x in X]
print("最终预测值（接近 0 或 1 越好）:", preds)

# 8. 绘制训练损失曲线
plt.figure(figsize=(6,4))
plt.plot(range(num_epochs), loss_history, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training Loss Curve of Variational Quantum Classifier")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()