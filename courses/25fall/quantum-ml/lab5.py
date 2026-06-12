"""
可直接运行的代码：4x4 手写数字二分类（0 vs 1）的变分量子分类器 VQC（PennyLane）

特点：
- 不需要联网下载 MNIST（使用 sklearn 内置 digits 数据集）
- 将 8x8 图像降采样到 4x4 -> 16 维特征
- 4 量子比特振幅编码（AmplitudeEmbedding）
- StronglyEntanglingLayers 作为 Ansatz
- 交叉熵损失 + Adam 优化 + 训练曲线可视化
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# -----------------------------
# 1) 数据加载与预处理：digits(8x8) -> downsample(4x4) -> flatten(16)
# -----------------------------
digits = load_digits()
X = digits.images  # (1797, 8, 8)
y = digits.target  # (1797,)

# 只取 0 vs 1
mask = (y == 0) | (y == 1)
X = X[mask]
y = y[mask].astype(np.int64)

# 归一化到 [0,1]（digits 像素范围通常是 0~16）
X = X / 16.0

def downsample_8x8_to_4x4(img8):
    # img8: (8,8) -> (4,2,4,2) -> 对 2x2 块取平均 -> (4,4)
    return img8.reshape(4, 2, 4, 2).mean(axis=(1, 3))

X4 = np.array([downsample_8x8_to_4x4(img) for img in X])   # (N,4,4)
X_feat = X4.reshape(len(X4), 16)                            # (N,16)

# 为了 amplitude embedding 更稳定：加一个很小的常数避免全零向量（一般不会出现，但更稳）
X_feat = X_feat + 1e-8

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_feat, y, test_size=0.25, random_state=42, stratify=y
)

# -----------------------------
# 2) 量子模型：4 qubits, AmplitudeEmbedding(16维), StronglyEntanglingLayers
# -----------------------------
n_qubits = 4
n_layers = 2

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def circuit(x, weights):
    # x: shape (16,)
    qml.AmplitudeEmbedding(x, wires=range(n_qubits), normalize=True)

    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    # 用第 0 个比特的 Z 期望做输出
    return qml.expval(qml.PauliZ(0))

def predict_proba(X_batch, weights):
    # 输出 p in (0,1)
    z = np.array([circuit(x, weights) for x in X_batch])         # in [-1,1]
    p = (z + 1.0) / 2.0                                          # in [0,1]
    # 数值稳定
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return p

def cross_entropy_loss(X_batch, y_batch, weights):
    p = predict_proba(X_batch, weights)
    yb = y_batch.astype(np.float64)
    return -np.mean(yb * np.log(p) + (1 - yb) * np.log(1 - p))

def accuracy(X_eval, y_eval, weights):
    p = predict_proba(X_eval, weights)
    y_pred = (p >= 0.5).astype(np.int64)
    return np.mean(y_pred == y_eval)

# -----------------------------
# 3) 训练设置
# -----------------------------
np.random.seed(0)
weights = 0.01 * np.random.randn(n_layers, n_qubits, 3, requires_grad=True)

opt = qml.AdamOptimizer(stepsize=0.08)

epochs = 30
batch_size = 16

loss_history = []
train_acc_history = []
test_acc_history = []

# -----------------------------
# 4) 训练循环（mini-batch）
# -----------------------------
n_train = len(X_train)

for ep in range(epochs):
    # 打乱
    idx = np.random.permutation(n_train)
    X_train_shuf = X_train[idx]
    y_train_shuf = y_train[idx]

    # mini-batch 更新
    for start in range(0, n_train, batch_size):
        end = min(start + batch_size, n_train)
        Xb = X_train_shuf[start:end]
        yb = y_train_shuf[start:end]

        weights = opt.step(lambda w: cross_entropy_loss(Xb, yb, w), weights)

    # 每个 epoch 记录指标
    train_loss = cross_entropy_loss(X_train, y_train, weights)
    train_acc = accuracy(X_train, y_train, weights)
    test_acc = accuracy(X_test, y_test, weights)

    loss_history.append(train_loss)
    train_acc_history.append(train_acc)
    test_acc_history.append(test_acc)

    print(f"Epoch {ep:02d} | loss={train_loss:.4f} | train_acc={train_acc:.3f} | test_acc={test_acc:.3f}")

print("\n训练完成。")
print("最终测试准确率:", float(test_acc_history[-1]))

# -----------------------------
# 5) 可视化：loss 曲线 & accuracy 曲线
# -----------------------------
plt.figure(figsize=(6,4))
plt.plot(range(epochs), loss_history, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training Loss Curve (VQC on 4x4 Digits 0 vs 1)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(range(epochs), train_acc_history, marker="o", label="train_acc")
plt.plot(range(epochs), test_acc_history, marker="o", label="test_acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve (VQC on 4x4 Digits 0 vs 1)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
