# -*- coding: utf-8 -*-
# submission.py 提交的代码
import os
import numpy as np
from sklearn.cluster import KMeans

def data_preprocess(example: np.ndarray) -> np.ndarray:
    """
    TODO: 完成数据预处理，需要返回处理后的example字典
    """
    img_np = np.array(example["image"], dtype=np.uint8)

    # 数据预处理
    # 归一化到 [0, 1]，防止数值过大
    img_np = img_np / 255.0
    # 标准化：减均值除以标准差
    mean = img_np.mean()
    std = img_np.std() + 1e-8
    img_np = (img_np - mean) / std

    # TODO: 1. 将图像扩展维度 (28,28) -> (1,28,28)
    img_np_with_channel = img_np[np.newaxis, :, :] # 为AutoEncoder做准备
    
    # TODO: 2. 将图像展平为一维数组 (28,28) -> (784,)
    img_np_flat = img_np.reshape(-1).astype(np.float32) # 为PCA做准备
    
    # TODO: 3. 将处理后的数据添加到example字典中
    # example["image2D"] = ...
    # example["image1D"] = ...
    example ["image2D"]= img_np_with_channel # (1, 28, 28)
    example ["image1D"]= img_np_flat       # (784,)

    # TODO: 4.（可选）进行相关数据预处理，例如：归一化、标准化等
  

    return example

class PCA:
    """
    使用奇异值分解（SVD）实现的简易PCA类

    属性
    ----
    mean_ : np.ndarray
        训练数据每个特征的均值，形状为 (D,)。
    components_ : np.ndarray
        主成分方向，形状为 (n_components, D)。
    explained_variance_ : np.ndarray
        每个主成分的方差解释量，形状为 (n_components,)。
    explained_variance_ratio_ : np.ndarray
        每个主成分的方差贡献比例。
    """

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components
        self.mean_: np.ndarray = None  # type: ignore
        self.components_: np.ndarray = None  # type: ignore
        self.explained_variance_: np.ndarray = None  # type: ignore
        self.explained_variance_ratio_: np.ndarray = None  # type: ignore

    def fit(self, X: np.ndarray) -> "PCA":
        """Fit the model with X.

        Parameters
        ----------
        X : np.ndarray of shape (N, D)
            Input data.
        """
        # 确保是 numpy 数组，且为 float 类型
        X = np.asarray(X, dtype=np.float32)

        # 样本数 N、特征维度 D
        n_samples, n_features = X.shape

        # 1. 计算每一维的均值，并中心化
        # mean_: 形状 (D,)
        self.mean_ = X.mean(axis=0) # axis=0表示对每一列求平均值
        X_centered = X - self.mean_  # 中心化

        # 2. 对中心化后的数据做 SVD
        # X_centered = U S Vt
        # U: (N, D), S: (min(N, D),), Vt: (D, D)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # 3. 主成分方向：Vt 的前 n_components 行
        # components_: (n_components, D)
        self.components_ = Vt[: self.n_components]

        # 4. 计算每个主成分的方差（特征值）
        # 根据 PCA 理论，特征值 = S^2 / (N - 1)
        eigenvalues = (S ** 2) / (n_samples - 1)  # 形状 (min(N,D),)

        # 取前 n_components 个特征值作为 explained_variance_
        self.explained_variance_ = eigenvalues[: self.n_components]

        # 5. 计算方差占比 explained_variance_ratio_
        total_var = eigenvalues.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the dimensionality reduction on X."""
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA must be fitted before calling transform().")
        Xc = X - self.mean_
        return Xc @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the model with X and apply the dimensionality reduction on X."""
        return self.fit(X).transform(X)

    def save_pretrained(self, path: str) -> None:
        """
        保存PCA参数到文件（npz格式）
        """
        np.savez_compressed(
            path,
            n_components=self.n_components,
            mean_=self.mean_,
            components_=self.components_,
            explained_variance_=self.explained_variance_,
            explained_variance_ratio_=self.explained_variance_ratio_
        )

    @classmethod
    def from_pretrained(cls, path: str) -> "PCA":
        """
        从文件加载PCA参数，返回PCA实例
        """
        data = np.load(path)
        n_components = int(data["n_components"]) if "n_components" in data else data["components_"].shape[0]
        obj = cls(n_components)
        obj.mean_ = data["mean_"]
        obj.components_ = data["components_"]
        obj.explained_variance_ = data["explained_variance_"]
        obj.explained_variance_ratio_ = data["explained_variance_ratio_"]
        return obj
    
class GMM:
    """高斯混合模型（EM）。使用全协方差，数值稳定性通过对角线正则项控制。

    参数
    ----
    n_components : int
        混合成分数量。
    max_iter : int
        最大 EM 迭代次数。
    tol : float
        对数似然相对改变量小于该阈值则停止。
    reg_covar : float
        协方差对角线正则，防止矩阵奇异。
    random_state : int
        随机种子。
    init_kmeans : bool
        是否用 KMeans 初始化均值和权重。
    """

    def __init__(self, n_components: int = 10, max_iter: int = 100, tol: float = 1e-3, reg_covar: float = 1e-6, random_state: int = 42, init_kmeans: bool = True) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.init_kmeans = init_kmeans
        # learned params
        self.weights_: np.ndarray | None = None  # (K,)
        self.means_: np.ndarray | None = None    # (K, D)
        self.covariances_: np.ndarray | None = None  # (K, D, D)
        self.converged_: bool = False
        self.n_iter_: int = 0
        self.lower_bound_: float | None = None

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.random_state)

    def _estep(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """
        E-step: 计算后验概率（responsibilities）
        
        输入:
            X: 数据矩阵 (N, D)
            
        返回:
            resp: 后验概率矩阵 (N, K)，resp[i,k] 表示样本i属于聚类k的概率
            lower_bound: 下界值（用于判断收敛）
        
        TODO: 实现E-step
        1. 计算每个样本在每个高斯分布下的对数概率
        2. 使用log-sum-exp技巧计算归一化的后验概率
        3. 返回responsibilities和下界值
        """
        N, D = X.shape
        K = self.n_components

        # log_prob[n,k] = log π_k + log N(x_n | μ_k, Σ_k)
        log_prob = np.empty((N, K), dtype=np.float64)

        for k in range(K):
            diff = X - self.means_[k]      # (N, D)
            cov = self.covariances_[k]
            cov_inv = np.linalg.inv(cov)
            det = np.linalg.det(cov)

            # log Gaussian density
            # log N(x|μ,Σ) = -1/2[(x-μ)^T Σ^{-1} (x-μ) + log det(Σ) + D log(2π)]
            exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
            norm_const = -0.5 * (np.log(det + 1e-12) + D * np.log(2 * np.pi))

            log_gauss = norm_const + exponent

            # 加上 log π_k
            log_prob[:, k] = np.log(self.weights_[k] + 1e-12) + log_gauss

        # logsumexp：防止溢出
        log_prob_max = np.max(log_prob, axis=1, keepdims=True)
        log_sum = log_prob_max + np.log(np.sum(np.exp(log_prob - log_prob_max), axis=1, keepdims=True))

        # responsibilities
        log_resp = log_prob - log_sum
        resp = np.exp(log_resp)   # (N, K)

        # lower bound (log likelihood)
        lower_bound = np.sum(log_sum)

        return resp, lower_bound

    def _mstep(self, X: np.ndarray, resp: np.ndarray) -> None:
        """
        M-step: 更新模型参数
        
        输入:
            X: 数据矩阵 (N, D)
            resp: 后验概率矩阵 (N, K)
        
        TODO: 实现M-step
        更新以下属性：
        1. self.weights_: 每个聚类的权重（先验概率）
        2. self.means_: 每个高斯分布的均值向量
        3. self.covariances_: 每个高斯分布的协方差矩阵
        """
        N, D = X.shape
        K = self.n_components

        # 每个簇的“责任度和” Nk
        Nk = resp.sum(axis=0)  # (K,)

        # 更新权重 π_k
        self.weights_ = Nk / N

        # 更新均值 μ_k
        self.means_ = (resp.T @ X) / Nk[:, np.newaxis]

        # 更新协方差 Σ_k
        covariances = []
        for k in range(K):
            diff = X - self.means_[k]       # (N, D)
            weighted = resp[:, k][:, np.newaxis] * diff
            cov_k = (weighted.T @ diff) / Nk[k]

            # 数值稳定性：加一点对角正则化
            cov_k += np.eye(D) * self.reg_covar
            covariances.append(cov_k)

        self.covariances_ = np.array(covariances)
       

    def fit(self, X: np.ndarray) -> "GMM":
        rng = self._rng()
        X = np.asarray(X, dtype=np.float64)
        N, D = X.shape
        # init
        if self.init_kmeans:
            rng_for_kmeans = np.random.RandomState(self.random_state)
            km = KMeans(n_clusters=self.n_components, n_init=5, random_state=rng_for_kmeans)
            labels = km.fit_predict(X)
            self.means_ = km.cluster_centers_.astype(np.float64)
            self.weights_ = np.array([(labels == k).mean() + 1e-12 for k in range(self.n_components)], dtype=np.float64)
            self.weights_ /= self.weights_.sum()
            self.covariances_ = np.stack([np.cov(X[labels == k].T) if np.any(labels == k) else np.eye(D) for k in range(self.n_components)], axis=0).astype(np.float64)
            for k in range(self.n_components):
                if not np.all(np.isfinite(self.covariances_[k])):
                    self.covariances_[k] = np.eye(D)
        else:
            self.means_ = X[rng.choice(N, size=self.n_components, replace=False)]
            self.weights_ = np.ones(self.n_components, dtype=np.float64) / self.n_components
            self.covariances_ = np.stack([np.eye(D) for _ in range(self.n_components)], axis=0)

        prev_lower = -100000 # 修改了prev_lower
        for it in range(1, self.max_iter + 1):
            resp, lower = self._estep(X)
            self._mstep(X, resp)
            self.n_iter_ = it
            improvement = (lower - prev_lower) / (abs(prev_lower) + 1e-12)
            if improvement < self.tol:
                self.converged_ = True
                break
            prev_lower = lower
        self.lower_bound_ = lower
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.means_ is None:
            raise RuntimeError("GMM must be fitted before calling predict_proba().")
        X = np.asarray(X, dtype=np.float64)
        resp, _ = self._estep(X)
        return resp

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
    
    def save_pretrained(self, path) -> None:
        """
        保存GMM模型到指定路径（兼容HuggingFace风格）
        """
        import os
        import pickle
        os.makedirs(path, exist_ok=True)
        
        # 保存模型参数
        model_data = {
            'n_components': self.n_components,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'reg_covar': self.reg_covar,
            'random_state': self.random_state,
            'init_kmeans': self.init_kmeans,
            'weights_': self.weights_,
            'means_': self.means_,
            'covariances_': self.covariances_,
            'converged_': self.converged_,
            'n_iter_': self.n_iter_,
            'lower_bound_': self.lower_bound_
        }
        
        with open(os.path.join(path, "gmm_model.pkl"), "wb") as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def from_pretrained(cls, path) -> "GMM":
        """
        从指定路径加载GMM模型
        """
        import pickle
        with open(os.path.join(path, "gmm_model.pkl"), "rb") as f:
            model_data = pickle.load(f)
        
        gmm = cls(
            n_components=model_data['n_components'],
            max_iter=model_data['max_iter'],
            tol=model_data['tol'],
            reg_covar=model_data['reg_covar'],
            random_state=model_data['random_state'],
            init_kmeans=model_data['init_kmeans']
        )
        
        # 恢复训练后的参数
        gmm.weights_ = model_data['weights_']
        gmm.means_ = model_data['means_']
        gmm.covariances_ = model_data['covariances_']
        gmm.converged_ = model_data['converged_']
        gmm.n_iter_ = model_data['n_iter_']
        gmm.lower_bound_ = model_data['lower_bound_']
        
        return gmm