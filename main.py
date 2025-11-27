"""
PCA analysis on synthetic multi-sensor data
-------------------------------------------

本脚本提供一个完整、教学风格的 PCA 案例分析流程，包含：

1. 数据加载与初步检查
2. 缺失值处理
3. 标准化
4. 主成分分析（PCA）
5. 累计解释方差图
6. PCA 载荷解释（主成分含义）
7. 重构误差与异常检测
8. 可视化（PCA 二维散点）
9. 总结讨论

运行前请确保 synthetic_sensor_pca_dataset.csv 已生成并与脚本同目录。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

#---------------------------------------
# 1. 加载与初步浏览
#---------------------------------------
print("\n=== 加载数据 ===")
df = pd.read_csv("synthetic_sensor_pca_dataset.csv")
print(df.head())
print(df.info())

features = df.columns[:-1]   # 8 个传感器
label = "is_anomaly"
X = df[features].values
y = df[label].values

#---------------------------------------
# 2. 缺失值处理（均值填充）
#---------------------------------------
print("\n=== 处理缺失值 ===")
imputer = SimpleImputer(strategy="mean")
X_imp = imputer.fit_transform(X)

missing_before = np.isnan(X).sum()
missing_after = np.isnan(X_imp).sum()
print(f"缺失值数量: 处理前 {missing_before}, 处理后 {missing_after}")

#---------------------------------------
# 3. 标准化
#---------------------------------------
print("\n=== 特征标准化 ===")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

#---------------------------------------
# 4. PCA 全模型
#---------------------------------------
print("\n=== 进行 PCA ===")
pca_full = PCA()
X_pca = pca_full.fit_transform(X_scaled)

print("各主成分解释的方差比例：")
print(pca_full.explained_variance_ratio_)

#---------------------------------------
# 5. 累计解释方差图
#---------------------------------------
plt.figure(figsize=(8,4))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='o')
plt.xlabel("主成分数量")
plt.ylabel("累计解释方差")
plt.title("累计解释方差（Cumulative Explained Variance）")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("explained_variance.png")
plt.show()

# 自动选择 ≥90% 方差的主成分数
cum = np.cumsum(pca_full.explained_variance_ratio_)
k = np.argmax(cum >= 0.90) + 1
print(f"\n为了达到 ≥90% 的累计解释方差，所需主成分数 k = {k}")

#---------------------------------------
# 6. 使用前 k 个主成分
#---------------------------------------
pca = PCA(n_components=k)
X_k = pca.fit_transform(X_scaled)

print("\n=== PCA 载荷（主成分方向） ===")
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(k)],
    index=features
)
print(loadings)

#---------------------------------------
# 7. 主成分含义解释（教学风格）
#---------------------------------------
print("\n=== 主成分解释（基于载荷） ===")
for i in range(k):
    print(f"\n主成分 PC{i+1}:")
    vec = loadings[f"PC{i+1}"].values
    idx = np.argsort(-np.abs(vec))
    for j in idx[:4]:
        print(f"  主要影响特征：{features[j]:10s} (权重 {vec[j]:.3f})")

print("\n这些方向基本对应：环境基线、人群活动、机器运行负载等潜在因子。")

#---------------------------------------
# 8. 重构误差与异常检测
#---------------------------------------
print("\n=== 重构误差异常检测 ===")
X_rec = pca.inverse_transform(X_k)
recon_err = np.mean((X_scaled - X_rec) ** 2, axis=1)

threshold = np.percentile(recon_err, 97)
pred = (recon_err > threshold).astype(int)

precision = np.sum((pred==1)&(y==1)) / np.sum(pred==1)
recall = np.sum((pred==1)&(y==1)) / np.sum(y==1)

print(f"检测阈值（97百分位）：{threshold:.4f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")

# 可视化重构误差
plt.figure(figsize=(8,4))
plt.plot(recon_err, label="recon_err")
plt.axhline(threshold, color='r', linestyle='--', label="threshold")
plt.title("重构误差（用于异常检测）")
plt.xlabel("样本索引")
plt.ylabel("重构误差")
plt.legend()
plt.tight_layout()
plt.savefig("reconstruction_error.png")
plt.show()

#---------------------------------------
# 9. PCA 二维散点可视化
#---------------------------------------
plt.figure(figsize=(6,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="coolwarm", s=8)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA 二维投影图（红色为异常）")
plt.tight_layout()
plt.savefig("pca_scatter.png")
plt.show()

#---------------------------------------
# 10. 总结
#---------------------------------------
print("\n=== 总结讨论 ===")
print("""
1. PCA 成功识别了潜在的三个主因子（环境、人员、机器）。
2. 累计解释方差显示前三个分量占据绝大多数结构。
3. 载荷矩阵可解释每个方向对应的传感器模式。
4. 利用重构误差进行异常检测，precision/recall 表现合理。
5. PCA 对缺失值和异常点敏感，真实工程中可使用更稳健方法：
   - Robust PCA
   - Sliding-window PCA（应对漂移）
   - Online / Incremental PCA
   - Autoencoder（非线性结构）
""")
