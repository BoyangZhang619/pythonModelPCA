import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from data_generation import generate_data as gd
from generalPackage.FormulaImageGenerator import FormulaImageGenerator

# base settings
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "csv")
IMG_DIR = os.path.join(BASE_DIR, "img")
origin_print_ = print
def print(*args, **kwargs):
    origin_print_(f"{'='*60}\n",*args, **kwargs)


# 创建数据
try:
    gd(ranSeed=random.randint(1,100))
except Exception as e:
    print(e)
    exit()
print("创建数据成功")

# 加载数据
data = pd.read_csv(os.path.join(CSV_DIR,"synthetic_sensor_pca_dataset.csv"))
print(data.head())
print(data.shape)
features = data.columns[:-1]
target = data.columns[-1]
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
print(data.columns)

# 数据处理
imputer = SimpleImputer(strategy="mean")
X_imp = imputer.fit_transform(X)
print(f"缺失值|处理前:{np.isnan(X).sum()},处理后:{np.isnan(X_imp).sum()}(数据生成上好像未添加nan值)")

# 标准化
scaler = StandardScaler()
X_std = scaler.fit_transform(X_imp)
print(X_std.shape)
print(X_std[:5,:])
print("standarded")

# 进行PCA
pca_full = PCA()
X_pca = pca_full.fit_transform(X_std)
pfevr = pca_full.explained_variance_ratio_
print(pfevr)

# 画图,累计解释方差
x = np.arange(1,len(pfevr)+1)
fig, ax1 = plt.subplots(figsize=(20,10))
ax1.plot(x, np.cumsum(pfevr), color='skyblue', label='累计解释方差值',marker="o")
ax1.set_ylabel('累计解释方差值', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax1.set_yticks([round(i/10,1) for i in range(11)])
ax1.set_xlabel("主成分数量")
ax2 = ax1.twinx()  # 创建共享 x 轴的新 y 轴
ax2.plot(x, pfevr, color='pink', label='解释方差值',marker="o")
ax2.set_ylabel('解释方差值', color='pink')
ax2.tick_params(axis='y', labelcolor='pink')
ax2.set_yticks([round(i/100,2) for i in range(int(pfevr[0]*100+2))])
plt.grid(color="skyblue",alpha=0.3)
plt.xticks([i for i in range(1,21)])
plt.title("累计解释方差(Cumulative Explained Variance)")
plt.savefig(os.path.join(IMG_DIR,"explained_variance.png"))
plt.show()

# 获取主要成分数(对应累计解释方差>=.9,可变)
k = np.argmax(np.cumsum(pfevr) >= .9) + 1
print(k)

# 取前k个主要特征来新建
pca = PCA(n_components=k)
X_k = pca.fit_transform(X_std)
print(X_k.shape)

loadings = pd.DataFrame(
    pca.components_.T,
    index=features,
    columns=[f"PC{i}" for i in range(1,k+1)]
)
print(loadings.shape)
print(loadings)


# 前4个主成分的对应各个主要影响特征
for i in range(k):
    print(f"\n主成分 PC{i+1}:")
    vec = loadings[f"PC{i+1}"].values
    idx = np.argsort(-np.abs(vec))
    for o,j in enumerate(idx):
        print(f"  主要影响特征{o}：{features[j]:10s} (权重 {vec[j]:.3f})")


# 1. 计算 squared loadings（载荷平方）
sq_loadings = loadings.iloc[:, :k] ** 2   # shape = (n_features, k)
# 2. 加权：loading^2 * EVR
weighted = sq_loadings * pfevr[:k]
# 3. 每个特征的总贡献度
feature_contrib = weighted.sum(axis=1)
# 放入 DataFrame 排序
contrib_df = pd.DataFrame({
    "feature": features,
    "contribution": feature_contrib.values
}).sort_values("contribution", ascending=False)
print("\n=== 特征贡献度排序（越高越重要） ===")
for i, row in contrib_df.iterrows():
    print(f"{row['feature']:12s} 贡献度 = {row['contribution']:.4f}")
plt.figure(figsize=(8,5))
plt.bar(contrib_df["feature"], contrib_df["contribution"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("特征贡献度")
plt.title("基于PCA的特征贡献度")
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(IMG_DIR,"contributionOfFeaturesOfPCA.png"))

# 重构误差与异常检测
print("\n=== 重构误差异常检测 ===")
X_rec = pca.inverse_transform(X_k)
recon_err = np.mean((X_std - X_rec) ** 2, axis=1)

threshold = np.percentile(recon_err, 90)
pred = (recon_err > threshold).astype(int)

precision = np.sum((pred==1)&(y==1)) / np.sum(pred==1)
recall = np.sum((pred==1)&(y==1)) / np.sum(y==1)

print(f"检测阈值（90百分位）：{threshold:.4f}")
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
plt.savefig(os.path.join(IMG_DIR,"reconstruction_error.png"))
plt.show()


# PCA 二维散点可视化
plt.figure(figsize=(5,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="coolwarm", s=4)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA 二维投影图（红色为异常）")
plt.tight_layout()
plt.axis('equal')
plt.savefig(os.path.join(IMG_DIR,"pca_scatter.png"))
plt.show()

formulas = {"标准化 (Standardization)": {"formula": r"z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}","explain": {r"x_{ij}": "第 i 个样本的第 j 个特征值",r"\mu_j": "第 j 个特征的平均值",r"\sigma_j": "第 j 个特征的标准差",r"z_{ij}": "标准化后的数值"}},"解释方差比 (Explained Variance Ratio)": {"formula": r"EVR_i = \frac{\lambda_i}{\sum_{j=1}^{n} \lambda_j}","explain": {r"\lambda_i": "第 i 个主成分的特征值（方差大小）",r"\sum_{j=1}^{n} \lambda_j": "全部特征值之和（总方差）",r"EVR_i": "第 i 个主成分的解释方差比"}},"主成分选取 (k-Value Determination)": {"formula": r"\sum_{i=1}^{k} EVR_i \geq \mathrm{Threshold}","explain": {r"EVR_i": "第 i 个主成分的解释方差比",r"k": "所选主成分数量",r"Threshold": "设定阈值（如 0.85 或 0.95）"}},"数据重构 (Data Reconstruction)": {"formula": r"\mathbf{X}_{rec} = \mathbf{T} \mathbf{P}_k^T","explain": {r"\mathbf{T}": "主成分得分矩阵（投影后数据）",r"\mathbf{P}_k": "前 k 个主成分载荷矩阵",r"\mathbf{X}_{rec}": "重构后的数据矩阵"}},"重构误差与 MSE (Reconstruction Error and MSE)": {"formula": r"MSE_i = \frac{1}{n} \sum_{j=1}^{n} (x_{ij} - \hat{x}_{ij})^2","explain": {r"x_{ij}": "原始数据的真实值",r"\hat{x}_{ij}": "重构后的估计值",r"n": "特征数量",r"MSE_i": "第 i 个样本的均方误差"}},"特征贡献度 (Feature Contribution)": {"formula": r"Contrib_j = \sum_{i=1}^{k} L_{ji}^2 \cdot EVR_i","explain": {r"L_{ji}": "特征 j 在第 i 个主成分上的载荷（loading）",r"EVR_i": "第 i 个主成分的解释方差比",r"Contrib_j": "特征 j 的总贡献度"}}}
generator = FormulaImageGenerator(output_dir="latex_img_clean")
generator.generate_batch(formulas)