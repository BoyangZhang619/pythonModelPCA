"""
生成可用于 PCA 的合成传感器监控数据集。
输出: pandas.DataFrame (n_samples x 8)，并保存为 CSV。
可直接把数据送入 sklearn.decomposition.PCA 做分析。
"""
import os
import numpy as np
import pandas as pd
def generate_data(ranSeed=42, n_samples=1000, n_sensors=8, latent_dim=3):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_DIR = os.path.join(BASE_DIR, "csv")
    np.random.seed(ranSeed)
    # latent_dim 真正影响传感器的潜在因子数

    # 1) 生成时间序列型的潜在因子（带慢速漂移 + 活动波动）
    t = np.arange(n_samples)
    # 因子1：环境基线（慢漂移）
    f1 = 0.5 * np.sin(2 * np.pi * t / 1000) + 0.002 * t
    # 因子2：人员活动（短期脉冲）
    f2 = 2.0 * (np.sin(2 * np.pi * t / 50) + 0.3 * np.random.randn(n_samples))
    # 因子3：机器负载（突变 + 中速波动）
    f3 = 1.0 * np.cos(2 * np.pi * t / 200) + 0.5 * (np.random.rand(n_samples) > 0.995).astype(float) * 10.0

    # 组合成 latent matrix
    F = np.vstack([f1, f2, f3]).T  # shape (n_samples, latent_dim)

    # 2) 随机构造一个 loading 矩阵（每个传感器对 latent 的敏感度）
    #    设计成某些传感器主要受某一因子影响以便可解释
    loadings = np.array([
        [1.0, 0.2, 0.1],   # 温度：主受环境基线影响
        [0.9, 0.3, 0.05],  # 湿度：与温度相关
        [0.7, 0.1, 0.4],   # 压强：混合影响
        [0.2, 1.2, 0.1],   # CO2：人员活动主导
        [0.1, 0.8, 0.2],   # 光照：与人员活动相关
        [0.05, 0.1, 1.1],  # 振动：机器负载主导
        [0.02, 0.2, 0.9],  # 声音：与机器相关
        [0.05, 1.0, 0.05], # 人员计数：强受人员活动影响
    ])  # shape (8,3)

    # 3) 生成观测：线性混合 + 高斯噪声
    noise_sigma = 0.5
    X_clean = F @ loadings.T  # (n_samples, 8)
    X = X_clean + noise_sigma * np.random.randn(n_samples, n_sensors)

    # 4) 注入异常（稀疏突发型），并记录其索引
    n_anoms = int(0.03 * n_samples)  # 3% 异常
    anom_idx = np.random.choice(n_samples, n_anoms, replace=False)
    # 对异常样本注入较大幅度的偏移到部分传感器上
    for i in anom_idx:
        sensor_pick = np.random.choice(n_sensors, size=np.random.randint(1,4), replace=False)
        X[i, sensor_pick] += np.random.randn(len(sensor_pick)) * np.array([8.0, 10.0, 6.0])[:len(sensor_pick)]

    # 5) 注入少量缺失值（随机）
    missing_frac = 0.02
    n_missing = int(missing_frac * n_samples * n_sensors)
    missing_positions = (
        np.random.randint(0, n_samples, n_missing),
        np.random.randint(0, n_sensors, n_missing)
    )
    X[missing_positions] = np.nan

    # 6) 保存为 DataFrame，带上列名和标签（是否异常的 ground truth）
    col_names = ["temp", "humidity", "pressure", "co2", "light", "vibration", "sound", "occupancy"]
    df = pd.DataFrame(X, columns=col_names)
    df["is_anomaly"] = 0
    df.loc[anom_idx, "is_anomaly"] = 1

    # 7) 可选：保存 CSV
    df.to_csv(os.path.join(BASE_DIR,"./csv/synthetic_sensor_pca_dataset.csv"), index=False)

    # 为进阶验证也保存 latent 和 loadings
    pd.DataFrame(F, columns=["f_env", "f_people", "f_machine"]).to_csv(os.path.join(BASE_DIR,"csv/latent_factors.csv"), index=False)
    pd.DataFrame(loadings, index=col_names, columns=["L_env", "L_people", "L_machine"]).to_csv(os.path.join(BASE_DIR,"csv/loadings_truth.csv"))

    print("生成完成：", df.shape)
    print("异常样本数（ground truth）:", n_anoms)
    print("CSV 已保存为: synthetic_sensor_pca_dataset.csv")

if __name__ == "__main__":
    generate_data()