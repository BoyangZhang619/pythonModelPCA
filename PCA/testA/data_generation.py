import os
import numpy as np
import pandas as pd


def generate_data(ranSeed=42, n_normal=1000, n_sensors=20, n_anomaly=50):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_DIR = os.path.join(BASE_DIR, "csv")
    np.random.seed(ranSeed)
    # 正常样本
    X_normal = np.random.normal(loc=np.linspace(10, 50, n_sensors), scale=2, size=(n_normal, n_sensors))

    # 异常样本
    X_anomaly = np.random.normal(loc=np.linspace(10, 50, n_sensors), scale=2, size=(n_anomaly, n_sensors))
    # 在部分传感器上增加异常偏移
    X_anomaly[:, [2, 5, 7]] += np.random.normal(10, 3, size=(n_anomaly, 3))

    # 合并
    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0] * n_normal + [1] * n_anomaly)

    columns = [f"sensor_{i + 1}" for i in range(n_sensors)]
    df = pd.DataFrame(X, columns=columns)
    df['is_anomaly'] = y

    df.to_csv(os.path.join(CSV_DIR,"synthetic_sensor_pca_dataset.csv"), index=False)
    print(df.head())
    print(df.shape)
    print("数据创建完成")


if __name__ == "__main__":
    generate_data()
