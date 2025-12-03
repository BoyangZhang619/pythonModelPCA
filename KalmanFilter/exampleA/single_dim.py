import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

# 生成的是经过变化的cos加上噪音

# 生成真实信号
def generate_noisy_cos(n=5000):
    x = np.linspace(0, 6*np.pi, n)
    true_signal = 0.7*np.cos(x) + 0.2*np.sin(0.3*x)   # 随意变形一下的 cos
    noise = np.random.normal(0, 0.2, n)
    measured = true_signal + noise
    return x, true_signal, measured

# 简单一维卡尔曼滤波器
def kalman_1d(zs, R=0.2, Q=1e-4):
    # zs 是观测序列
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([[0.]])         # 初始估计
    kf.F = np.array([[1.]])         # 状态转移：x_k = x_(k-1)
    kf.H = np.array([[1.]])         # 观测模型：z_k = x_k
    kf.P = np.array([[1.]])         # 初始协方差
    kf.R = R                         # 观测噪声协方差
    kf.Q = np.array([[Q]])           # 过程噪声协方差

    filtered = []
    for z in zs:
        kf.predict()
        kf.update(z)
        filtered.append(kf.x[0,0])
    return np.array(filtered)


def kalman_1d_smooth(zs, dt=0.1, R=0.15, Q=1e-4):
    kf = KalmanFilter(dim_x=2, dim_z=1)

    kf.x = np.array([[0.0], [0.0]])  # 初始：值 + 变化率
    kf.F = np.array([[1, dt],
                     [0, 1]])  # 经典匀速模型
    kf.H = np.array([[1, 0]])  # 只观测“值”
    kf.P *= 1.0
    kf.R = R
    kf.Q = Q * np.array([[dt ** 4 / 4, dt ** 3 / 2],
                         [dt ** 3 / 2, dt ** 2]])  # 匀速模型的过程噪声结构

    filtered = []
    for z in zs:
        kf.predict()
        kf.update(z)
        filtered.append(kf.x[0, 0])
    print(list(map(float,filtered)))
    return np.array(filtered)

# 生成数据
x, true_signal, measured = generate_noisy_cos()
# 卡尔曼过滤
filtered = kalman_1d(measured)

# 可选：画图
plt.figure(figsize=(10,4))
plt.plot(x, true_signal, label='True')
plt.plot(x, measured, label='Measured', alpha=0.5)
plt.plot(x, filtered, label='Kalman')
plt.legend()
plt.title("zby666")
plt.tight_layout()
plt.show()


# 卡尔曼过滤
filtered = kalman_1d_smooth(measured)

# 可选：画图
plt.figure(figsize=(10,4))
plt.plot(x, true_signal, label='True')
plt.plot(x, measured, label='Measured', alpha=0.5)
plt.plot(x, filtered, label='Kalman')
plt.legend()
plt.title("zby6666666666666666666")
plt.tight_layout()
plt.show()
