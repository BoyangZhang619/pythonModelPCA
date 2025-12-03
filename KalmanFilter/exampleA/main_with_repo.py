import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

np.random.seed(42)

# ---------- 仿真参数 ----------
dt = 1.0
num_steps = 60

# ---------- 系统动力学 ----------
F = np.array([[1, dt],
              [0, 1]])          # 状态转移矩阵

H = np.array([[1, 0]])          # 只观测位置

# 过程噪声（与原代码相同公式）
q = 0.1
Q = q * np.array([
    [dt**4 / 4, dt**3 / 2],
    [dt**3 / 2, dt**2]
])

R = np.array([[1.0]])           # 测量噪声

# ---------- 真实初始状态 ----------
true_state = np.array([[0.0],
                       [1.0]])

# ---------- KalmanFilter 对象 ----------
kf = KalmanFilter(dim_x=2, dim_z=1)
kf.x = np.array([[0.0],          # 初始估计位置
                 [0.0]])         # 初始估计速度
kf.F = F
kf.H = H
kf.P = np.eye(2) * 5.0
kf.Q = Q
kf.R = R

# ---------- 存储结果 ----------
history_true = []
history_meas = []
history_est = []
history_cov = []

# ---------- 仿真 + 滤波 ----------
for _ in range(num_steps):

    # ----- 真实系统 -----
    process_noise = np.random.multivariate_normal(mean=[0, 0], cov=Q).reshape(2, 1)
    true_state = F @ true_state + process_noise

    # 测量
    measurement = H @ true_state + np.random.normal(0, np.sqrt(R[0, 0]), size=(1, 1))

    # ----- 卡尔曼滤波 -----
    kf.predict()
    kf.update(measurement)

    # 保存
    history_true.append(true_state.flatten())
    history_meas.append(measurement.flatten())
    history_est.append(kf.x.flatten())
    history_cov.append(kf.P.copy())

# 转成数组
history_true = np.array(history_true)
history_meas = np.array(history_meas).flatten()
history_est = np.array(history_est)
history_cov = np.array(history_cov)

# ---------- 绘图 ----------
plt.figure(figsize=(10, 5))
plt.plot(history_true[:, 0], label="True Position")
plt.plot(history_meas, label="Measurements (noisy)")
plt.plot(history_est[:, 0], label="Kalman Estimated Position")
plt.title("Position: True vs Measurements vs Kalman Estimate")
plt.legend()
plt.grid()
plt.show()

# ---------- 误差 + 2sigma ----------
error = history_est[:, 0] - history_true[:, 0]
pos_var = history_cov[:, 0, 0]

plt.figure(figsize=(10, 5))
plt.plot(error, label="Estimation Error (pos)")
plt.plot(2 * np.sqrt(pos_var), "--", label="+2 sigma")
plt.plot(-2 * np.sqrt(pos_var), "--", label="-2 sigma")
plt.title("Estimation Error with 2-sigma Bounds (FilterPy)")
plt.legend()
plt.grid()
plt.show()
