import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------- 仿真参数 (Simulation parameters) ----------
time_step = 1.0               # Δt (dt)
num_steps = 60                # 时间步数 (n_steps)

# ---------- 系统状态维度说明 ----------
# 状态向量为 [position, velocity]^T (state vector: [position, velocity])

# ---------- 系统动力学矩阵 (State transition matrix F) ----------
state_transition = np.array([[1, time_step],
                             [0, 1]])  # F

# ---------- 观测矩阵 (Observation matrix H) ----------
observation_matrix = np.array([[1, 0]])  # H, 我们只测量位置 (we only observe position)

# ---------- 过程噪声与测量噪声协方差 (Process Q, Measurement R) ----------
process_noise_intensity = 0.1  # q，过程噪声强度 (process noise intensity)
# 使用常见的连续到离散近似（针对匀速模型的过程噪声离散化）
process_noise_cov = process_noise_intensity * np.array([
    [time_step**4 / 4, time_step**3 / 2],
    [time_step**3 / 2, time_step**2]
])  # Q

measurement_noise_cov = np.array([[1.0]])  # R，测量噪声协方差 (measurement noise covariance)

# ---------- 真实初始状态 (True initial state) ----------
true_state = np.array([[0.0],   # position
                       [1.0]])  # velocity (1 unit per time step)

# ---------- 滤波器的初始估计与协方差 (Initial estimate and covariance) ----------
estimated_state = np.array([[0.0],
                            [0.0]])      # x_est 初始估计 (initial state estimate)
estimate_covariance = np.eye(2) * 5.0     # P，初始较大不确定性 (initial covariance P)

# ---------- 用于存储历史数据以便绘图 (Storage for plotting) ----------
history_true_states = []
history_measurements = []
history_estimated_states = []
history_covariances = []

# ---------- 仿真与卡尔曼滤波主循环 (Simulate & apply Kalman filter) ----------
for step in range(num_steps):
    # --- 模拟真实系统演化 (Simulate true dynamics) ---
    # 过程噪声 w ~ N(0, Q) (process noise)
    process_noise = np.random.multivariate_normal(mean=[0, 0], cov=process_noise_cov).reshape(2, 1)
    true_state = state_transition @ true_state + process_noise

    # 测量 z = H x + v，测量噪声 v ~ N(0, R) (measurement)
    measurement = observation_matrix @ true_state + np.random.normal(
        loc=0.0, scale=np.sqrt(measurement_noise_cov[0, 0]), size=(1, 1)
    )

    # --- 卡尔曼预测步骤 (Predict) ---
    # 状态预测: x_pred = F x_prev
    predicted_state = state_transition @ estimated_state

    # 协方差预测: P_pred = F P F^T + Q
    predicted_covariance = state_transition @ estimate_covariance @ state_transition.T + process_noise_cov

    # --- 卡尔曼更新步骤 (Update) ---
    # 1) 创新协方差（innovation covariance） S = H P_pred H^T + R
    innovation_covariance = observation_matrix @ predicted_covariance @ observation_matrix.T + measurement_noise_cov

    # 2) 卡尔曼增益（Kalman gain） K = P_pred H^T S^{-1}
    kalman_gain = predicted_covariance @ observation_matrix.T @ np.linalg.inv(innovation_covariance)

    # 3) 创新（residual / innovation） y = z - H x_pred
    innovation = measurement - observation_matrix @ predicted_state

    # 4) 更新状态估计 x = x_pred + K y
    estimated_state = predicted_state + kalman_gain @ innovation

    # 5) 更新估计协方差 P = (I - K H) P_pred
    identity = np.eye(predicted_covariance.shape[0])
    estimate_covariance = (identity - kalman_gain @ observation_matrix) @ predicted_covariance

    # --- 存储历史值 (store for plotting) ---
    history_true_states.append(true_state.flatten())
    history_measurements.append(measurement.flatten())
    history_estimated_states.append(estimated_state.flatten())
    history_covariances.append(estimate_covariance.copy())

# ---------- 转为 numpy arrays 方便处理 (convert to arrays) ----------
history_true_states = np.array(history_true_states)        # shape (num_steps, 2)
history_measurements = np.array(history_measurements).flatten()  # shape (num_steps,)
history_estimated_states = np.array(history_estimated_states)    # shape (num_steps, 2)
history_covariances = np.array(history_covariances)        # shape (num_steps, 2, 2)

# ---------- 绘图：位置对比 (Plot 1) ----------
plt.figure(figsize=(10, 5))
plt.plot(history_true_states[:, 0], label='True Position')
plt.plot(history_measurements, label='Measurements (noisy)')
plt.plot(history_estimated_states[:, 0], label='Kalman Estimated Position')
plt.title('Position: True vs Measurements vs Kalman Estimate')
plt.xlabel('Time step')
plt.ylabel('Position')
plt.legend()
plt.grid(True)
plt.show()

# ---------- 绘图：估计误差与 2-sigma 区间 (Plot 2) ----------
estimation_error = history_estimated_states[:, 0] - history_true_states[:, 0]
position_variance = history_covariances[:, 0, 0]  # 从 P 中读取位置分量的方差
upper_bound = 2 * np.sqrt(position_variance)
lower_bound = -2 * np.sqrt(position_variance)

plt.figure(figsize=(10, 5))
plt.plot(estimation_error, label='Estimation Error (pos)')
plt.plot(upper_bound, linestyle='--', label='+2 sigma (filter)')
plt.plot(lower_bound, linestyle='--', label='-2 sigma (filter)')
plt.title('Estimation Error with 2-sigma Bounds from Kalman Covariance')
plt.xlabel('Time step')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()
