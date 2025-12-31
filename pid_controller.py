import numpy as np


class PIDController:
    """
    传统 PID 控制器 (对照组)
    """

    def __init__(self, kp=2.0, ki=0.01, kd=0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral = 0.0
        self.prev_error = 0.0
        self.dt = 0.1  # 仿真步长

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def get_action(self, current_v, target_v):
        """
        根据速度误差计算控制量 u
        """
        error = target_v - current_v

        # P项
        p_term = self.kp * error

        # I项 (带积分限幅防止饱和)
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -10.0, 10.0)
        i_term = self.ki * self.integral

        # D项
        d_term = self.kd * (error - self.prev_error) / self.dt

        # 总输出
        u = p_term + i_term + d_term

        # 更新状态
        self.prev_error = error

        # 归一化到 [-1, 1]
        return np.clip(u, -1.0, 1.0)