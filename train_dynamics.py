import numpy as np


class TrainDynamics:
    """
    动力学模型 (支持 高铁/地铁 双模式) - 增强制动版
    """

    def __init__(self, vehicle_type="CR400AF"):
        self.dt = 0.1  # 仿真步长 100ms
        self.vehicle_type = vehicle_type

        if vehicle_type == "METRO":
            # === 亦庄线地铁参数 (修正版) ===
            self.mass = 30.0  # 吨
            self.max_traction_force = 35.0

            # [修改点1] 大幅增强制动力，防止AI反应慢导致刹不住
            # 原 40.0 -> 改为 55.0 (保证 -1.0 输出能产生近 1.8 m/s^2 的减速)
            self.max_braking_force = 55.0

            self.max_power = 800.0

            self.da = 2.0
            self.db = 0.05
            self.dc = 0.0

            # [修改点2] 缩短制动响应时间，模拟电制动的快速响应
            self.traction_tau = 0.2
            self.braking_tau = 0.2  # 原 0.4 -> 0.2

        else:
            # === CR400AF 高铁参数 ===
            self.mass = 55.0
            self.max_traction_force = 28.0
            self.max_braking_force = 50.0
            self.max_power = 1300.0

            self.da = 1.0
            self.db = 0.01
            self.dc = 0.0008

            self.traction_tau = 0.5
            self.braking_tau = 1.2

        self.current_force = 0.0

    def step(self, position, velocity, target_control_input):
        if np.isnan(target_control_input): target_control_input = 0.0

        u = np.clip(target_control_input, -1.0, 1.0)

        # 1. 计算目标力
        if u >= 0:
            v_safe = max(velocity, 0.1)
            power_limit = self.max_power / v_safe
            f_avail = min(self.max_traction_force, power_limit)
            f_target = u * f_avail
            tau = self.traction_tau
        else:
            f_target = u * self.max_braking_force
            tau = self.braking_tau

        # 2. 延迟模拟
        alpha = self.dt / (tau + self.dt)
        self.current_force = (1 - alpha) * self.current_force + alpha * f_target

        # 3. 阻力计算
        resistance = self.da + self.db * velocity + self.dc * (velocity ** 2)

        # 4. 加速度 (F = ma)
        if velocity < 0.1 and abs(self.current_force) < resistance:
            acc = 0.0
        else:
            acc = (self.current_force - resistance) / self.mass

        # 5. 更新状态
        next_vel = velocity + acc * self.dt
        next_pos = position + velocity * self.dt + 0.5 * acc * (self.dt ** 2)

        if next_vel < 0:
            next_vel = 0.0
            acc = 0.0

        norm_u = self.current_force / self.max_traction_force if self.current_force > 0 else self.current_force / self.max_braking_force

        return next_pos, next_vel, norm_u