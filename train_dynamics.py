import numpy as np


class TrainDynamics:
    """
    CR400AF 动力学模型 (最终修正版)
    包含：恒功率特性、Davis阻力、执行机构延迟、起步死区、防倒车
    """

    def __init__(self):
        # === 车辆参数 (归一化为单节模型) ===
        self.mass = 55.0  # 吨 (t)
        self.dt = 0.1  # 仿真步长 100ms

        # 牵引能力
        self.max_traction_force = 28.0  # kN
        self.max_power = 1300.0  # kW (恒功率限制)

        # 制动能力
        self.max_braking_force = 50.0  # kN

        # === 阻力系数 (高速列车) ===
        self.da = 1.0  # 机械阻力
        self.db = 0.01  # 摩擦阻力
        self.dc = 0.0008  # 空气阻力

        # 执行机构状态
        self.current_force = 0.0
        self.traction_tau = 0.5
        self.braking_tau = 1.2

    def step(self, position, velocity, target_control_input):
        # --- 健壮性检查：防止 NaN 输入 ---
        if np.isnan(target_control_input):
            target_control_input = 0.0

        u = np.clip(target_control_input, -1.0, 1.0)

        # 1. 计算理想目标力 (含恒功率限制)
        if u >= 0:
            v_safe = max(velocity, 1.0)
            power_limit_force = self.max_power / v_safe
            available_max_force = min(self.max_traction_force, power_limit_force)
            f_target = u * available_max_force
            tau = self.traction_tau
        else:
            f_target = u * self.max_braking_force
            tau = self.braking_tau

        # 2. 模拟延迟
        alpha = self.dt / (tau + self.dt)
        self.current_force = (1 - alpha) * self.current_force + alpha * f_target

        # 3. 计算阻力
        resistance = self.da + self.db * velocity + self.dc * (velocity ** 2)

        # 4. 起步摩擦逻辑 (这一段写得非常好！)
        start_resistance = max(resistance, 3.0)  # 模拟静摩擦
        acceleration = 0.0

        # 情况A: 速度极低
        if velocity < 0.1:
            # 如果是牵引(F>0)但推不动(F<=f)，或者处于刹停状态 -> 加速度为0
            if (self.current_force > 0 and self.current_force <= start_resistance) or (
                    velocity == 0 and self.current_force <= 0):
                acceleration = 0.0
            else:
                # 否则(比如大力牵引，或者正在倒车制动)，正常计算
                acceleration = (self.current_force - resistance) / self.mass
        else:
            # 情况B: 正常运行 -> 正常计算
            acceleration = (self.current_force - resistance) / self.mass

        # 5. 状态更新
        # === 修复点：补上了位置更新公式 ===
        next_position = position + velocity * self.dt + 0.5 * acceleration * (self.dt ** 2)

        next_velocity = velocity + acceleration * self.dt

        # 防止倒车 (Anti-rollback)
        if next_velocity < 0:
            next_velocity = 0
            acceleration = 0  # 修正：刹停瞬间加速度归零

        # 6. 返回归一化控制量 (用于记录数据给AI学习)
        if self.current_force >= 0:
            norm_u = self.current_force / self.max_traction_force
        else:
            norm_u = self.current_force / self.max_braking_force

        return next_position, next_velocity, norm_u