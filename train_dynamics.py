import numpy as np


class TrainDynamics:
    """
    动力学模型 (支持 CR400AF 高铁 / DKZ32 地铁)
    """

    def __init__(self, vehicle_type="CR400AF"):
        self.dt = 0.1  # 仿真步长 100ms
        self.vehicle_type = vehicle_type

        if vehicle_type == "METRO":
            # === 北京地铁亦庄线 (DKZ32型 B型车) 参数 ===
            # 数据来源：北车长客 DKZ32 技术规格
            # 编组：6辆编组 (4动2拖)
            # 设计最高速：80 km/h

            # 1. 质量 (Mass)
            # B型车单节均重(含满载乘客 AW2) 约为 34吨
            self.mass = 34.0

            # 2. 力 (Force)
            # 地铁起步快，加速度约 1.0 m/s^2。F = ma = 34 * 1.0 = 34kN。
            # 给定 40kN 留出余量，模拟强劲的鼠笼式异步电机起步
            self.max_traction_force = 40.0

            # 紧急制动/最大常用制动
            # 为了保证 AI 安全，制动力需要设定得比牵引力大 (约 1.2 m/s^2)
            self.max_braking_force = 55.0

            # 3. 功率 (Power)
            # 单车平均有效功率 (考虑动拖比)
            self.max_power = 600.0

            # 4. 戴维斯阻力系数 (Davis Coefficients)
            # 地铁低速运行，基本阻力(摩擦)占比大，空气阻力占比小
            # F_res = da + db*v + dc*v^2
            self.da = 2.5  # 轮轨摩擦较大
            self.db = 0.08  # 机械传动阻力 (传动比 7.69)
            self.dc = 0.002  # 隧道空气阻力

            # 5. 响应延迟 (三相异步电机 + 电空配合)
            # 电制动响应极快
            self.traction_tau = 0.2
            self.braking_tau = 0.2

        else:
            # === CR400AF 复兴号动车组参数 ===
            self.mass = 55.0  # 缩放后的单节等效质量
            self.max_traction_force = 28.0
            self.max_braking_force = 50.0
            self.max_power = 1300.0

            # 高铁气动外形好，摩擦小，但高速下空气阻力(dc)极大
            self.da = 1.0
            self.db = 0.01
            self.dc = 0.0008

            self.traction_tau = 0.5
            self.braking_tau = 1.2

        self.current_force = 0.0

    def step(self, position, velocity, target_control_input):
        # 0. 输入清洗
        if np.isnan(target_control_input): target_control_input = 0.0
        u = np.clip(target_control_input, -1.0, 1.0)

        # 1. 计算目标力 (Target Force)
        if u >= 0:
            # 牵引工况 (Traction)
            v_safe = max(velocity, 0.1)  # 防止除零

            # 恒功率特性 (P = F*v => F = P/v)
            # 异步电机在低速区恒扭矩(恒力)，高速区恒功率
            power_limit = self.max_power / v_safe
            f_avail = min(self.max_traction_force, power_limit)

            f_target = u * f_avail
            tau = self.traction_tau
        else:
            # 制动工况 (Braking)
            f_target = u * self.max_braking_force
            tau = self.braking_tau

        # 2. 执行机构延迟 (一阶惯性环节)
        # 模拟管路充气或电机建立磁场的时间
        alpha = self.dt / (tau + self.dt)
        self.current_force = (1 - alpha) * self.current_force + alpha * f_target

        # 3. 运行阻力 (Davis Formula)
        resistance = self.da + self.db * velocity + self.dc * (velocity ** 2)

        # 4. 动力学方程 (Newton's Second Law)
        # 简单的起步摩擦处理：如果推力小于阻力且没动，就不动
        if velocity < 0.1 and abs(self.current_force) < resistance:
            acc = 0.0
        else:
            acc = (self.current_force - resistance) / self.mass

        # 5. 欧拉积分更新状态
        next_vel = velocity + acc * self.dt
        next_pos = position + velocity * self.dt + 0.5 * acc * (self.dt ** 2)

        # 6. 物理约束
        if next_vel < 0:
            next_vel = 0.0
            acc = 0.0

        # 7. 反馈归一化控制力 (用于观察)
        norm_u = self.current_force / self.max_traction_force if self.current_force > 0 else self.current_force / self.max_braking_force

        return next_pos, next_vel, norm_u