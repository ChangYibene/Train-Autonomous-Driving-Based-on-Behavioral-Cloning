import numpy as np


class SimpleMPC:
    """考虑到功率限制的 MPC (修正版)"""

    def __init__(self, model, track):
        self.model = model
        self.track = track
        self.horizon = 30  # 预测3秒

    def get_action(self, pos, vel):
        min_cost = float('inf')
        best_u = -1.0  # 默认最大制动保底

        # 候选动作
        candidates = [-1.0, -0.6, 0.0, 0.3, 0.6, 1.0]

        for u in candidates:
            cost = 0
            # 快速预测变量初始化
            p_curr = pos
            v_curr = vel
            f_curr = self.model.current_force
            valid = True

            for t in range(self.horizon):
                # 1. 确定物理参数 (对齐 TrainDynamics)
                if u >= 0:
                    tau_sim = 0.5  # 牵引响应快
                else:
                    tau_sim = 1.2  # 制动响应慢

                # 动态计算 alpha (dt = 0.1)
                alpha = 0.1 / (tau_sim + 0.1)

                # 2. 计算理想目标力 (含恒功率限制)
                if u >= 0:
                    # 补全牵引逻辑：恒力区 vs 恒功区
                    v_safe = max(v_curr, 1.0)
                    p_limit = self.model.max_power / v_safe
                    # 真正的可用力是：电机最大力 与 功率限制力 的较小值
                    f_avail = min(self.model.max_traction_force, p_limit)
                    f_tgt = u * f_avail
                else:
                    f_tgt = u * self.model.max_braking_force

                # 3. 状态更新 (只更新一次！)
                # 模拟一阶惯性延迟
                f_curr = (1 - alpha) * f_curr + alpha * f_tgt

                # 4. 计算阻力与加速度
                res = 1.0 + 0.01 * v_curr + 0.0008 * (v_curr ** 2)
                acc = (f_curr - res) / self.model.mass

                # 5. 欧拉积分
                v_curr += acc * 0.1
                p_curr += v_curr * 0.1

                # 物理限制：速度不能为负
                if v_curr < 0: v_curr = 0

                # --- 约束检查 (ATP) ---
                # 允许稍微超过目标曲线一点点(2km/h)，但绝不能超过太多
                limit_v = self.track.get_target_v(p_curr) + 2.0 / 3.6
                if v_curr > limit_v:
                    valid = False
                    break

                # --- 代价计算 ---
                target_v = self.track.get_target_v(p_curr)
                # 误差平方和
                cost += (v_curr - target_v) ** 2

            if valid:
                if cost < min_cost:
                    min_cost = cost
                    best_u = u

        return best_u