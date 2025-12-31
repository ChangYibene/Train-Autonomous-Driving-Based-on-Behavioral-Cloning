import numpy as np


class SimpleMPC:
    """
    改进版 MPC：多目标优化 + 状态记忆
    """

    def __init__(self, model, track):
        self.model = model
        self.track = track
        self.horizon = 30  # 预测 3.0 秒
        self.last_u = 0.0  # [关键] 记录上一步控制量

        # 权重参数 (张淼论文)
        self.w_tracking = 1.0
        self.w_comfort = 0.5
        self.w_energy = 0.05

    def reset(self):
        """重置内部状态"""
        self.last_u = 0.0

    def get_action(self, pos, vel):
        min_cost = float('inf')
        best_u = -1.0  # 默认最大制动保底

        # 搜索空间
        candidates = np.linspace(-1.0, 1.0, 11)

        # 获取当前ATP限速
        atp_limit = self.track.get_target_v(pos)  # 简化处理，假设target即limit基础

        for u in candidates:
            cost = 0
            # 基础舒适度代价 (Jerk)
            cost += self.w_comfort * ((u - self.last_u) ** 2)
            # 节能代价
            cost += self.w_energy * (u ** 2)

            # --- 快速预测 ---
            p_curr = pos
            v_curr = vel
            f_curr = self.model.current_force
            valid_path = True

            for t in range(self.horizon):
                # 1. 物理预测
                if u >= 0:
                    tau = 0.5
                    # 恒功率限制逻辑
                    p_lim = self.model.max_power / max(v_curr, 1.0)
                    f_avail = min(self.model.max_traction_force, p_lim)
                    f_tgt = u * f_avail
                else:
                    tau = 1.2
                    f_tgt = u * self.model.max_braking_force

                alpha = 0.1 / (tau + 0.1)
                f_curr = (1 - alpha) * f_curr + alpha * f_tgt

                res = 1.0 + 0.01 * v_curr + 0.0008 * v_curr ** 2
                acc = (f_curr - res) / self.model.mass

                v_curr += acc * 0.1
                p_curr += v_curr * 0.1
                if v_curr < 0: v_curr = 0

                # 2. 安全约束 (预测不能超速)
                limit_future = self.track.get_target_v(p_curr)
                if v_curr > limit_future:
                    valid_path = False
                    break

                # 3. 跟踪代价
                target_future = self.track.get_target_v(p_curr)
                # 高速下主动降速3km/h以留余量
                if target_future > 200 / 3.6: target_future -= 3.0 / 3.6

                err = v_curr - target_future
                cost += self.w_tracking * (err ** 2)

            if valid_path:
                if cost < min_cost:
                    min_cost = cost
                    best_u = u

        # [关键] 更新记忆
        self.last_u = best_u
        return best_u