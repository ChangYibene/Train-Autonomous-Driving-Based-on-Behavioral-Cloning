import numpy as np


class TrackProfile:
    def __init__(self, total_distance=15000):
        self.total_dist = total_distance
        self.x = np.linspace(0, total_distance, int(total_distance) + 1)
        self.static_limit = np.full_like(self.x, 350.0 / 3.6)  # 默认 350 km/h
        self.target_curve = self.static_limit.copy()

    def reset_limits(self):
        self.static_limit = np.full_like(self.x, 350.0 / 3.6)

    def apply_limit(self, start, end, limit_kmh):
        s = int(np.clip(start, 0, self.total_dist))
        e = int(np.clip(end, 0, self.total_dist))
        self.static_limit[s:e] = limit_kmh / 3.6

    def calculate_braking_curve(self):
        """生成平滑曲线，考虑高速下列车制动距离变长"""
        self.static_limit[-1] = 0
        self.target_curve = self.static_limit.copy()

        # 使用保守制动加速度 -0.5 m/s^2 计算曲线
        acc = 0.5
        dx = 1.0

        for i in range(len(self.x) - 2, -1, -1):
            v_next = self.target_curve[i + 1]
            v_physics = np.sqrt(v_next ** 2 + 2 * acc * dx)
            self.target_curve[i] = min(self.static_limit[i], v_physics)

    def get_target_v(self, pos):
        idx = int(np.clip(pos, 0, self.total_dist))
        return self.target_curve[idx]