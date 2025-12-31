import os
import pandas as pd
import numpy as np
from track_profile import TrackProfile


class DataLoader:
    @staticmethod
    def load_yizhuang_data(file_path):
        if not os.path.exists(file_path):
            return None, None, None, "文件不存在"

        try:
            # 1. 读取
            if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
                df = pd.read_excel(file_path)
            else:
                try:
                    df = pd.read_csv(file_path, encoding='gbk')
                except:
                    df = pd.read_csv(file_path, encoding='utf-8')

            # 2. 列名映射
            col_map = {
                'v_act': ['实际速度', 'Speed', 'v'],
                'v_tgt': ['目标速度', 'TargetSpeed', 'v_ref'],
                'u_cmd': ['PID控制器期望输出的加速度', 'Handle', 'u', 'Control']
            }

            cols = {}
            for k, v_list in col_map.items():
                for c in v_list:
                    if c in df.columns:
                        cols[k] = c
                        break

            if len(cols) < 3:
                return None, None, None, "列名匹配失败，请检查CSV表头"

            # 3. 物理量转换
            # 地铁数据通常单位: cm/s
            v_act = df[cols['v_act']].values / 100.0
            v_tgt = df[cols['v_tgt']].values / 100.0

            # 控制量: 亦庄线数据里的 PID输出 其实是 cm/s^2
            # 100 cm/s^2 = 1.0 m/s^2。
            # 我们直接将其作为归一化动作 [-1, 1]，因为 1.0 m/s^2 也是地铁的最大加速度左右
            u_cmd = df[cols['u_cmd']].values / 100.0
            u_cmd = np.clip(u_cmd, -1.0, 1.0)

            # 积分位置
            dt = 0.2
            pos = np.cumsum(v_act * dt)

            # 4. 构建 TrackProfile
            max_d = pos[-1] if len(pos) > 0 else 1000
            track = TrackProfile(max_d)

            for p, v in zip(pos, v_tgt):
                idx = int(np.clip(p, 0, max_d))
                if idx < len(track.target_curve):
                    track.target_curve[idx] = v
                    track.static_limit[idx] = v + 3.0 / 3.6  # 3km/h 缓冲

            # 填补空洞
            for i in range(1, len(track.target_curve)):
                if track.target_curve[i] == 0:
                    track.target_curve[i] = track.target_curve[i - 1]
                    track.static_limit[i] = track.static_limit[i - 1]

            # 5. 特征工程 (关键：针对地铁速度优化归一化系数)
            dataset = []
            ext_data = {"pos": [], "vel": [], "u": [], "target": []}

            # === 归一化参数 ===
            # 地铁最高速约 22m/s (80km/h)。
            # 如果除以 100，数值太小。改为除以 25.0，让数值分布在 0~1 之间。
            NORM_V = 25.0
            NORM_ERR = 10.0

            count = len(df)
            for i in range(count - 50):
                curr_p = pos[i]
                curr_v = v_act[i]
                curr_u = u_cmd[i]
                curr_t = v_tgt[i]

                # Scalar
                last_u = u_cmd[i - 1] if i > 0 else 0.0
                err = curr_t - curr_v

                scalar = [curr_v / NORM_V, err / NORM_ERR, last_u]

                # Sequence
                seq = []
                look_dist = 0
                sim_v = max(curr_v, 1.0)
                idx_search = i

                for _ in range(10):
                    look_dist += sim_v * 1.0
                    p_future = curr_p + look_dist

                    while idx_search < count - 1 and pos[idx_search] < p_future:
                        idx_search += 1

                    v_next = v_tgt[idx_search]
                    seq.append((v_next - curr_v) / NORM_ERR)

                dataset.append(((scalar, seq), curr_u))

                # 绘图数据
                ext_data["pos"].append(curr_p)
                ext_data["vel"].append(curr_v * 3.6)
                ext_data["target"].append(curr_t * 3.6)
                ext_data["u"].append(curr_u)

            return dataset, ext_data, track, "数据加载完成"

        except Exception as e:
            return None, None, None, str(e)