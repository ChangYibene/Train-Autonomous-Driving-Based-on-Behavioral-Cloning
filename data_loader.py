import os
import pandas as pd
import numpy as np
from track_profile import TrackProfile


class DataLoader:
    """
    数据读取与预处理模块 (支持 CSV 和 Excel)
    核心升级：视野扩大至 50 步 (约10秒)，适配 LSTM 长序列输入
    """

    @staticmethod
    def load_yizhuang_data(file_path):
        if not os.path.exists(file_path):
            return None, None, None, "文件不存在"

        try:
            # 1. 读取数据
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
                return None, None, None, f"列名匹配失败，需包含实际速度、目标速度、控制量"

            # 3. 物理量转换
            v_act = df[cols['v_act']].values / 100.0  # cm/s -> m/s
            v_tgt = df[cols['v_tgt']].values / 100.0
            u_cmd = df[cols['u_cmd']].values / 100.0
            u_cmd = np.clip(u_cmd, -1.0, 1.0)

            # 位置积分 (dt = 0.2s)
            dt = 0.2
            pos = np.cumsum(v_act * dt)

            # 4. 构建 TrackProfile
            max_d = pos[-1] if len(pos) > 0 else 1000
            if max_d < 100: max_d = 1000

            track = TrackProfile(max_d)
            for p, v in zip(pos, v_tgt):
                idx = int(np.clip(p, 0, max_d))
                if idx < len(track.target_curve):
                    track.target_curve[idx] = v
                    track.static_limit[idx] = v + 3.0 / 3.6  # 缓冲

            # 填补空洞
            for i in range(1, len(track.target_curve)):
                if track.target_curve[i] == 0:
                    track.target_curve[i] = track.target_curve[i - 1]
                    track.static_limit[i] = track.static_limit[i - 1]

            # 5. 特征工程 (关键升级)
            dataset = []
            ext_data = {"pos": [], "vel": [], "u": [], "target": []}

            # [核心修改] 扩大视野至 50 步
            LOOK_AHEAD_STEPS = 50
            NORM_V = 25.0
            NORM_ERR = 10.0

            count = len(df)
            for i in range(count - LOOK_AHEAD_STEPS):
                curr_p = pos[i]
                curr_v = v_act[i]
                curr_u = u_cmd[i]
                curr_t = v_tgt[i]

                # Scalar
                last_u = u_cmd[i - 1] if i > 0 else 0.0
                err = curr_t - curr_v
                scalar = [curr_v / NORM_V, err / NORM_ERR, last_u]

                # Sequence (50步长)
                seq = []
                look_dist = 0
                sim_v = max(curr_v, 1.0)
                idx_search = i

                for _ in range(LOOK_AHEAD_STEPS):
                    look_dist += sim_v * 0.2  # 假设dt=0.2
                    p_future = curr_p + look_dist

                    while idx_search < count - 1 and pos[idx_search] < p_future:
                        idx_search += 1

                    v_next = v_tgt[idx_search]
                    seq.append((v_next - curr_v) / NORM_ERR)

                dataset.append(((scalar, seq), curr_u))

                ext_data["pos"].append(curr_p)
                ext_data["vel"].append(curr_v * 3.6)
                ext_data["target"].append(curr_t * 3.6)
                ext_data["u"].append(curr_u)

            return dataset, ext_data, track, f"读取成功: {len(dataset)} 条 (视野: {LOOK_AHEAD_STEPS})"

        except Exception as e:
            import traceback
            return None, None, None, str(e)