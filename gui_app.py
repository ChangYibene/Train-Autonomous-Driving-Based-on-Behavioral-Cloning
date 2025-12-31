import sys
import os
import traceback
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
                             QPushButton, QLabel, QMessageBox, QTableWidget,
                             QTableWidgetItem, QGroupBox, QApplication, QHeaderView,
                             QComboBox, QFileDialog)
from PyQt5.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from simulation_core import SimulationCore
    from training_service import TrainingWorker
except ImportError as e:
    print(f"【严重错误】缺少核心模块: {e}")
    sys.exit(1)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能列车驾驶仿真平台 (数据驱动增强版)")
        self.resize(1380, 850)

        self.core = SimulationCore()

        self.sim_mode = "IDLE"
        self.plot_data = {"pos": [], "vel": [], "target": []}
        self.train_thread = None

        # 记录上次打开的路径，方便下次选择
        self.last_open_path = "D:\\"

        self.setup_ui()
        self.init_default_track()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loop)

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # --- 左侧: 控制面板 ---
        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        l_layout = QVBoxLayout(left_panel)

        # A. 模式选择
        grp_mode = QGroupBox("0. 模式选择")
        v_mode = QVBoxLayout(grp_mode)
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["模式一：CR400AF + MPC仿真", "模式二：真实数据驱动 (亦庄线/地铁)"])
        self.combo_mode.currentIndexChanged.connect(self.on_mode_changed)
        v_mode.addWidget(self.combo_mode)
        l_layout.addWidget(grp_mode)

        # B. 线路配置
        self.grp_line = QGroupBox("1. 线路配置 (MPC模式)")
        v_line = QVBoxLayout(self.grp_line)
        self.table = QTableWidget(4, 3)
        self.table.setHorizontalHeaderLabels(["起点", "终点", "限速"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        btn_gen = QPushButton("应用线路设置")
        btn_gen.clicked.connect(self.apply_track_settings)
        v_line.addWidget(self.table)
        v_line.addWidget(btn_gen)
        l_layout.addWidget(self.grp_line)

        # C. 操作流程
        grp_ctrl = QGroupBox("2. 操作流程")
        v_ctrl = QVBoxLayout(grp_ctrl)

        # 动态按钮：在模式二下变成"选择文件"
        self.btn_step1 = QPushButton("第一步：启动 MPC (采集)")
        self.btn_step1.clicked.connect(self.on_step1_clicked)
        self.btn_step1.setStyleSheet("background-color: #ADD8E6; padding: 10px; font-weight: bold;")

        self.btn_train = QPushButton("第二步：训练 AI 模型")
        self.btn_train.clicked.connect(self.start_training)

        self.btn_ai = QPushButton("第三步：启动 AI (验证)")
        self.btn_ai.clicked.connect(lambda: self.start_sim("AI"))
        self.btn_ai.setStyleSheet("background-color: #FFB6C1; padding: 10px; font-weight: bold;")

        v_ctrl.addWidget(self.btn_step1)
        v_ctrl.addWidget(self.btn_train)
        v_ctrl.addWidget(self.btn_ai)
        l_layout.addWidget(grp_ctrl)

        # D. 仪表盘
        grp_dash = QGroupBox("3. 状态监视")
        v_dash = QVBoxLayout(grp_dash)
        self.lbl_status = QLabel("系统就绪")
        self.lbl_val = QLabel("位置: 0.0m | 速度: 0.0km/h")
        self.lbl_acc = QLabel("加速度: 0.00 m/s²")
        self.lbl_warn = QLabel("")
        self.lbl_warn.setStyleSheet("color: red; font-weight: bold;")

        v_dash.addWidget(self.lbl_status)
        v_dash.addWidget(self.lbl_val)
        v_dash.addWidget(self.lbl_acc)
        v_dash.addWidget(self.lbl_warn)
        l_layout.addWidget(grp_dash)

        l_layout.addStretch()

        # --- 右侧: 绘图区 ---
        right_panel = QWidget()
        r_layout = QVBoxLayout(right_panel)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        r_layout.addWidget(self.canvas)

        layout.addWidget(left_panel)
        layout.addWidget(right_panel)

    def init_default_track(self):
        defaults = [(0, 5000, 200), (5000, 10000, 350), (10000, 12000, 160), (12000, 15000, 80)]
        for r, (s, e, v) in enumerate(defaults):
            self.table.setItem(r, 0, QTableWidgetItem(str(s)))
            self.table.setItem(r, 1, QTableWidgetItem(str(e)))
            self.table.setItem(r, 2, QTableWidgetItem(str(v)))
        self.apply_track_settings()

    def on_mode_changed(self, index):
        if index == 0:
            # 模式一
            self.btn_step1.setText("第一步：启动 MPC (采集)")
            self.grp_line.setTitle("1. 线路配置 (MPC模式)")
            self.grp_line.setEnabled(True)
        else:
            # 模式二
            self.btn_step1.setText("第一步：选择数据文件 (xlsx/csv)")
            self.grp_line.setTitle("1. 线路配置 (自动读取文件)")
            self.grp_line.setEnabled(False)

    def on_step1_clicked(self):
        if self.combo_mode.currentIndex() == 0:
            self.start_sim("MPC")
        else:
            self.load_real_data()

    def load_real_data(self):
        """弹出文件选择框读取数据"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择训练数据文件",
            self.last_open_path,
            "Data Files (*.xlsx *.xls *.csv);;All Files (*)"
        )

        if not file_path:
            return

        self.last_open_path = os.path.dirname(file_path)
        self.lbl_status.setText(f"正在读取: {os.path.basename(file_path)}...")
        QApplication.processEvents()

        # 调用 Core -> DataLoader
        success, msg = self.core.load_external_data(file_path)

        if success:
            QMessageBox.information(self, "读取成功", msg)
            self.lbl_status.setText(f"已加载: {os.path.basename(file_path)}")

            # 绘图预览
            self.ax.clear()
            ext = self.core.external_data
            if ext["pos"]:
                self.ax.plot(ext["pos"], ext["vel"], 'g-', label='Human Driver (Real)')
                self.ax.plot(ext["pos"], ext["target"], 'r--', alpha=0.6, label='Target Profile')
            self.ax.set_title(f"Data Preview: {os.path.basename(file_path)}")
            self.ax.set_xlabel("Distance (m)")
            self.ax.set_ylabel("Speed (km/h)")
            self.ax.legend()
            self.ax.grid(True)
            self.canvas.draw()
        else:
            QMessageBox.critical(self, "读取失败", msg)
            self.lbl_status.setText("读取失败")

    def apply_track_settings(self):
        if self.combo_mode.currentIndex() == 1: return
        try:
            self.core.track.reset_limits()
            for r in range(self.table.rowCount()):
                s = self.table.item(r, 0)
                e = self.table.item(r, 1)
                v = self.table.item(r, 2)
                if s and e and v:
                    self.core.track.apply_limit(float(s.text()), float(e.text()), float(v.text()))
            self.core.track.calculate_braking_curve()
            self.refresh_static_plot()
        except ValueError:
            QMessageBox.warning(self, "错误", "表格输入数据无效")

    def refresh_static_plot(self):
        self.ax.clear()
        self.ax.plot(self.core.track.x, self.core.track.static_limit * 3.6, 'k--', alpha=0.3, label='ATP Limit')
        self.ax.plot(self.core.track.x, self.core.track.target_curve * 3.6, 'r-', label='Target')
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()

    def start_sim(self, mode):
        if mode == "AI" and not self.core.dataset:
            QMessageBox.warning(self, "警告", "无训练数据！请先执行第一步。")
            return

        self.sim_mode = mode
        self.core.reset()
        self.plot_data = {"pos": [], "vel": [], "target": []}

        if mode == "MPC":
            self.core.dataset = []
            self.lbl_status.setText("状态: MPC 运行中...")
        else:
            self.lbl_status.setText("状态: AI 接管验证中...")

        self.timer.start(100)

    def update_loop(self):
        try:
            data = self.core.step(self.sim_mode)

            if data.get("finished"):
                self.timer.stop()
                self.lbl_status.setText("运行结束")
                if self.sim_mode == "MPC":
                    QMessageBox.information(self, "完成", f"采集数据: {data['dataset_count']} 条")
                return

            self.lbl_val.setText(f"位置: {data['pos']:.1f}m | 速度: {data['vel']:.1f} km/h")
            self.lbl_acc.setText(f"加速度: {data['acc']:.2f} m/s²")

            if data['is_emergency']:
                self.lbl_warn.setText("⚠️ ATP 紧急制动触发！")
            else:
                self.lbl_warn.setText("")

            self.plot_data["pos"].append(data['pos'])
            self.plot_data["vel"].append(data['vel'])
            self.plot_data["target"].append(data['target_v'])

            if len(self.plot_data["pos"]) % 5 == 0:
                self.ax.clear()

                # 模式二下绘制真实数据背景
                if self.combo_mode.currentIndex() == 1:
                    ext = self.core.external_data
                    if ext["pos"]:
                        self.ax.plot(ext["pos"], ext["vel"], 'g-', alpha=0.3, label='Human (Real)')
                        self.ax.set_ylim(0, 120)  # 亦庄线速度低
                else:
                    self.ax.plot(self.core.track.x, self.core.track.target_curve * 3.6, 'r-', alpha=0.4, label='Target')
                    self.ax.set_ylim(0, 420)

                self.ax.plot(self.plot_data["pos"], self.plot_data["vel"], 'b-', linewidth=2, label='AI/MPC')
                self.ax.legend(loc='upper right')
                self.ax.grid(True)
                self.canvas.draw()

        except Exception as e:
            self.timer.stop()
            print(traceback.format_exc())
            QMessageBox.critical(self, "运行错误", f"仿真异常:\n{e}")

    # === 训练 ===
    def start_training(self):
        if not self.core.dataset:
            QMessageBox.warning(self, "无数据", "请先采集或读取数据！")
            return

        self.btn_train.setEnabled(False)
        self.lbl_status.setText("初始化训练...")

        self.trainer = TrainingWorker(self.core.net, self.core.dataset)
        self.trainer.progress_signal.connect(lambda c, t, l: self.lbl_status.setText(f"训练中: {c}/{t} Loss:{l:.4f}"))
        self.trainer.finished_signal.connect(self.on_train_done)
        self.trainer.error_signal.connect(lambda msg: QMessageBox.critical(self, "训练错误", msg))

        self.trainer.start()

    def on_train_done(self, loss):
        self.lbl_status.setText(f"训练完成 Loss: {loss:.5f}")
        QMessageBox.information(self, "成功", f"模型训练完成\n最终 Loss: {loss:.5f}")
        self.btn_train.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())