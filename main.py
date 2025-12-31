import sys
import os

# === 【关键修复 1】强制指定 Matplotlib 后端 ===
# 必须在导入 pyplot 或 gui_app 之前设置，防止与 PyQt5 冲突
import matplotlib

matplotlib.use('Qt5Agg')

# === 【关键修复 2】解决 Anaconda Qt 插件路径冲突 (0xC0000409 主要原因) ===
# 这一步会强制程序去寻找 PyQt5 自己的插件，而不是 Anaconda 系统环境里的乱七八糟的插件
try:
    import PyQt5

    qt_plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), 'Qt5', 'plugins')
    if os.path.exists(qt_plugin_path):
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugin_path
        print(f"已修复 Qt 插件路径: {qt_plugin_path}")
except Exception as e:
    print(f"环境配置警告: {e}")

# === 导入 GUI ===
from PyQt5.QtWidgets import QApplication

# 注意：gui_app 必须在上述环境配置完成后再导入
try:
    from gui_app import MainWindow
except ImportError as e:
    print("【启动失败】无法导入 gui_app，请检查是否安装了 pandas 或 openpyxl。")
    print(f"错误详情: {e}")
    # 保持控制台开启以便查看错误
    input("按回车键退出...")
    sys.exit(1)

if __name__ == "__main__":
    # 高分屏适配 (可选)
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    print("程序已启动...")
    sys.exit(app.exec_())