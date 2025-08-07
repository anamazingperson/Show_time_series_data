import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import font_manager as fm
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, 
                            QPushButton, QFileDialog, QLabel, QListWidget, QCheckBox,
                            QSplitter, QAction, QToolBar, QStatusBar, QScrollArea,
                            QDateTimeEdit, QFormLayout, QGroupBox, QSizePolicy, QFrame,
                            QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt, QSettings, QDateTime
from PyQt5.QtGui import QIcon, QFont

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class TimeSeriesVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("时序数据分析工具")
        self.setGeometry(100, 100, 1400, 900)
        
        # 初始化数据
        self.data = pd.DataFrame()
        self.merged_data = pd.DataFrame()
        self.var_short_names = {}  # 存储变量简称
        self.plot_mode = "multi"    # 默认多图模式
        
        # 创建UI
        self.create_ui()
        self.create_menu()
        self.create_toolbar()
        
        # 加载设置
        self.settings = QSettings("MyCompany", "TimeSeriesVisualizer")
        self.restoreGeometry(self.settings.value("geometry", self.saveGeometry()))
        
    def create_ui(self):
        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # 使用分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧控制面板 - 固定宽度
        left_panel = QWidget()
        left_panel.setMinimumWidth(400)
        left_panel.setMaximumWidth(500)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # 文件加载区域
        file_group = QGroupBox("数据文件")
        file_layout = QVBoxLayout(file_group)
        
        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(150)
        file_layout.addWidget(self.file_list)
        
        btn_layout = QHBoxLayout()
        btn_load = QPushButton("添加文件")
        btn_load.clicked.connect(self.load_files)
        btn_layout.addWidget(btn_load)
        
        btn_clear = QPushButton("清除所有")
        btn_clear.clicked.connect(self.clear_files)
        btn_layout.addWidget(btn_clear)
        
        file_layout.addLayout(btn_layout)
        left_layout.addWidget(file_group)
        
        # 绘图模式选择
        mode_group = QGroupBox("绘图模式")
        mode_layout = QVBoxLayout(mode_group)
        
        self.mode_group = QButtonGroup(self)
        
        self.multi_mode = QRadioButton("多图模式 (每个变量单独子图)")
        self.multi_mode.setChecked(True)
        self.mode_group.addButton(self.multi_mode)
        mode_layout.addWidget(self.multi_mode)
        
        self.single_mode = QRadioButton("单图模式 (所有变量在同一图中)")
        self.mode_group.addButton(self.single_mode)
        mode_layout.addWidget(self.single_mode)
        
        self.mode_group.buttonClicked.connect(self.plot_data)
        left_layout.addWidget(mode_group)
        
        # 变量选择区域
        variable_group = QGroupBox("选择变量")
        variable_layout = QVBoxLayout(variable_group)
        
        self.variable_list = QWidget()
        self.variable_layout = QVBoxLayout(self.variable_list)
        self.variable_layout.setAlignment(Qt.AlignTop)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.variable_list)
        scroll.setMinimumHeight(300)
        variable_layout.addWidget(scroll)
        
        # 添加全选/全不选按钮
        btn_layout = QHBoxLayout()
        btn_select_all = QPushButton("全选")
        btn_select_all.clicked.connect(lambda: self.toggle_all_vars(True))
        btn_layout.addWidget(btn_select_all)
        
        btn_deselect_all = QPushButton("全不选")
        btn_deselect_all.clicked.connect(lambda: self.toggle_all_vars(False))
        btn_layout.addWidget(btn_deselect_all)
        
        variable_layout.addLayout(btn_layout)
        left_layout.addWidget(variable_group)
        left_layout.addStretch()
        
        # 添加到分割器
        splitter.addWidget(left_panel)
        
        # 右侧绘图区域
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        
        # 创建图形和画布
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # 添加Matplotlib导航工具栏
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        
        splitter.addWidget(plot_widget)
        splitter.setSizes([400, 1000])
        
        main_layout.addWidget(splitter)
        
        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
    def create_menu(self):
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件")
        
        load_action = QAction("添加文件", self)
        load_action.triggered.connect(self.load_files)
        file_menu.addAction(load_action)
        
        export_action = QAction("导出图片", self)
        export_action.triggered.connect(self.export_image)
        file_menu.addAction(export_action)
        
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
    def create_toolbar(self):
        # 创建主工具栏
        self.main_toolbar = self.addToolBar("主工具栏")
        self.main_toolbar.setMovable(False)
        
        # 添加文件操作按钮
        self.main_toolbar.addAction(QIcon.fromTheme("document-open"), "添加文件", self.load_files)
        self.main_toolbar.addAction(QIcon.fromTheme("document-save-as"), "导出图片", self.export_image)
        self.main_toolbar.addSeparator()
        
        # 添加时间范围选择
        self.main_toolbar.addWidget(QLabel("  开始时间:"))
        self.start_time_edit = QDateTimeEdit()
        self.start_time_edit.setCalendarPopup(True)
        self.start_time_edit.setDateTime(QDateTime.currentDateTime().addDays(-7))
        self.start_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.start_time_edit.setMinimumWidth(150)
        self.main_toolbar.addWidget(self.start_time_edit)
        
        self.main_toolbar.addWidget(QLabel("  结束时间:"))
        self.end_time_edit = QDateTimeEdit()
        self.end_time_edit.setCalendarPopup(True)
        self.end_time_edit.setDateTime(QDateTime.currentDateTime())
        self.end_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.end_time_edit.setMinimumWidth(150)
        self.main_toolbar.addWidget(self.end_time_edit)
        
        btn_apply_time = QPushButton("应用时间范围")
        btn_apply_time.clicked.connect(self.plot_data)
        self.main_toolbar.addWidget(btn_apply_time)
        
        # 添加分隔符
        self.main_toolbar.addSeparator()
        
        # 添加刷新按钮
        btn_refresh = QPushButton("刷新图表")
        btn_refresh.clicked.connect(self.plot_data)
        self.main_toolbar.addWidget(btn_refresh)
        
    def toggle_all_vars(self, select):
        """全选或全不选变量"""
        for i in range(self.variable_layout.count()):
            widget = self.variable_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox):
                widget.setChecked(select)
        self.plot_data()
        
    def get_short_name(self, full_name):
        """生成变量简称"""
        # 如果已经有简称，直接返回
        if full_name in self.var_short_names:
            return self.var_short_names[full_name]
        
        # 尝试提取括号前的内容
        if '(' in full_name and ')' in full_name:
            # 尝试提取括号前的英文部分
            parts = full_name.split('(')
            if parts[0].strip():
                short_name = parts[0].strip()
            else:
                # 如果没有括号前的内容，尝试提取括号内的内容
                bracket_content = full_name.split('(')[1].split(')')[0].strip()
                short_name = bracket_content
        else:
            # 如果没有括号，取前15个字符
        
            short_name = full_name[:15] + ('...' if len(full_name) > 15 else '')
        # 保存简称
        self.var_short_names[full_name] = short_name
        return short_name
        
    def load_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择数据文件", "", 
            "数据文件 (*.csv *.xls *.xlsx);;所有文件 (*)"
        )
        
        if not files:
            return
        
        for file in files:
            if file not in [self.file_list.item(i).text() for i in range(self.file_list.count())]:
                self.file_list.addItem(file)
                
                # 读取数据文件
                try:
                    if file.endswith('.csv'):
                        df = pd.read_csv(file)
                    else:  # Excel文件
                        df = pd.read_excel(file)
                    
                    # 检查是否有数据
                    if df.empty:
                        self.status_bar.showMessage(f"文件 {file} 为空", 3000)
                        continue
                    
                    # 确保第一列是时间列
                    time_col = df.columns[0]
                    # print(time_col)
                    
                    # 尝试解析时间列
                    try:
                        df[time_col] = pd.to_datetime(df[time_col])
                    except:
                        self.status_bar.showMessage(f"无法解析时间列: {time_col}", 5000)
                        continue
                    
                    # 设置时间索引
                    df = df.set_index(time_col)
                    
                    # 添加文件名前缀到列名，避免冲突
                    file_prefix = file.split('/')[-1].split('.')[0] + "_"
                    
                    # 重命名列，只添加前缀到数据列（跳过时间列）
                    new_columns = {}
                    for col in df.columns:
                        # print(col)
                        # 跳过任何可能的时间列
                        if col == time_col:
                            # print(col)
                            continue

                        # 检查列名是否为空（可能是多余的逗号导致的空列）
                        if col.startswith('Unnamed:') or col.strip() == '':
                            # print(f"跳过空列: {col}")
                            continue
                        new_columns[col] = file_prefix + col
                    # 修改列名，添加文件名的形式
                    df = df.rename(columns=new_columns)
                    
                    # 合并数据
                    if self.merged_data.empty:
                        self.merged_data = df
                    else:
                        # 使用外连接合并数据，保留所有时间点
                        self.merged_data = pd.merge(
                            self.merged_data, df, 
                            left_index=True, right_index=True, 
                            how='outer', suffixes=('', '_dup')
                        )
                        
                        # 移除重复列（如果有）
                        for col in self.merged_data.columns:
                            if col.endswith('_dup'):
                                original_col = col[:-4]
                                if original_col in self.merged_data.columns:
                                    self.merged_data = self.merged_data.drop(columns=[col])
                    
                    self.status_bar.showMessage(f"已加载文件: {file}", 3000)
                    
                except Exception as e:
                    self.status_bar.showMessage(f"加载文件错误: {str(e)}", 5000)
        
        # 更新变量列表和时间范围
        self.update_variable_list()
        self.update_time_range()
        
    def update_time_range(self):
        """更新时间选择控件的范围"""
        if not self.merged_data.empty:
            # 获取数据的时间范围
            min_time = self.merged_data.index.min()
            max_time = self.merged_data.index.max()
            
            # 转换为QDateTime
            min_qdt = QDateTime(min_time)
            max_qdt = QDateTime(max_time)
            
            # 设置时间控件范围
            self.start_time_edit.setDateTimeRange(min_qdt, max_qdt)
            self.end_time_edit.setDateTimeRange(min_qdt, max_qdt)
            
            # 设置默认值
            if min_time is not pd.NaT and max_time is not pd.NaT:
                # 默认显示最后7天的数据
                start_time = max_time - pd.Timedelta(days=7)
                if start_time < min_time:
                    start_time = min_time
                
                self.start_time_edit.setDateTime(QDateTime(start_time))
                self.end_time_edit.setDateTime(QDateTime(max_time))
    
    def clear_files(self):
        self.file_list.clear()
        self.merged_data = pd.DataFrame()
        self.update_variable_list()
        self.figure.clear()
        self.canvas.draw()
        
    def update_variable_list(self):
        """更新变量选择列表"""
        # 清除现有变量列表
        while self.variable_layout.count():
            child = self.variable_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # 添加新变量复选框
        if not self.merged_data.empty:
            # 只显示数据列，跳过时间索引
            for col in self.merged_data.columns:
                # 跳过任何可能的时间列
                if col.lower().startswith(('time', 'date', '时间')):
                    continue
                        # 检查列名是否为空（可能是多余的逗号导致的空列）
                if col.startswith('Unnamed:') or col.strip() == '':
                    # print(f"跳过空列: {col}")
                    continue
                # 获取变量简称
                short_name = self.get_short_name(col)
                
                # cb = QCheckBox(f"{short_name} ({col})")
                cb = QCheckBox(f"({short_name})")
                cb.setFont(QFont("Microsoft YaHei", 9))
                cb.setToolTip(col)  # 完整名称作为提示
                cb.setProperty("full_name", col)  # 保存完整名称
                cb.stateChanged.connect(self.plot_data)
                self.variable_layout.addWidget(cb)
    
    def plot_data(self):
        if self.merged_data.empty:
            return
            
        # 获取选中的变量
        selected_vars = []
        for i in range(self.variable_layout.count()):
            widget = self.variable_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox) and widget.isChecked():
                full_name = widget.property("full_name")
                selected_vars.append(full_name)
        
        if not selected_vars:
            self.figure.clear()
            self.canvas.draw()
            return
            
        # 准备绘图
        self.figure.clear()
        
        # 获取时间范围
        start_time = self.start_time_edit.dateTime().toPyDateTime()
        end_time = self.end_time_edit.dateTime().toPyDateTime()
        
        # 根据时间范围筛选数据
        time_filtered = self.merged_data.loc[
            (self.merged_data.index >= start_time) & 
            (self.merged_data.index <= end_time)
        ]
        
        if time_filtered.empty:
            self.status_bar.showMessage("选择的时间范围内没有数据", 3000)
            return
        
        # 确定绘图模式
        self.plot_mode = "single" if self.single_mode.isChecked() else "multi"
        
        if self.plot_mode == "multi":
            # 多图模式 - 每个变量单独子图
            self.plot_multi_mode(time_filtered, selected_vars)
        else:
            # 单图模式 - 所有变量在同一图中
            self.plot_single_mode(time_filtered, selected_vars)
    
    def plot_multi_mode(self, data, selected_vars):
        """多图模式 - 每个变量单独子图"""
        # 根据变量数量创建子图
        n = len(selected_vars)
        axes = self.figure.subplots(n, 1, sharex=True)
        
        # 如果只有一个变量，确保axes是列表
        if n == 1:
            axes = [axes]
        
        # 绘制每个变量的时序图
        for i, var in enumerate(selected_vars):
            ax = axes[i]
            
            # 获取变量简称
            short_name = self.get_short_name(var)
            
            # 绘制数据
            data[var].plot(ax=ax, legend=False)
            
            # 设置标签
            ax.set_ylabel(short_name, fontsize=9)
            
            # 设置网格
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 自动调整Y轴范围，保留10%的边距
            y_min = data[var].min()
            y_max = data[var].max()
            margin = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - margin, y_max + margin)
            
            # 添加完整名称到标题
            ax.set_title(f"{var}", fontsize=10, pad=5)
            
            # 格式化X轴日期显示
            if i == n - 1:  # 只在最后一个子图显示X轴标签
                ax.set_xlabel("时间")
                self.format_xaxis(ax, data.index.min(), data.index.max())
        
        # 调整布局
        self.figure.tight_layout()
        self.figure.subplots_adjust(hspace=0.4)
        self.canvas.draw()
        
        # 添加十字光标功能
        self.setup_crosshair(axes)
    
    def plot_single_mode(self, data, selected_vars):
        """单图模式 - 所有变量在同一图中"""
        ax = self.figure.add_subplot(111)
        
        # 绘制所有选中的变量
        for var in selected_vars:
            # 获取变量简称
            short_name = self.get_short_name(var)
            data[var].plot(ax=ax, label=f"{short_name} ({var})")
        
        # 设置网格
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 添加图例
        ax.legend(fontsize=9)
        
        # 设置标题和标签
        ax.set_title("多变量对比图", fontsize=12)
        ax.set_ylabel("数值", fontsize=10)
        ax.set_xlabel("时间", fontsize=10)
        
        # 格式化X轴日期显示
        self.format_xaxis(ax, data.index.min(), data.index.max())
        
        # 调整布局
        self.figure.tight_layout()
        self.canvas.draw()
        
        # 添加十字光标功能
        self.setup_crosshair([ax])
    
    def format_xaxis(self, ax, min_time, max_time):
        """格式化X轴日期显示"""
        # 根据时间范围设置日期格式
        time_span = max_time - min_time
        days = time_span.days
        
        if days > 180:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        elif days > 30:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        elif days > 7:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        elif days > 1:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        else:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def setup_crosshair(self, axes):
        """设置十字光标功能"""
        # 清除之前的连接
        if hasattr(self, 'crosshair_connections'):
            for conn in self.crosshair_connections:
                self.canvas.mpl_disconnect(conn)
        
        self.crosshair_connections = []
        
        for ax in axes:
            conn = self.canvas.mpl_connect('motion_notify_event', 
                                          lambda event: self.on_mouse_move(event, axes))
            self.crosshair_connections.append(conn)
    
    def on_mouse_move(self, event, axes):
        if not event.inaxes:
            return
            
        # 清除之前的十字线
        for ax in axes:
            # 安全移除垂直线
            if hasattr(ax, 'vline') and ax.vline in ax.lines:
                ax.vline.remove()
                delattr(ax, 'vline')
            
            # 安全移除水平线
            if hasattr(ax, 'hline') and ax.hline in ax.lines:
                ax.hline.remove()
                delattr(ax, 'hline')
            
            # 安全移除注解
            if hasattr(ax, 'annotation') and ax.annotation in ax.texts:
                ax.annotation.remove()
                delattr(ax, 'annotation')
        
        # 获取鼠标位置
        x = event.xdata
        y = event.ydata
        
        # 在每个子图上绘制十字线
        for ax in axes:
            # 绘制垂直线
            ax.vline = ax.axvline(x, color='gray', linestyle='--', alpha=0.7)
            
            # 只在当前子图绘制水平线
            if ax == event.inaxes:
                ax.hline = ax.axhline(y, color='gray', linestyle='--', alpha=0.7)
            
            # 在第一个图上显示时间
            if ax == axes[0]:
                try:
                    x_date = pd.to_datetime(x)
                    ax.annotation = ax.annotate(
                        f"{x_date.strftime('%Y-%m-%d %H:%M:%S')}",
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                        fontsize=8
                    )
                except:
                    pass
        
        self.canvas.draw_idle()
    
    def export_image(self):
        if self.figure.axes:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存图片", "", 
                "PNG图像 (*.png);;JPEG图像 (*.jpg);;PDF文件 (*.pdf);;SVG文件 (*.svg)"
            )
            
            if file_path:
                try:
                    self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                    self.status_bar.showMessage(f"图片已保存到: {file_path}", 5000)
                except Exception as e:
                    self.status_bar.showMessage(f"保存失败: {str(e)}", 5000)
    
    def closeEvent(self, event):
        # 保存窗口设置
        self.settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置全局字体
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)
    
    window = TimeSeriesVisualizer()
    window.show()
    sys.exit(app.exec_())