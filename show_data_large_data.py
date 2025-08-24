import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import font_manager as fm
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QListWidget, QCheckBox,
    QSplitter, QAction, QToolBar, QStatusBar, QScrollArea,
    QDateTimeEdit, QFormLayout, QGroupBox, QSizePolicy, QFrame,
    QRadioButton, QButtonGroup, QComboBox, QMessageBox
)
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
        self.plot_mode = "multi"   # 默认多图模式

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

        file_menu.addSeparator()

        export_current_action = QAction("导出当前时间段数据", self)
        export_current_action.triggered.connect(self.export_current_data)
        file_menu.addAction(export_current_action)

        export_merged_action = QAction("导出合并数据", self)
        export_merged_action.triggered.connect(self.export_merged_data)
        file_menu.addAction(export_merged_action)

        file_menu.addSeparator()

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

        # ===== 新增：采样频率与方式 =====
        self.main_toolbar.addWidget(QLabel("  采样频率:"))
        self.freq_combo = QComboBox()
        self.freq_combo.addItems(["原始数据", "1S", "5S", "10S", "30S", "1T", "5T", "15T", "30T", "1H"])
        self.freq_combo.setCurrentText("原始数据")
        self.freq_combo.setMinimumWidth(90)
        self.main_toolbar.addWidget(self.freq_combo)

        self.main_toolbar.addWidget(QLabel("  统计方式:"))
        self.agg_combo = QComboBox()
        self.agg_combo.addItems(["mean", "first", "max", "min", "median"])
        self.agg_combo.setCurrentText("mean")
        self.agg_combo.setMinimumWidth(80)
        self.main_toolbar.addWidget(self.agg_combo)

        btn_apply_freq = QPushButton("应用采样")
        btn_apply_freq.clicked.connect(self.plot_data)
        self.main_toolbar.addWidget(btn_apply_freq)
        # ============================

        # 添加分隔符
        self.main_toolbar.addSeparator()

        # 添加刷新按钮
        btn_refresh = QPushButton("刷新图表")
        btn_refresh.clicked.connect(self.plot_data)
        self.main_toolbar.addWidget(btn_refresh)

        # 导出当前时间段数据按钮（便捷）
        btn_export_current = QPushButton("导出当前时间段数据")
        btn_export_current.clicked.connect(self.export_current_data)
        self.main_toolbar.addWidget(btn_export_current)

        btn_export_merged = QPushButton("导出合并数据")
        btn_export_merged.clicked.connect(self.export_merged_data)
        self.main_toolbar.addWidget(btn_export_merged)

    def toggle_all_vars(self, select):
        """全选或全不选变量"""
        for i in range(self.variable_layout.count()):
            widget = self.variable_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox):
                widget.setChecked(select)
        self.plot_data()

    def get_short_name(self, full_name):
        """生成变量简称"""
        if full_name in self.var_short_names:
            return self.var_short_names[full_name]

        if '(' in full_name and ')' in full_name:
            parts = full_name.split('(')
            if parts[0].strip():
                short_name = parts[0].strip()
            else:
                bracket_content = full_name.split('(')[1].split(')')[0].strip()
                short_name = bracket_content
        else:
            short_name = full_name[:15] + ('...' if len(full_name) > 15 else '')

        self.var_short_names[full_name] = short_name
        return short_name

    def _normalize_time_index(self, idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """确保无时区并过滤极端日期，避免matplotlib溢出"""
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(None)
        # 过滤异常时间（避免 2262 年等溢出）
        lower = pd.Timestamp("1970-01-01")
        upper = pd.Timestamp("2100-01-01")
        idx = pd.DatetimeIndex(np.clip(idx.view('i8'),
                                       lower.value, upper.value)).tz_localize(None)
        return idx

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

                try:
                    # 读取数据文件
                    if file.endswith('.csv'):
                        df = pd.read_csv(file)
                    else:
                        df = pd.read_excel(file)

                    if df.empty:
                        self.status_bar.showMessage(f"文件 {file} 为空", 3000)
                        continue

                    # 第一列为时间列
                    time_col = df.columns[0]

                    # 解析时间列（尽可能容错，支持UTC与纳秒）
                    try:
                        # 若原时间为ISO带Z/时区，先以utc解析，再去tz
                        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors='coerce')
                        df[time_col] = df[time_col].dt.tz_convert(None)
                    except Exception:
                        try:
                            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                        except Exception:
                            self.status_bar.showMessage(f"无法解析时间列: {time_col}", 5000)
                            continue

                    # 去除无效时间
                    df = df.dropna(subset=[time_col])

                    # 设置时间索引
                    df = df.set_index(time_col)
                    # 规范索引
                    df.index = self._normalize_time_index(df.index)

                    # 添加文件名前缀到列名，避免冲突
                    file_prefix = file.split('/')[-1].split('\\')[-1].split('.')[0] + "_"

                    new_columns = {}
                    for col in df.columns:
                        if col == time_col:
                            continue
                        if isinstance(col, str) and (col.startswith('Unnamed:') or col.strip() == ''):
                            continue
                        new_columns[col] = file_prefix + str(col)

                    df = df.rename(columns=new_columns)

                    # 合并数据（外连接）
                    if self.merged_data.empty:
                        self.merged_data = df
                    else:
                        self.merged_data = pd.merge(
                            self.merged_data, df,
                            left_index=True, right_index=True,
                            how='outer', suffixes=('', '_dup')
                        )
                        # 清除重复列
                        dup_cols = [c for c in self.merged_data.columns if c.endswith('_dup')]
                        for col in dup_cols:
                            base = col[:-4]
                            if base in self.merged_data.columns:
                                self.merged_data = self.merged_data.drop(columns=[col])

                    self.status_bar.showMessage(f"已加载文件: {file}", 3000)

                except Exception as e:
                    self.status_bar.showMessage(f"加载文件错误: {str(e)}", 5000)

        # 统一规范索引（再次保险）
        if not self.merged_data.empty:
            self.merged_data.index = self._normalize_time_index(self.merged_data.index)

        # 更新变量列表和时间范围
        self.update_variable_list()
        self.update_time_range()

    def update_time_range(self):
        """更新时间选择控件的范围"""
        if not self.merged_data.empty:
            min_time = self.merged_data.index.min()
            max_time = self.merged_data.index.max()

            # 转换为QDateTime需要python datetime
            min_qdt = QDateTime(min_time.to_pydatetime())
            max_qdt = QDateTime(max_time.to_pydatetime())

            self.start_time_edit.setDateTimeRange(min_qdt, max_qdt)
            self.end_time_edit.setDateTimeRange(min_qdt, max_qdt)

            if min_time is not pd.NaT and max_time is not pd.NaT:
                start_time = max_time - pd.Timedelta(days=7)
                if start_time < min_time:
                    start_time = min_time

                self.start_time_edit.setDateTime(QDateTime(start_time.to_pydatetime()))
                self.end_time_edit.setDateTime(QDateTime(max_time.to_pydatetime()))

    def clear_files(self):
        self.file_list.clear()
        self.merged_data = pd.DataFrame()
        self.update_variable_list()
        self.figure.clear()
        self.canvas.draw()

    def update_variable_list(self):
        """更新变量选择列表"""
        while self.variable_layout.count():
            child = self.variable_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if not self.merged_data.empty:
            for col in self.merged_data.columns:
                if isinstance(col, str) and col.lower().startswith(('time', 'date', '时间')):
                    continue
                if isinstance(col, str) and (col.startswith('Unnamed:') or col.strip() == ''):
                    continue

                short_name = self.get_short_name(col)
                cb = QCheckBox(f"({short_name})")
                cb.setFont(QFont("Microsoft YaHei", 9))
                cb.setToolTip(str(col))
                cb.setProperty("full_name", str(col))
                cb.stateChanged.connect(self.plot_data)
                self.variable_layout.addWidget(cb)

    def _get_selected_vars(self):
        selected_vars = []
        for i in range(self.variable_layout.count()):
            widget = self.variable_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox) and widget.isChecked():
                full_name = widget.property("full_name")
                selected_vars.append(full_name)
        return selected_vars

    def _apply_time_filter_and_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """按时间范围过滤并按设置的频率/统计方式进行采样"""
        if df.empty:
            return df

        # 时间范围
        start_time = self.start_time_edit.dateTime().toPyDateTime()
        end_time = self.end_time_edit.dateTime().toPyDateTime()

        # 过滤
        df = df.loc[(df.index >= start_time) & (df.index <= end_time)]
        if df.empty:
            return df

        # 采样
        freq = self.freq_combo.currentText()
        agg = self.agg_combo.currentText()

        if freq != "原始数据":
            try:
                # numeric_only=True 避免非数值列破坏聚合
                df = df.resample(freq).agg(agg, numeric_only=True)
                # 全NaN的时间点丢弃
                df = df.dropna(how='all')
            except Exception as e:
                self.status_bar.showMessage(f"采样失败: {e}", 5000)
                # 采样失败则返回未采样数据
        return df

    def plot_data(self):
        if self.merged_data.empty:
            return

        selected_vars = self._get_selected_vars()
        if not selected_vars:
            self.figure.clear()
            self.canvas.draw()
            return

        self.figure.clear()

        # 过滤 + 采样
        time_filtered = self._apply_time_filter_and_sampling(self.merged_data)
        if time_filtered.empty:
            self.status_bar.showMessage("选择的时间范围内没有数据", 3000)
            return

        # 只保留所选列（列可能在采样时因非数值被丢弃）

        keep_cols = [c for c in selected_vars if c in time_filtered.columns]
        if not keep_cols:
            self.status_bar.showMessage("所选变量在当前采样/时间范围内无有效数据", 3000)
            self.figure.clear()
            self.canvas.draw()
            return

        time_filtered = time_filtered[keep_cols]

        # 确保索引无时区，避免matplotlib溢出
        time_filtered.index = self._normalize_time_index(time_filtered.index)

        # 确定绘图模式
        self.plot_mode = "single" if self.single_mode.isChecked() else "multi"

        if self.plot_mode == "multi":
            self.plot_multi_mode(time_filtered, keep_cols)
        else:
            self.plot_single_mode(time_filtered, keep_cols)

    def plot_multi_mode(self, data, selected_vars):
        """多图模式 - 每个变量单独子图"""
        n = len(selected_vars)
        axes = self.figure.subplots(n, 1, sharex=True)
        if n == 1:
            axes = [axes]

        for i, var in enumerate(selected_vars):
            ax = axes[i]
            short_name = self.get_short_name(var)

            data[var].plot(ax=ax, legend=False)

            ax.set_ylabel(short_name, fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.7)

            y_min = data[var].min()
            y_max = data[var].max()
            margin = (y_max - y_min) * 0.1 if (y_max - y_min) != 0 else 1.0
            ax.set_ylim(y_min - margin, y_max + margin)

            ax.set_title(f"{var}", fontsize=10, pad=5)

            if i == n - 1:
                ax.set_xlabel("时间")
                self.format_xaxis(ax, data.index.min(), data.index.max())

        self.figure.tight_layout()
        self.figure.subplots_adjust(hspace=0.4)
        self.canvas.draw()

        self.setup_crosshair(axes)

    def plot_single_mode(self, data, selected_vars):
        """单图模式 - 所有变量在同一图中"""
        ax = self.figure.add_subplot(111)

        for var in selected_vars:
            short_name = self.get_short_name(var)
            data[var].plot(ax=ax, label=f"{short_name} ({var})")

        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=9)
        ax.set_title("多变量对比图", fontsize=12)
        ax.set_ylabel("数值", fontsize=10)
        ax.set_xlabel("时间", fontsize=10)

        self.format_xaxis(ax, data.index.min(), data.index.max())

        self.figure.tight_layout()
        self.canvas.draw()

        self.setup_crosshair([ax])

    def format_xaxis(self, ax, min_time, max_time):
        """格式化X轴日期显示"""
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
        if hasattr(self, 'crosshair_connections'):
            for conn in self.crosshair_connections:
                self.canvas.mpl_disconnect(conn)

        self.crosshair_connections = []

        for ax in axes:
            conn = self.canvas.mpl_connect(
                'motion_notify_event',
                lambda event: self.on_mouse_move(event, axes)
            )
            self.crosshair_connections.append(conn)

    def on_mouse_move(self, event, axes):
        if not event.inaxes:
            return

        # 清除之前的十字线/注解
        for ax in axes:
            if hasattr(ax, 'vline') and ax.vline in ax.lines:
                ax.vline.remove()
                delattr(ax, 'vline')
            if hasattr(ax, 'hline') and ax.hline in ax.lines:
                ax.hline.remove()
                delattr(ax, 'hline')
            if hasattr(ax, 'annotation') and ax.annotation in ax.texts:
                ax.annotation.remove()
                delattr(ax, 'annotation')

        x = event.xdata
        y = event.ydata

        for ax in axes:
            ax.vline = ax.axvline(x, color='gray', linestyle='--', alpha=0.7)
            if ax == event.inaxes:
                ax.hline = ax.axhline(y, color='gray', linestyle='--', alpha=0.7)

            if ax == axes[0]:
                try:
                    # 用 matplotlib 的日期转换，避免溢出
                    x_date = mdates.num2date(x).replace(tzinfo=None)
                    ax.annotation = ax.annotate(
                        f"{x_date.strftime('%Y-%m-%d %H:%M:%S')}",
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                        fontsize=8
                    )
                except Exception:
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

    # ===== 新增：导出功能 =====
    def _build_current_dataframe(self) -> pd.DataFrame:
        """构建当前时间范围 + 当前采样结果 + 当前选择列 的数据表"""
        if self.merged_data.empty:
            return pd.DataFrame()

        selected_vars = self._get_selected_vars()
        if not selected_vars:
            return pd.DataFrame()

        df = self._apply_time_filter_and_sampling(self.merged_data)
        if df.empty:
            return df

        keep_cols = [c for c in selected_vars if c in df.columns]
        df = df[keep_cols]
        df = df.sort_index()
        return df

    def export_current_data(self):
        """导出当前时间范围（含采样）的所选变量数据"""
        df = self._build_current_dataframe()
        if df.empty:
            QMessageBox.information(self, "提示", "当前时间范围内无可导出的数据或未选择变量。")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出当前时间段数据", "",
            "CSV 文件 (*.csv)"
        )
        if not file_path:
            return

        try:
            df_out = df.copy()
            df_out.index.name = "time"
            df_out.to_csv(file_path, encoding='utf-8')
            self.status_bar.showMessage(f"已导出当前时间段数据到: {file_path}", 5000)
        except Exception as e:
            self.status_bar.showMessage(f"导出失败: {e}", 5000)

    def export_merged_data(self):
        """导出合并数据（可选使用当前采样设置）——与当前变量选择无关，导出全部列"""
        if self.merged_data.empty:
            QMessageBox.information(self, "提示", "暂无可导出的合并数据。")
            return

        # 询问是否应用当前采样设置
        apply_sampling = QMessageBox.question(
            self, "采样确认",
            "导出时是否应用当前采样频率与统计方式？\n（是=应用采样；否=导出原始合并数据）",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        ) == QMessageBox.Yes

        if apply_sampling:
            df = self._apply_time_filter_and_sampling(self.merged_data)
        else:
            # 仍然尊重当前时间范围以避免导出过大文件；若希望全量导出，可在此放开
            start_time = self.start_time_edit.dateTime().toPyDateTime()
            end_time = self.end_time_edit.dateTime().toPyDateTime()
            df = self.merged_data.loc[(self.merged_data.index >= start_time) &
                                      (self.merged_data.index <= end_time)]

        if df.empty:
            QMessageBox.information(self, "提示", "导出结果为空（可能是时间范围或采样设置导致）。")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出合并数据", "",
            "CSV 文件 (*.csv)"
        )
        if not file_path:
            return

        try:
            df_out = df.copy()
            df_out.index.name = "time"
            df_out.to_csv(file_path, encoding='utf-8')
            self.status_bar.showMessage(f"已导出合并数据到: {file_path}", 5000)
        except Exception as e:
            self.status_bar.showMessage(f"导出失败: {e}", 5000)
    # =======================

    def closeEvent(self, event):
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
