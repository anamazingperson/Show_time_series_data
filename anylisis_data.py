#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
time_series_visualizer_enhanced.py

增强版时序数据分析工具（含大量变量选择、整齐结果区、PID 估计中间过程展示、基于分位数的模糊规则抽取）
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout,
                             QPushButton, QFileDialog, QLabel, QListWidget, QCheckBox,
                             QSplitter, QAction, QToolBar, QStatusBar, QScrollArea,
                             QDateTimeEdit, QGroupBox, QSizePolicy, QRadioButton,
                             QButtonGroup, QPlainTextEdit, QLineEdit, QListWidgetItem)
from PyQt5.QtCore import Qt, QSettings, QDateTime
from PyQt5.QtGui import QIcon, QFont

# 依赖检查
try:
    from statsmodels.tsa.stattools import grangercausalitytests
except Exception:
    grangercausalitytests = None

from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# 中文字体设置（如系统没有这些字体可调整）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

class TimeSeriesVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("时序数据分析工具（增强版）")
        self.setGeometry(80, 80, 1500, 950)

        # 数据
        self.merged_data = pd.DataFrame()
        self.var_short_names = {}

        # UI
        self.create_ui()
        self.create_menu()
        self.create_toolbar()

        # 设置持久化（窗口位置）
        self.settings = QSettings("MyCompany", "TimeSeriesVisualizer")
        try:
            geom = self.settings.value("geometry")
            if geom:
                self.restoreGeometry(geom)
        except Exception:
            pass

    def create_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        splitter = QSplitter(Qt.Horizontal)

        # 左侧面板
        left_panel = QWidget()
        left_panel.setMinimumWidth(460)
        left_panel.setMaximumWidth(560)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(6, 6, 6, 6)

        # 文件组
        file_group = QGroupBox("数据文件")
        file_layout = QVBoxLayout(file_group)
        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(100)
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

        # 绘图模式
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

        # 变量选择（使用 QListWidget，多选，带搜索）
        variable_group = QGroupBox("选择变量（可多选）")
        variable_layout = QVBoxLayout(variable_group)

        # 搜索框
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("搜索:"))
        self.var_search = QLineEdit()
        self.var_search.setPlaceholderText("输入关键词以过滤变量...")
        self.var_search.textChanged.connect(self.filter_variable_list)
        search_layout.addWidget(self.var_search)
        variable_layout.addLayout(search_layout)

        # 列表
        self.var_list_widget = QListWidget()
        self.var_list_widget.setSelectionMode(QListWidget.MultiSelection)
        self.var_list_widget.itemSelectionChanged.connect(self.plot_data)
        self.var_list_widget.setMinimumHeight(300)
        variable_layout.addWidget(self.var_list_widget)

        # 全选/反选按钮
        btn_layout2 = QHBoxLayout()
        btn_select_all = QPushButton("全选")
        btn_select_all.clicked.connect(self.select_all_vars)
        btn_layout2.addWidget(btn_select_all)
        btn_deselect_all = QPushButton("全不选")
        btn_deselect_all.clicked.connect(self.deselect_all_vars)
        btn_layout2.addWidget(btn_deselect_all)
        variable_layout.addLayout(btn_layout2)

        left_layout.addWidget(variable_group)

        # 分析面板（按钮）
        analysis_group = QGroupBox("分析")
        analysis_layout = QVBoxLayout(analysis_group)
        btn_stats = QPushButton("计算统计指标")
        btn_stats.clicked.connect(self.compute_statistics)
        analysis_layout.addWidget(btn_stats)
        btn_corr = QPushButton("相关性矩阵（热力图）")
        btn_corr.clicked.connect(self.compute_and_plot_correlation)
        analysis_layout.addWidget(btn_corr)
        btn_granger = QPushButton("Granger 因果检验")
        btn_granger.clicked.connect(self.compute_granger)
        analysis_layout.addWidget(btn_granger)
        btn_pid = QPushButton("自动估计 PID 参数（显示中间过程）")
        btn_pid.clicked.connect(self.estimate_pid_for_selected)
        analysis_layout.addWidget(btn_pid)
        btn_fuzzy = QPushButton("生成模糊规则（分位数法）")
        btn_fuzzy.clicked.connect(self.generate_fuzzy_rules)
        analysis_layout.addWidget(btn_fuzzy)
        left_layout.addWidget(analysis_group)

        # 结果区（等宽，无换行）
        result_group = QGroupBox("分析结果")
        result_layout = QVBoxLayout(result_group)
        self.result_text = QPlainTextEdit()
        self.result_text.setReadOnly(True)
        mono_font = QFont("Courier New", 9)
        self.result_text.setFont(mono_font)
        self.result_text.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.result_text.setMinimumHeight(260)
        result_layout.addWidget(self.result_text)
        left_layout.addWidget(result_group)

        left_layout.addStretch()
        splitter.addWidget(left_panel)

        # 右侧绘图区域
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        splitter.addWidget(plot_widget)

        splitter.setSizes([480, 1000])
        main_layout.addWidget(splitter)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # 时间范围控件放在工具栏里（create_toolbar）
        # 其他初始化变量
        self.crosshair_connections = []

    def create_menu(self):
        menubar = self.menuBar()
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
        self.main_toolbar = self.addToolBar("主工具栏")
        self.main_toolbar.setMovable(False)
        self.main_toolbar.addAction(QIcon.fromTheme("document-open"), "添加文件", self.load_files)
        self.main_toolbar.addAction(QIcon.fromTheme("document-save-as"), "导出图片", self.export_image)
        self.main_toolbar.addSeparator()

        self.main_toolbar.addWidget(QLabel("  开始时间:"))
        self.start_time_edit = QDateTimeEdit()
        self.start_time_edit.setCalendarPopup(True)
        self.start_time_edit.setDateTime(QDateTime.currentDateTime().addDays(-7))
        self.start_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.start_time_edit.setMinimumWidth(160)
        self.main_toolbar.addWidget(self.start_time_edit)

        self.main_toolbar.addWidget(QLabel("  结束时间:"))
        self.end_time_edit = QDateTimeEdit()
        self.end_time_edit.setCalendarPopup(True)
        self.end_time_edit.setDateTime(QDateTime.currentDateTime())
        self.end_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.end_time_edit.setMinimumWidth(160)
        self.main_toolbar.addWidget(self.end_time_edit)

        btn_apply_time = QPushButton("应用时间范围")
        btn_apply_time.clicked.connect(self.plot_data)
        self.main_toolbar.addWidget(btn_apply_time)

        self.main_toolbar.addSeparator()
        btn_refresh = QPushButton("刷新图表")
        btn_refresh.clicked.connect(self.plot_data)
        self.main_toolbar.addWidget(btn_refresh)

    # -------------------- 文件和变量管理 --------------------
    def load_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "选择数据文件", "",
                                                "数据文件 (*.csv *.xls *.xlsx);;所有文件 (*)")
        if not files:
            return
        for file in files:
            if file not in [self.file_list.item(i).text() for i in range(self.file_list.count())]:
                self.file_list.addItem(file)
                try:
                    if file.lower().endswith('.csv'):
                        df = pd.read_csv(file)
                    else:
                        df = pd.read_excel(file)
                    if df.empty:
                        self.status_bar.showMessage(f"文件 {file} 为空", 3000)
                        continue
                    time_col = df.columns[0]
                    try:
                        df[time_col] = pd.to_datetime(df[time_col])
                    except Exception:
                        self.status_bar.showMessage(f"无法解析时间列: {time_col}", 5000)
                        continue
                    df = df.set_index(time_col)
                    file_prefix = file.split('/')[-1].split('.')[0] + "_"
                    new_columns = {}
                    for col in df.columns:
                        if col == time_col:
                            continue
                        if str(col).startswith('Unnamed:') or str(col).strip() == '':
                            continue
                        new_columns[col] = file_prefix + str(col)
                    df = df.rename(columns=new_columns)
                    if self.merged_data.empty:
                        self.merged_data = df
                    else:
                        self.merged_data = pd.merge(self.merged_data, df,
                                                   left_index=True, right_index=True, how='outer', suffixes=('', '_dup'))
                        # 删除重复后缀
                        for col in list(self.merged_data.columns):
                            if col.endswith('_dup'):
                                original_col = col[:-4]
                                if original_col in self.merged_data.columns:
                                    self.merged_data = self.merged_data.drop(columns=[col])
                    self.status_bar.showMessage(f"已加载文件: {file}", 3000)
                except Exception as e:
                    self.status_bar.showMessage(f"加载文件错误: {str(e)}", 5000)
        # 更新变量列表与时间范围
        self.update_variable_list()
        self.update_time_range()

    def update_time_range(self):
        if not self.merged_data.empty:
            min_time = self.merged_data.index.min()
            max_time = self.merged_data.index.max()
            try:
                min_qdt = QDateTime(min_time)
                max_qdt = QDateTime(max_time)
                self.start_time_edit.setDateTimeRange(min_qdt, max_qdt)
                self.end_time_edit.setDateTimeRange(min_qdt, max_qdt)
                # 默认显示最后7天
                start_time = max_time - pd.Timedelta(days=7)
                if start_time < min_time:
                    start_time = min_time
                self.start_time_edit.setDateTime(QDateTime(start_time))
                self.end_time_edit.setDateTime(QDateTime(max_time))
            except Exception:
                pass

    def clear_files(self):
        self.file_list.clear()
        self.merged_data = pd.DataFrame()
        self.var_list_widget.clear()
        self.figure.clear()
        self.canvas.draw()
        self.result_text.clear()

    def get_short_name(self, full_name):
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

    def update_variable_list(self):
        """把所有变量放到 QListWidget 中，支持搜索与多选"""
        self.var_list_widget.clear()
        if self.merged_data.empty:
            return
        for col in self.merged_data.columns:
            if str(col).lower().startswith(('time', 'date', '时间')):
                continue
            if str(col).startswith('Unnamed:') or str(col).strip() == '':
                continue
            item = QListWidgetItem(self.get_short_name(col))
            item.setToolTip(col)
            item.setData(Qt.UserRole, col)  # 保存完整名称
            self.var_list_widget.addItem(item)
        # Apply filter if search has text
        self.filter_variable_list(self.var_search.text())

    def filter_variable_list(self, text):
        """根据搜索框过滤 QListWidget 的内容（简单子串匹配完整名或简称）"""
        text = text.strip().lower()
        for i in range(self.var_list_widget.count()):
            item = self.var_list_widget.item(i)
            fullname = item.data(Qt.UserRole)
            short = item.text()
            display = True
            if text:
                if text not in str(fullname).lower() and text not in short.lower():
                    display = False
            item.setHidden(not display)

    def select_all_vars(self):
        for i in range(self.var_list_widget.count()):
            item = self.var_list_widget.item(i)
            if not item.isHidden():
                item.setSelected(True)
        self.plot_data()

    def deselect_all_vars(self):
        for i in range(self.var_list_widget.count()):
            item = self.var_list_widget.item(i)
            item.setSelected(False)
        self.plot_data()

    def get_selected_vars(self):
        items = self.var_list_widget.selectedItems()
        return [it.data(Qt.UserRole) for it in items]

    # -------------------- 绘图 --------------------
    def plot_data(self):
        if self.merged_data.empty:
            return
        selected_vars = self.get_selected_vars()
        if not selected_vars:
            self.figure.clear()
            self.canvas.draw()
            return
        self.figure.clear()
        start_time = self.start_time_edit.dateTime().toPyDateTime()
        end_time = self.end_time_edit.dateTime().toPyDateTime()
        time_filtered = self.merged_data.loc[(self.merged_data.index >= start_time) &
                                             (self.merged_data.index <= end_time), selected_vars]
        if time_filtered.empty:
            self.status_bar.showMessage("选择的时间范围内没有数据", 3000)
            return
        self.plot_mode = "single" if self.single_mode.isChecked() else "multi"
        if self.plot_mode == "multi":
            self.plot_multi_mode(time_filtered, selected_vars)
        else:
            self.plot_single_mode(time_filtered, selected_vars)

    def plot_multi_mode(self, data, selected_vars):
        n = len(selected_vars)
        axes = self.figure.subplots(n, 1, sharex=True)
        if n == 1:
            axes = [axes]
        for i, var in enumerate(selected_vars):
            ax = axes[i]
            short_name = self.get_short_name(var)
            try:
                data[var].plot(ax=ax, legend=False)
            except Exception:
                ax.plot(data.index, data[var].values)
            ax.set_ylabel(short_name, fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.7)
            # 自动调整Y轴范围
            y_min = data[var].min()
            y_max = data[var].max()
            if pd.notnull(y_min) and pd.notnull(y_max) and y_max > y_min:
                margin = (y_max - y_min) * 0.1
                ax.set_ylim(y_min - margin, y_max + margin)
            ax.set_title(f"{var}", fontsize=10, pad=5)
            if i == n - 1:
                ax.set_xlabel("时间")
                self.format_xaxis(ax, data.index.min(), data.index.max())
        self.figure.tight_layout()
        self.figure.subplots_adjust(hspace=0.35)
        self.canvas.draw()
        self.setup_crosshair(axes)

    def plot_single_mode(self, data, selected_vars):
        ax = self.figure.add_subplot(111)
        for var in selected_vars:
            short_name = self.get_short_name(var)
            try:
                data[var].plot(ax=ax, label=f"{short_name} ({var})")
            except Exception:
                ax.plot(data.index, data[var].values, label=f"{short_name} ({var})")
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
        try:
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
        except Exception:
            pass

    def setup_crosshair(self, axes):
        if hasattr(self, 'crosshair_connections') and self.crosshair_connections:
            for conn in self.crosshair_connections:
                try:
                    self.canvas.mpl_disconnect(conn)
                except Exception:
                    pass
        self.crosshair_connections = []
        for ax in axes:
            conn = self.canvas.mpl_connect('motion_notify_event',
                                          lambda event: self.on_mouse_move(event, axes))
            self.crosshair_connections.append(conn)

    def on_mouse_move(self, event, axes):
        if not event.inaxes:
            return
        for ax in axes:
            if hasattr(ax, 'vline') and ax.vline in ax.lines:
                try:
                    ax.vline.remove()
                except Exception:
                    pass
                try:
                    delattr(ax, 'vline')
                except Exception:
                    pass
            if hasattr(ax, 'hline') and ax.hline in ax.lines:
                try:
                    ax.hline.remove()
                except Exception:
                    pass
                try:
                    delattr(ax, 'hline')
                except Exception:
                    pass
            if hasattr(ax, 'annotation') and ax.annotation in ax.texts:
                try:
                    ax.annotation.remove()
                except Exception:
                    pass
                try:
                    delattr(ax, 'annotation')
                except Exception:
                    pass
        x = event.xdata
        y = event.ydata
        for ax in axes:
            ax.vline = ax.axvline(x, color='gray', linestyle='--', alpha=0.7)
            if ax == event.inaxes:
                ax.hline = ax.axhline(y, color='gray', linestyle='--', alpha=0.7)
            if ax == axes[0]:
                try:
                    x_date = pd.to_datetime(x)
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
            file_path, _ = QFileDialog.getSaveFileName(self, "保存图片", "",
                                                       "PNG图像 (*.png);;JPEG图像 (*.jpg);;PDF文件 (*.pdf);;SVG文件 (*.svg)")
            if file_path:
                try:
                    self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                    self.status_bar.showMessage(f"图片已保存到: {file_path}", 5000)
                except Exception as e:
                    self.status_bar.showMessage(f"保存失败: {str(e)}", 5000)

    def closeEvent(self, event):
        try:
            self.settings.setValue("geometry", self.saveGeometry())
        except Exception:
            pass
        super().closeEvent(event)

    # -------------------- 统计/相关/Granger --------------------
    def compute_statistics(self):
        if self.merged_data.empty:
            self.status_bar.showMessage("请先加载数据", 3000)
            return
        selected = self.get_selected_vars()
        if not selected:
            self.status_bar.showMessage("请选择变量后再计算统计指标", 3000)
            return
        start_time = self.start_time_edit.dateTime().toPyDateTime()
        end_time = self.end_time_edit.dateTime().toPyDateTime()
        df = self.merged_data.loc[(self.merged_data.index >= start_time) & (self.merged_data.index <= end_time), selected]
        stats = pd.DataFrame(index=selected,
                             columns=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'missing_rate', 'skew', 'kurtosis'])
        for col in selected:
            s = df[col]
            stats.loc[col, 'count'] = int(s.count())
            stats.loc[col, 'mean'] = float(s.mean()) if s.count() else np.nan
            stats.loc[col, 'std'] = float(s.std()) if s.count() else np.nan
            stats.loc[col, 'min'] = float(s.min()) if s.count() else np.nan
            stats.loc[col, '25%'] = float(s.quantile(0.25)) if s.count() else np.nan
            stats.loc[col, '50%'] = float(s.median()) if s.count() else np.nan
            stats.loc[col, '75%'] = float(s.quantile(0.75)) if s.count() else np.nan
            stats.loc[col, 'max'] = float(s.max()) if s.count() else np.nan
            stats.loc[col, 'missing_rate'] = float(s.isna().mean())
            stats.loc[col, 'skew'] = float(s.skew()) if s.count() else np.nan
            stats.loc[col, 'kurtosis'] = float(s.kurtosis()) if s.count() else np.nan
        # 输出到结果区（等宽）
        self.result_text.clear()
        self.result_text.appendPlainText("统计指标（选中变量）:\n")
        self.result_text.appendPlainText(stats.to_string(float_format=lambda x: f"{x:.4g}" if pd.notnull(x) else "nan"))
        self.status_bar.showMessage("已计算统计指标", 3000)

    def compute_and_plot_correlation(self):
        if self.merged_data.empty:
            self.status_bar.showMessage("请先加载数据", 3000)
            return
        selected = self.get_selected_vars()
        if len(selected) < 2:
            self.status_bar.showMessage("至少选择两个变量用于相关性分析", 3000)
            return
        start_time = self.start_time_edit.dateTime().toPyDateTime()
        end_time = self.end_time_edit.dateTime().toPyDateTime()
        df = self.merged_data.loc[(self.merged_data.index >= start_time) & (self.merged_data.index <= end_time), selected]
        df_clean = df.interpolate().dropna()
        if df_clean.empty:
            self.status_bar.showMessage("数据不足，无法计算相关性", 3000)
            return
        corr = df_clean.corr(method='pearson')
        # 输出矩阵文本
        self.result_text.clear()
        self.result_text.appendPlainText("皮尔逊相关系数矩阵:\n")
        self.result_text.appendPlainText(corr.to_string(float_format=lambda x: f"{x:.4f}"))
        # 绘制热力图
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        cax = ax.imshow(corr.values, aspect='auto', interpolation='nearest')
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels([self.get_short_name(c) for c in corr.columns], rotation=45, ha='right')
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels([self.get_short_name(c) for c in corr.index])
        self.figure.colorbar(cax, ax=ax)
        ax.set_title('相关性热力图')
        self.figure.tight_layout()
        self.canvas.draw()
        self.status_bar.showMessage("已绘制相关性热力图", 3000)

    def compute_granger(self):
        if grangercausalitytests is None:
            self.status_bar.showMessage("缺少 statsmodels 包，无法执行 Granger 检验。请安装：pip install statsmodels", 7000)
            return
        if self.merged_data.empty:
            self.status_bar.showMessage("请先加载数据", 3000)
            return
        selected = self.get_selected_vars()
        if len(selected) < 2:
            self.status_bar.showMessage("至少选择两个变量用于 Granger 检验", 3000)
            return
        start_time = self.start_time_edit.dateTime().toPyDateTime()
        end_time = self.end_time_edit.dateTime().toPyDateTime()
        df = self.merged_data.loc[(self.merged_data.index >= start_time) & (self.merged_data.index <= end_time), selected]
        df_clean = df.interpolate().dropna()
        if df_clean.empty:
            self.status_bar.showMessage("数据不足，无法执行 Granger 检验", 3000)
            return
        maxlag = min(10, max(1, int(len(df_clean) / 5)))
        out_lines = []
        for i in range(len(selected)):
            for j in range(len(selected)):
                if i == j:
                    continue
                x = df_clean[selected[i]]
                y = df_clean[selected[j]]
                try:
                    test_result = grangercausalitytests(pd.concat([y, x], axis=1), maxlag=maxlag, verbose=False)
                    pvals = [(lag, test_result[lag][0]['ssr_ftest'][1]) for lag in test_result]
                    best = sorted(pvals, key=lambda z: z[1])[0]
                    out_lines.append(f"{selected[i]} -> {selected[j]} : best_lag={best[0]}, pvalue={best[1]:.4g}")
                except Exception as e:
                    out_lines.append(f"{selected[i]} -> {selected[j]} : error {e}")
        self.result_text.clear()
        self.result_text.appendPlainText("Granger 因果检验（y <- x 表示 x 可用于预测 y）:\n")
        for line in out_lines:
            self.result_text.appendPlainText(line)
        self.status_bar.showMessage("Granger 检验完成（注意：这是可预测性检验，不是严格因果）", 5000)

    # -------------------- PID 估计（显示中间过程） --------------------
    def estimate_pid_for_selected(self):
        """
        对选中的变量逐个检测明显阶跃并拟合一阶滞后响应，输出中间过程：
        - 检测到的阶跃区间（索引/时间）
        - 稳态前后值、过程增益 K、时间常数 tau
        - 拟合优度 R^2
        - 基于简单经验的 PID 建议（Ziegler-Nichols 风格启发）
        并把拟合曲线和原始曲线画到画布上，标注阶跃点。
        """
        if self.merged_data.empty:
            self.status_bar.showMessage("请先加载数据", 3000)
            return
        selected = self.get_selected_vars()
        if not selected:
            self.status_bar.showMessage("请选择变量后执行 PID 估计（建议选择控制量与被控量）", 3000)
            return
        start_time = self.start_time_edit.dateTime().toPyDateTime()
        end_time = self.end_time_edit.dateTime().toPyDateTime()
        df = self.merged_data.loc[(self.merged_data.index >= start_time) & (self.merged_data.index <= end_time), selected]
        df_clean = df.interpolate().dropna()
        if df_clean.empty:
            self.status_bar.showMessage("时间范围或数据不足，无法估计 PID", 3000)
            return

        self.result_text.clear()
        self.result_text.appendPlainText("PID 估计结果（显示中间过程）:\n")

        # 我们会一次只在图上绘制一个变量的拟合结果（最后一个变量绘制覆盖）
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        for col in df_clean.columns:
            self.result_text.appendPlainText(f"------ 变量: {col} ------")
            vals = df_clean[col].values
            if len(vals) < 30:
                self.result_text.appendPlainText("样本过少，跳过\n")
                continue
            steps = self._detect_steps(df_clean[col])
            if not steps:
                self.result_text.appendPlainText("未检测到明显阶跃\n")
                continue
            # 使用第一个检测到的阶跃段
            start_idx, end_idx = steps[0]
            t_base = (df_clean.index - df_clean.index[0]).total_seconds().astype(float)
            t_segment = t_base[start_idx:end_idx+1] - t_base[start_idx]
            y_segment = vals[start_idx:end_idx+1]
            # 如果 segment 长度很短，放弃
            if len(y_segment) < 6:
                self.result_text.appendPlainText("检测到的阶跃区间太短，无法拟合\n")
                continue

            # 初始输出中间步骤
            t0_time = df_clean.index[start_idx]
            t1_time = df_clean.index[end_idx]
            self.result_text.appendPlainText(f"阶跃区间索引: {start_idx} - {end_idx}")
            self.result_text.appendPlainText(f"阶跃时间: {t0_time} -> {t1_time}")

            # 估计稳态前后值：前段平均，后段平均
            pre_mean = np.mean(vals[max(0, start_idx-10):start_idx+1])
            post_mean = np.mean(vals[end_idx:min(len(vals)-1, end_idx+10)+1])
            self.result_text.appendPlainText(f"稳态前值 (y0) ≈ {pre_mean:.4g}, 稳态后值 (y_inf) ≈ {post_mean:.4g}")

            # 拟合一阶滞后模型 y(t) = K*(1 - exp(-t/tau)) + y0
            def first_order(t, K, tau, y0):
                return K * (1 - np.exp(-t / (tau + 1e-9))) + y0

            # 初值猜测
            K_guess = post_mean - pre_mean
            tau_guess = max(1.0, (t_segment[-1] - t_segment[0]) / 3.0)
            y0_guess = pre_mean
            try:
                popt, pcov = curve_fit(first_order, t_segment, y_segment, p0=[K_guess, tau_guess, y0_guess], maxfev=10000)
                K_est, tau_est, y0_est = popt
                # 计算拟合优度 R^2
                y_fit = first_order(t_segment, *popt)
                ss_res = np.sum((y_segment - y_fit) ** 2)
                ss_tot = np.sum((y_segment - np.mean(y_segment)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                self.result_text.appendPlainText(f"拟合结果: K={K_est:.6g}, tau={tau_est:.3f}s, y0={y0_est:.6g}, R^2={r2:.4f}")
                # 延迟估计 L（一个简单启发：设 L = 0.1 * tau）
                L_est = max(0.0, 0.1 * tau_est)
                # Ziegler-Nichols-like 建议（仅作启发）
                # ZN 步响应法（P I D for P process）简化:
                # Kp = 1.2 * tau / (K * L)
                if abs(K_est) < 1e-9:
                    K_est = 1e-9
                Kp_zn = (1.2 * tau_est) / (abs(K_est) * (L_est + 1e-9))
                Ti_zn = 2 * L_est
                Td_zn = 0.5 * L_est
                self.result_text.appendPlainText(f"延迟近似 L ≈ {L_est:.3f}s -> 建议 (启发式): Kp={Kp_zn:.4g}, Ti={Ti_zn:.3f}, Td={Td_zn:.3f}\n")
                # 在图中绘制
                ax.plot(df_clean.index, df_clean[col], label=f"{col} 原始")
                t_fit_index = df_clean.index[start_idx:end_idx+1]
                ax.plot(t_fit_index, y_fit, '--', label=f"{col} 拟合")
                # 标注阶跃起点
                ax.axvline(df_clean.index[start_idx], color='r', linestyle='--', alpha=0.6)
            except Exception as e:
                self.result_text.appendPlainText(f"拟合失败: {e}\n")
                # 也画原始曲线
                ax.plot(df_clean.index, df_clean[col], label=f"{col} 原始 (拟合失败)")

        ax.set_title("阶跃拟合与原始曲线（若有拟合结果）")
        ax.legend(fontsize=9)
        self.figure.tight_layout()
        self.canvas.draw()
        self.status_bar.showMessage("PID 参数估计完成（输出中间过程）", 5000)

    def _detect_steps(self, series, prominence=None, distance=5):
        """
        检测阶跃位置：基于一阶差分找显著突变点
        返回 list of (start_idx, end_idx)
        """
        vals = series.values
        if len(vals) < 10:
            return []
        diff = np.diff(vals)
        # 平滑差分
        diff_s = pd.Series(diff).rolling(window=3, min_periods=1, center=True).median().values
        # 自适应阈值：基于中位数绝对偏差
        mad = np.median(np.abs(diff_s - np.median(diff_s)))
        if prominence is None:
            prominence = max(1e-6, 3 * mad)
        peaks, props = find_peaks(np.abs(diff_s), prominence=prominence, distance=distance)
        steps = []
        for p in peaks:
            start = max(0, p - 8)
            end = min(len(vals) - 1, p + 8)
            steps.append((start, end))
        return steps

    # -------------------- 模糊规则生成（分位数法） --------------------
    def generate_fuzzy_rules(self, top_k=20):
        """
        简单模糊规则生成（基于分位数）
        - 要求至少选择 2 个变量：前 n-1 为 antecedents，最后一个为 consequent（输出）
        - 将每个变量按 33%/66% 分位数量化为 '低'/'中'/'高'
        - 统计 antecedent -> consequent 的高频组合，输出 Top K
        """
        if self.merged_data.empty:
            self.status_bar.showMessage("请先加载数据", 3000)
            return
        selected = self.get_selected_vars()
        if len(selected) < 2:
            self.status_bar.showMessage("请至少选择 2 个变量（前者为输入，最后一个为输出）", 4000)
            return
        start_time = self.start_time_edit.dateTime().toPyDateTime()
        end_time = self.end_time_edit.dateTime().toPyDateTime()
        df = self.merged_data.loc[(self.merged_data.index >= start_time) & (self.merged_data.index <= end_time), selected]
        df_clean = df.interpolate().dropna()
        if df_clean.empty:
            self.status_bar.showMessage("数据不足，无法生成规则", 3000)
            return

        inputs = selected[:-1]
        output = selected[-1]

        # 计算分位数阈值
        q_lo = df_clean.quantile(0.33)
        q_hi = df_clean.quantile(0.66)

        # 量化函数
        def fuzzify_val(col, val):
            lo = q_lo[col]
            hi = q_hi[col]
            if val <= lo:
                return '低'
            elif val >= hi:
                return '高'
            else:
                return '中'

        rules_count = {}
        # 遍历样本，构造规则
        for idx in range(len(df_clean)):
            row = df_clean.iloc[idx]
            antecedent = tuple(fuzzify_val(col, row[col]) for col in inputs)
            consequent = fuzzify_val(output, row[output])
            key = (antecedent, consequent)
            rules_count[key] = rules_count.get(key, 0) + 1

        # 排序并格式化输出
        sorted_rules = sorted(rules_count.items(), key=lambda x: x[1], reverse=True)
        self.result_text.clear()
        self.result_text.appendPlainText(f"模糊规则（基于分位数量化）: 输入={inputs} -> 输出={output}\n")
        self.result_text.appendPlainText("规则格式: IF (<输入1>, <输入2>, ...) THEN (<输出>) : 支持度\n")
        for i, ((ant, cons), cnt) in enumerate(sorted_rules[:top_k]):
            ant_str = " AND ".join(f"{inputs[j]}={ant[j]}" for j in range(len(inputs)))
            self.result_text.appendPlainText(f"{i+1}. IF {ant_str} THEN {output}={cons} : count={cnt}")
        self.status_bar.showMessage("模糊规则生成完成（分位数方法，供参考）", 5000)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 设置等宽字体以便输出整齐
    app.setFont(QFont("Microsoft YaHei", 10))
    window = TimeSeriesVisualizer()
    window.show()
    sys.exit(app.exec_())
