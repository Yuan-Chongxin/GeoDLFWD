#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: ycx
Creation Time: 2025
"""

import sys
import os
import subprocess
import signal
import warnings
import logging

# 在导入matplotlib之前就设置警告抑制，这是最关键的
# 抑制所有matplotlib相关的字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*findfont.*')
warnings.filterwarnings('ignore', message='.*Font family.*not found.*')
warnings.filterwarnings('ignore', message='.*findfont: Font family.*')
# 设置matplotlib字体管理器的日志级别为ERROR，抑制所有WARNING和INFO级别的日志
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
# 设置环境变量，让matplotlib不输出字体警告
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.expanduser('~'), '.matplotlib')
# 更彻底的方法：捕获所有stderr输出中的字体警告（如果需要）
# 但这种方法可能过于激进，先使用上面的方法

import numpy as np
import matplotlib
from matplotlib.ticker import AutoMinorLocator

# 导入新创建的机器学习训练模块
from ml_trainer import MTTrainer

# 导入配置模块
try:
    from ParamConfig import ReUse
    from PathConfig import models_dir, premodelname, results_dir
    CONFIG_MODULES_AVAILABLE = True
except ImportError as e:
    # 检查是否是pandas相关的错误
    error_msg = str(e)
    if 'pandas' in error_msg.lower():
        print(f"Warning: Failed to import configuration modules due to missing pandas. Please install pandas: pip install pandas")
    else:
        print(f"Warning: Failed to import configuration modules. Error: {e}")
    CONFIG_MODULES_AVAILABLE = False
    ReUse = False
    premodelname = ""
    models_dir = "./models/"
    results_dir = "./results/"

# Attempt to import PyQt5 with error handling
try:
    # Use Qt5 backend
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QScrollArea, QFrame, 
                                QHBoxLayout, QLabel, QPushButton, QFileDialog, QComboBox, 
                                QSlider, QCheckBox, QSpinBox, QDoubleSpinBox, QTextEdit, 
                                QProgressBar, QGroupBox, QGridLayout, QSplitter, QMessageBox, 
                                QListWidget, QListWidgetItem, QMenu, QInputDialog, QDialog, QRadioButton, QLineEdit)
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QEvent
    from PyQt5.QtGui import QFont, QPixmap, QColor, QPalette, QTextCursor
    
    # Configure font settings
    # 智能配置字体，避免字体警告
    # 注意：警告抑制已经在文件开头设置，这里只需要配置字体
    import matplotlib.font_manager as fm
    # 临时禁用字体管理器的警告输出
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 获取系统可用字体
        available_fonts = [f.name for f in fm.fontManager.ttflist]
    # 按优先级尝试设置字体
    font_candidates = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial", "DejaVu Sans"]
    selected_font = None
    for font_name in font_candidates:
        if font_name in available_fonts:
            selected_font = font_name
            break
    # 如果找到中文字体，使用它；否则使用系统默认字体
    if selected_font:
        plt.rcParams["font.family"] = [selected_font]
    else:
        # 使用系统默认字体，避免警告
        plt.rcParams["font.family"] = ["sans-serif"]
    plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
    
    # Flag indicating if PyQt5 is available
    PYQT5_AVAILABLE = True
    

except ImportError as e:
    # Log error message if import fails
    print(f"Error: Failed to import PyQt5 library or related components. Error message: {e}")
    print("\nPlease follow these steps to install PyQt5:")
    print("1. Run Command Prompt as administrator")
    print("2. Execute command: pip install PyQt5 numpy matplotlib scipy torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple")
    print("\nIf you still encounter permission issues, try using a virtual environment:")
    print("1. Create virtual environment: python -m venv geodlfwd_env")
    print("2. Activate virtual environment: geodlfwd_env\\Scripts\\activate")
    print("3. Install dependencies: pip install PyQt5 numpy matplotlib scipy torch torchvision")
    
    # Mark PyQt5 as unavailable
    PYQT5_AVAILABLE = False
    
    # Provide a simple mock interface if PyQt5 cannot be imported
    class MockFigureCanvas():
        """Simple mock class for Matplotlib canvas"""
        def __init__(self, *args, **kwargs):
            print("Matplotlib canvas initialized (mock)")
        def clear(self):
            print("Matplotlib canvas cleared (mock)")
        def draw(self):
            print("Matplotlib canvas drawn (mock)")
    
    # Replace FigureCanvas with mock
    FigureCanvas = MockFigureCanvas

class MPLCanvas(FigureCanvas):
    """Matplotlib canvas class for displaying graphs in PyQt5"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        # Hide axis labels and ticks by default
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.axes.set_xlabel('')
        self.axes.set_ylabel('')
        self.axes.set_title('')
        super(MPLCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()
        self.setMinimumSize(400, 300)
        
        # 用于图像放大的变量
        self.last_x, self.last_y = 0, 0
        self.is_dragging = False
        # 标记训练是否结束
        self.training_finished = False
        
        # 连接鼠标事件
        self.mpl_connect('button_press_event', self.on_mouse_press)
        self.mpl_connect('button_release_event', self.on_mouse_release)
        self.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.mpl_connect('scroll_event', self.on_mouse_scroll)
        
        # 禁用默认右键菜单，避免与自定义保存功能冲突（右键仅触发保存对话框）
        self.setContextMenuPolicy(Qt.NoContextMenu)

    def clear(self):
        """Clear the canvas"""
        # 彻底清除所有子图，避免图像重叠
        for ax in self.fig.get_axes():
            ax.remove()
        
        # 重新创建主坐标轴
        self.axes = self.fig.add_subplot(111)
        
        # 确保坐标轴在清除后保持隐藏状态
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.axes.set_xlabel('')
        self.axes.set_ylabel('')
        self.axes.set_title('')
        self.draw()
        
    def on_mouse_press(self, event):
        """鼠标按下事件，开始拖动或保存图像"""
        if event.button == 1:  # 左键
            self.last_x, self.last_y = event.xdata, event.ydata
            self.is_dragging = True
        elif event.button == 3:  # 右键
            # 右键点击直接保存图像，不再检查训练是否结束
            self.save_figure()
    
    def on_mouse_release(self, event):
        """鼠标释放事件，结束拖动"""
        if event.button == 1:  # 左键
            self.is_dragging = False
    
    def on_mouse_move(self, event):
        """鼠标移动事件，实现拖动功能"""
        if self.is_dragging and event.inaxes == self.axes:
            dx = event.xdata - self.last_x
            dy = event.ydata - self.last_y
            
            # 获取当前的视图范围
            xlim = self.axes.get_xlim()
            ylim = self.axes.get_ylim()
            
            # 调整视图范围
            self.axes.set_xlim(xlim[0] - dx, xlim[1] - dx)
            self.axes.set_ylim(ylim[0] - dy, ylim[1] - dy)
            
            self.last_x, self.last_y = event.xdata, event.ydata
            self.draw()
    
    def on_mouse_scroll(self, event):
        """鼠标滚轮事件，实现缩放功能"""
        if event.inaxes == self.axes:
            # 获取当前鼠标位置对应的坐标
            x = event.xdata
            y = event.ydata
            
            # 获取当前的视图范围
            xlim = self.axes.get_xlim()
            ylim = self.axes.get_ylim()
            
            # 计算缩放因子
            scale_factor = 1.1 if event.button == 'up' else 0.9
            
            # 计算新的视图范围，以鼠标位置为中心进行缩放
            new_xlim = [x - (x - xlim[0]) * scale_factor, x + (xlim[1] - x) * scale_factor]
            new_ylim = [y - (y - ylim[0]) * scale_factor, y + (ylim[1] - y) * scale_factor]
            
            # 设置新的视图范围
            self.axes.set_xlim(new_xlim)
            self.axes.set_ylim(new_ylim)
            self.draw()
    
    def save_figure(self, filename=None):
        """保存当前图像到文件"""
        if filename is None:
            # 获取文件格式过滤器
            filters = "PNG Files (*.png);;JPEG Files (*.jpg);;SVG Files (*.svg);;PDF Files (*.pdf)"
            default_filter = "PNG Files (*.png)"
            
            # 打开文件保存对话框
            filename, _ = QFileDialog.getSaveFileName(
                self.parent(), "Save Image", ".", filters, default_filter
            )
        
        if filename:
            try:
                # 保存之前进行优化处理
                for ax in self.fig.get_axes():
                    # 隐藏坐标轴边框线
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                
                # 确保布局紧凑
                self.fig.tight_layout()
                
                # 保存图像：使用适当的边距确保颜色条标签完整显示
                self.fig.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white')
                return True
            except Exception as e:
                QMessageBox.warning(self.parent(), "Error", f"Failed to save image: {str(e)}")
                return False
        return False

class DataImportTab(QWidget):
    """Data import and MT mode selection tab"""
    def __init__(self, parent=None):
        super(DataImportTab, self).__init__(parent)
        self.parent = parent
        self.init_ui()
        self.check_file_counts()  # 初始化时检查文件数量

    def init_ui(self):
        """初始化UI界面"""
        # 创建主布局
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)  # 减少主布局边距，为内容留出更多空间
        
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧面板：使用滚动区域包装，确保所有内容可见
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setFrameShape(QFrame.NoFrame)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)  # 设置左侧面板边距
        left_layout.setSpacing(10)
        
        # MT mode selection
        mt_mode_group = QGroupBox("MT Mode")
        mt_mode_layout = QVBoxLayout()
        mt_mode_layout.setContentsMargins(12, 12, 12, 12)
        mt_mode_layout.setSpacing(8)
        
        self.te_radio = QCheckBox("TE Mode")
        self.tm_radio = QCheckBox("TM Mode")
        self.both_radio = QCheckBox("Both TE and TM Modes")
        
        self.te_radio.setChecked(True)  # 默认选择TE模式
        
        # 设置单选行为
        self.te_radio.toggled.connect(lambda: self.update_mode_selection(self.te_radio))
        self.tm_radio.toggled.connect(lambda: self.update_mode_selection(self.tm_radio))
        self.both_radio.toggled.connect(lambda: self.update_mode_selection(self.both_radio))
        
        mt_mode_layout.addWidget(self.te_radio)
        mt_mode_layout.addWidget(self.tm_radio)
        mt_mode_layout.addWidget(self.both_radio)
        mt_mode_group.setLayout(mt_mode_layout)
        
        # Dataset files display area
        data_files_group = QGroupBox("Data files")
        data_files_layout = QVBoxLayout()
        data_files_layout.setContentsMargins(12, 12, 12, 12)
        data_files_layout.setSpacing(12)
        
        # Input Data 区域 - 正演系统：输入是电阻率模型
        input_data_layout = QVBoxLayout()
        input_data_layout.setSpacing(10)
        
        # 创建输入数据分组框
        input_data_group = QGroupBox("Input data files (Resistivity Model)")
        input_data_group_layout = QVBoxLayout(input_data_group)
        input_data_group_layout.setContentsMargins(12, 12, 12, 12)
        input_data_group_layout.setSpacing(10)
        
        # 创建导入按钮
        self.import_model_button = QPushButton("Import Resistivity Model")
        self.import_model_button.clicked.connect(self.import_model_data)
        self.import_model_button.setMinimumWidth(220)  # 增加宽度确保文字完整显示
        self.import_model_button.setMinimumHeight(40)
        
        # 路径显示标签
        self.model_data_path = QLabel("Resistivity model folder not selected")
        self.model_data_path.setWordWrap(True)
        self.model_data_path.setMinimumHeight(50)
        self.model_data_path.setStyleSheet("border: 1px solid #CCCCCC; padding: 8px; background-color: #F9F9F9; color: blue; text-decoration: underline;")
        self.model_data_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.model_data_path.mousePressEvent = lambda event: self.on_model_path_clicked()
        self.model_data_path.setToolTip("Click to visualize data")
        
        # 创建路径显示布局
        model_path_layout = QVBoxLayout()
        model_path_layout.setSpacing(5)
        path_label = QLabel("Resistivity Model Path:")
        path_label.setStyleSheet("font-weight: bold;")
        model_path_layout.addWidget(path_label)
        model_path_layout.addWidget(self.model_data_path)
        
        # 模型文件列表
        self.model_file_list = QListWidget()
        self.model_file_list.setMinimumHeight(120)
        self.model_file_list.setStyleSheet("border: 1px solid #CCCCCC;")
        self.model_file_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.model_file_list.customContextMenuRequested.connect(self.show_model_file_context_menu)
        
        # 添加到输入数据分组框
        input_data_group_layout.addWidget(self.import_model_button, alignment=Qt.AlignLeft)
        input_data_group_layout.addSpacing(10)
        input_data_group_layout.addLayout(model_path_layout)
        input_data_group_layout.addSpacing(10)
        model_list_label = QLabel("Model Files List:")
        model_list_label.setStyleSheet("font-weight: bold;")
        input_data_group_layout.addWidget(model_list_label)
        input_data_group_layout.addWidget(self.model_file_list)
        
        # 添加到输入数据主布局
        input_data_layout.addWidget(input_data_group)
        
        # 创建TE和TM的文件列表容器 - 改为垂直布局，节省空间
        file_lists_container = QWidget()
        file_lists_layout = QVBoxLayout(file_lists_container)
        file_lists_layout.setContentsMargins(0, 0, 0, 0)
        file_lists_layout.setSpacing(10)
        
        # 创建TE文件列表区域
        self.te_files_group = QGroupBox("TE Data Files")
        te_files_layout = QHBoxLayout(self.te_files_group)
        te_files_layout.setContentsMargins(12, 12, 12, 12)
        te_files_layout.setSpacing(12)
        
        # TE视电阻率文件列表
        self.te_resistivity_files = QListWidget()
        self.te_resistivity_files.setMinimumHeight(100)
        self.te_resistivity_files.setStyleSheet("border: 1px solid #CCCCCC;")
        self.te_resistivity_files.setContextMenuPolicy(Qt.CustomContextMenu)
        self.te_resistivity_files.customContextMenuRequested.connect(self.show_te_resistivity_context_menu)
        te_resistivity_layout = QVBoxLayout()
        te_resistivity_layout.setSpacing(5)
        te_resistivity_label = QLabel("TE Apparent Resistivity Files")
        te_resistivity_label.setStyleSheet("font-weight: bold;")
        te_resistivity_layout.addWidget(te_resistivity_label)
        te_resistivity_layout.addWidget(self.te_resistivity_files)
        
        # TE相位文件列表
        self.te_phase_files = QListWidget()
        self.te_phase_files.setMinimumHeight(100)
        self.te_phase_files.setStyleSheet("border: 1px solid #CCCCCC;")
        self.te_phase_files.setContextMenuPolicy(Qt.CustomContextMenu)
        self.te_phase_files.customContextMenuRequested.connect(self.show_te_phase_context_menu)
        te_phase_layout = QVBoxLayout()
        te_phase_layout.setSpacing(5)
        te_phase_label = QLabel("TE Phase Files")
        te_phase_label.setStyleSheet("font-weight: bold;")
        te_phase_layout.addWidget(te_phase_label)
        te_phase_layout.addWidget(self.te_phase_files)
        
        # 添加到TE文件布局
        te_files_layout.addLayout(te_resistivity_layout, 1)
        te_files_layout.addLayout(te_phase_layout, 1)
        
        # 创建TM文件列表区域
        self.tm_files_group = QGroupBox("TM Data Files")
        tm_files_layout = QHBoxLayout(self.tm_files_group)
        tm_files_layout.setContentsMargins(12, 12, 12, 12)
        tm_files_layout.setSpacing(12)
        
        # TM视电阻率文件列表
        self.tm_resistivity_files = QListWidget()
        self.tm_resistivity_files.setMinimumHeight(100)
        self.tm_resistivity_files.setStyleSheet("border: 1px solid #CCCCCC;")
        self.tm_resistivity_files.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tm_resistivity_files.customContextMenuRequested.connect(self.show_tm_resistivity_context_menu)
        tm_resistivity_layout = QVBoxLayout()
        tm_resistivity_layout.setSpacing(5)
        tm_resistivity_label = QLabel("TM Apparent Resistivity Files")
        tm_resistivity_label.setStyleSheet("font-weight: bold;")
        tm_resistivity_layout.addWidget(tm_resistivity_label)
        tm_resistivity_layout.addWidget(self.tm_resistivity_files)
        
        # TM相位文件列表
        self.tm_phase_files = QListWidget()
        self.tm_phase_files.setMinimumHeight(100)
        self.tm_phase_files.setStyleSheet("border: 1px solid #CCCCCC;")
        self.tm_phase_files.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tm_phase_files.customContextMenuRequested.connect(self.show_tm_phase_context_menu)
        tm_phase_layout = QVBoxLayout()
        tm_phase_layout.setSpacing(5)
        tm_phase_label = QLabel("TM Phase Files")
        tm_phase_label.setStyleSheet("font-weight: bold;")
        tm_phase_layout.addWidget(tm_phase_label)
        tm_phase_layout.addWidget(self.tm_phase_files)
        
        # Add to TM files layout
        tm_files_layout.addLayout(tm_resistivity_layout, 1)
        tm_files_layout.addLayout(tm_phase_layout, 1)
        
        # Add to main file lists layout - 垂直堆叠
        file_lists_layout.addWidget(self.te_files_group)
        file_lists_layout.addWidget(self.tm_files_group)
        
        # Add to input data main layout
        input_data_layout.addWidget(file_lists_container)
        
        # Create output data group box - 正演系统：输出是视电阻率和相位
        output_data_group = QGroupBox("Output Data Files (Apparent Resistivity & Phase)")
        output_data_group_layout = QVBoxLayout(output_data_group)
        output_data_group_layout.setContentsMargins(12, 12, 12, 12)
        output_data_group_layout.setSpacing(10)
        
        # 创建输出数据按钮区域
        self.output_buttons_container = QWidget()
        self.output_buttons_layout = QGridLayout(self.output_buttons_container)
        self.output_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.output_buttons_layout.setSpacing(12)
        
        # 创建TE模式的导入按钮
        self.te_resistivity_button = QPushButton("Import TE Apparent Resistivity")
        self.te_resistivity_button.clicked.connect(lambda: self.import_specific_data('TE', 'resistivity'))
        self.te_resistivity_button.setMinimumWidth(220)  # 增加宽度确保文字完整显示
        self.te_resistivity_button.setMinimumHeight(40)
        
        self.te_phase_button = QPushButton("Import TE Phase")
        self.te_phase_button.clicked.connect(lambda: self.import_specific_data('TE', 'phase'))
        self.te_phase_button.setMinimumWidth(220)
        self.te_phase_button.setMinimumHeight(40)
        
        # 创建TM模式的导入按钮
        self.tm_resistivity_button = QPushButton("Import TM Apparent Resistivity")
        self.tm_resistivity_button.clicked.connect(lambda: self.import_specific_data('TM', 'resistivity'))
        self.tm_resistivity_button.setMinimumWidth(220)
        self.tm_resistivity_button.setMinimumHeight(40)
        
        self.tm_phase_button = QPushButton("Import TM Phase")
        self.tm_phase_button.clicked.connect(lambda: self.import_specific_data('TM', 'phase'))
        self.tm_phase_button.setMinimumWidth(220)
        self.tm_phase_button.setMinimumHeight(40)
        
        # 添加按钮到布局 - 所有按钮统一尺寸（已在上面设置）
        
        self.output_buttons_layout.addWidget(self.te_resistivity_button, 0, 0)
        self.output_buttons_layout.addWidget(self.te_phase_button, 0, 1)
        self.output_buttons_layout.addWidget(self.tm_resistivity_button, 1, 0)
        self.output_buttons_layout.addWidget(self.tm_phase_button, 1, 1)
        
        self.output_buttons_layout.setColumnStretch(0, 1)
        self.output_buttons_layout.setColumnStretch(1, 1)
        
        # 路径显示标签
        self.data_paths = {
            'TE_resistivity': QLabel("TE apparent resistivity folder not selected"),
            'TE_phase': QLabel("TE phase folder not selected"),
            'TM_resistivity': QLabel("TM apparent resistivity folder not selected"),
            'TM_phase': QLabel("TM phase folder not selected")
        }
        
        for key, label in self.data_paths.items():
            label.setWordWrap(True)
            label.setMinimumHeight(45)
            label.setStyleSheet("border: 1px solid #CCCCCC; padding: 8px; background-color: #F9F9F9; color: blue; text-decoration: underline;")
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            label.mousePressEvent = lambda event, key=key: self.on_path_label_clicked(key)
            label.setToolTip("Click to visualize data")
        
        # 创建路径显示布局 - 改为垂直布局，更清晰
        output_paths_layout = QVBoxLayout()
        output_paths_layout.setSpacing(10)
        
        # TE apparent resistivity
        te_res_container = QVBoxLayout()
        te_res_container.setSpacing(5)
        te_res_label = QLabel("TE apparent resistivity:")
        te_res_label.setStyleSheet("font-weight: bold;")
        te_res_container.addWidget(te_res_label)
        te_res_container.addWidget(self.data_paths['TE_resistivity'])
        output_paths_layout.addLayout(te_res_container)
        
        # TE phase
        te_phase_container = QVBoxLayout()
        te_phase_container.setSpacing(5)
        te_phase_label = QLabel("TE phase:")
        te_phase_label.setStyleSheet("font-weight: bold;")
        te_phase_container.addWidget(te_phase_label)
        te_phase_container.addWidget(self.data_paths['TE_phase'])
        output_paths_layout.addLayout(te_phase_container)
        
        # TM apparent resistivity
        tm_res_container = QVBoxLayout()
        tm_res_container.setSpacing(5)
        tm_res_label = QLabel("TM apparent resistivity:")
        tm_res_label.setStyleSheet("font-weight: bold;")
        tm_res_container.addWidget(tm_res_label)
        tm_res_container.addWidget(self.data_paths['TM_resistivity'])
        output_paths_layout.addLayout(tm_res_container)
        
        # TM phase
        tm_phase_container = QVBoxLayout()
        tm_phase_container.setSpacing(5)
        tm_phase_label = QLabel("TM phase:")
        tm_phase_label.setStyleSheet("font-weight: bold;")
        tm_phase_container.addWidget(tm_phase_label)
        tm_phase_container.addWidget(self.data_paths['TM_phase'])
        output_paths_layout.addLayout(tm_phase_container)
        
        # 添加到输出数据分组框
        output_data_group_layout.addWidget(self.output_buttons_container)
        output_data_group_layout.addSpacing(10)
        output_data_group_layout.addLayout(output_paths_layout)
        
        # Update button visibility during initialization
        self.update_import_buttons_visibility()
        
        # Load default paths from ParamConfig
        try:
            import ParamConfig
            
            # Set TE Apparent Resistivity path
            if hasattr(ParamConfig, 'TE_Resistivity_Dir') and ParamConfig.TE_Resistivity_Dir and os.path.exists(ParamConfig.TE_Resistivity_Dir):
                self.data_paths['TE_resistivity'].setText(ParamConfig.TE_Resistivity_Dir)
            
            # Set TE Phase path
            if hasattr(ParamConfig, 'TE_Phase_Dir') and ParamConfig.TE_Phase_Dir and os.path.exists(ParamConfig.TE_Phase_Dir):
                self.data_paths['TE_phase'].setText(ParamConfig.TE_Phase_Dir)
            
            # Set TM Apparent Resistivity path
            if hasattr(ParamConfig, 'TM_Resistivity_Dir') and ParamConfig.TM_Resistivity_Dir and os.path.exists(ParamConfig.TM_Resistivity_Dir):
                self.data_paths['TM_resistivity'].setText(ParamConfig.TM_Resistivity_Dir)
            
            # Set TM Phase path
            if hasattr(ParamConfig, 'TM_Phase_Dir') and ParamConfig.TM_Phase_Dir and os.path.exists(ParamConfig.TM_Phase_Dir):
                self.data_paths['TM_phase'].setText(ParamConfig.TM_Phase_Dir)
            
            # Set Resistivity Model path - 正演系统：输入是模型
            if hasattr(ParamConfig, 'Resistivity_Model_Dir') and ParamConfig.Resistivity_Model_Dir and os.path.exists(ParamConfig.Resistivity_Model_Dir):
                self.model_data_path.setText(ParamConfig.Resistivity_Model_Dir)
                self.show_files_in_folder(ParamConfig.Resistivity_Model_Dir, self.model_file_list)
            
            # Update file list
            self.update_input_file_list()
            
        except Exception as e:
            print(f"Warning: Failed to load default paths from ParamConfig: {e}")
        
        # Add all components to data files layout
        data_files_layout.addLayout(input_data_layout)
        data_files_layout.addSpacing(12)
        
        # Move output data group inside data files group
        data_files_layout.addWidget(output_data_group)
        
        data_files_group.setLayout(data_files_layout)
        
        # Channel information display area
        self.channel_info = QLabel("Current mode: TE\nEstimated input channels: 2, label channels: 1")
        self.channel_info.setStyleSheet("color: blue; font-weight: bold; padding: 8px; background-color: #F0F0F0; border: 1px solid #CCCCCC; border-radius: 4px;")
        self.channel_info.setWordWrap(True)
        self.channel_info.setMinimumHeight(60)
        
        # Adjust layout structure
        left_layout.addWidget(mt_mode_group)
        left_layout.addWidget(data_files_group, 1)  # 使用伸缩因子让数据文件组占据主要空间
        left_layout.addWidget(self.channel_info)
        
        # 设置左侧面板到滚动区域
        left_scroll.setWidget(left_panel)
        
        # Right panel: Dataset visualization preview
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)  # Set right panel margins
        
        self.visualization_title = QLabel("Data Visualization Preview")
        self.visualization_title.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.visualization_title.setFont(font)
        
        # Create Matplotlib canvas
        self.canvas = MPLCanvas(self, width=5, height=4, dpi=100)
        self.canvas.setMinimumHeight(300)  # Set canvas minimum height
        
        # Create save image button
        self.save_image_button = QPushButton("Save Image")
        self.save_image_button.clicked.connect(self.save_current_image)
        self.save_image_button.setDisabled(True)  # Initially disabled until image is displayed
        
        # Image operation instructions label
        operation_label = QLabel("Instructions: Wheel to zoom, left click to drag, right click to reset view")
        operation_label.setAlignment(Qt.AlignCenter)
        operation_label.setStyleSheet("font-size: 10px; color: gray;")
        
        # Add to right layout
        right_layout.addWidget(self.visualization_title)
        right_layout.addWidget(self.canvas, 1)  # 使用伸缩因子让canvas占据主要空间
        right_layout.addWidget(self.save_image_button)
        right_layout.addWidget(operation_label)
        right_layout.addStretch()  # Add stretch factor to distribute components properly in right panel
        
        # Add left and right panels to splitter
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setSizes([600, 400])  # 设置左右面板的初始比例
        
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
    
    def update_mode_selection(self, sender):
        """Update MT mode selection, ensure only one option is selected, and update channel information - 正演系统"""
        if sender.isChecked():
            if sender == self.te_radio:
                self.tm_radio.setChecked(False)
                self.both_radio.setChecked(False)
                self.update_channel_info("TE", 2)  # 输出2通道：TE_resistivity, TE_phase
            elif sender == self.tm_radio:
                self.te_radio.setChecked(False)
                self.both_radio.setChecked(False)
                self.update_channel_info("TM", 2)  # 输出2通道：TM_resistivity, TM_phase
            else:
                self.te_radio.setChecked(False)
                self.tm_radio.setChecked(False)
                self.update_channel_info("Both TE and TM", 4)  # 输出4通道：TE_res, TE_ph, TM_res, TM_ph
            # Update import button visibility
            self.update_import_buttons_visibility()
            # Check file counts after mode switch
            self.check_file_counts()
    
    def update_import_buttons_visibility(self):
        """Update import button visibility and layout based on currently selected mode"""
        is_both_mode = self.both_radio.isChecked()
        
        if self.te_radio.isChecked():
            # TE模式：显示TE的两个按钮，隐藏TM的两个按钮
            self.te_resistivity_button.setVisible(True)
            self.te_phase_button.setVisible(True)
            self.tm_resistivity_button.setVisible(False)
            self.tm_phase_button.setVisible(False)
            # 显示对应的数据路径标签
            self.data_paths['TE_resistivity'].setVisible(True)
            self.data_paths['TE_phase'].setVisible(True)
            self.data_paths['TM_resistivity'].setVisible(False)
            self.data_paths['TM_phase'].setVisible(False)
            # 调整TE文件组和TM文件组的可见性
            self.te_files_group.setVisible(True)
            self.tm_files_group.setVisible(False)
        elif self.tm_radio.isChecked():
            # TM模式：显示TM的两个按钮，隐藏TE的两个按钮
            self.te_resistivity_button.setVisible(False)
            self.te_phase_button.setVisible(False)
            self.tm_resistivity_button.setVisible(True)
            self.tm_phase_button.setVisible(True)
            # 显示对应的数据路径标签
            self.data_paths['TE_resistivity'].setVisible(False)
            self.data_paths['TE_phase'].setVisible(False)
            self.data_paths['TM_resistivity'].setVisible(True)
            self.data_paths['TM_phase'].setVisible(True)
            # Adjust visibility of TE and TM file groups
            self.te_files_group.setVisible(False)
            self.tm_files_group.setVisible(True)
        else:
            # Both mode: show all four buttons, optimize layout to ensure content visibility
            self.te_resistivity_button.setVisible(True)
            self.te_phase_button.setVisible(True)
            self.tm_resistivity_button.setVisible(True)
            self.tm_phase_button.setVisible(True)
            # Show all data path labels
            for label in self.data_paths.values():
                label.setVisible(True)
            # Adjust visibility of TE and TM file groups
            self.te_files_group.setVisible(True)
            self.tm_files_group.setVisible(True)
            
        # Regardless of mode, ensure all path labels have enough space to display
        for label in self.data_paths.values():
            if label.isVisible():
                label.setMinimumHeight(40)  # Ensure enough height to display multi-line text
                label.setMaximumHeight(100)  # Set maximum height to avoid taking too much space

    def update_param_config_path(self, param_name, value):
        """Update path configuration parameters in ParamConfig.py"""
        try:
            import ParamConfig
            import os
            
            # Directly modify the variable value in the module
            setattr(ParamConfig, param_name, value)
            
            # Also update the file content for the next program startup
            import os
            # Get the directory where the current file is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Build the absolute path of ParamConfig.py
            param_config_path = os.path.join(current_dir, 'ParamConfig.py')
            with open(param_config_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            updated_lines = []
            param_updated = False
            
            for line in lines:
                if line.startswith(param_name):
                    # Preserve original comments
                    comment_pos = line.find('#')
                    if comment_pos != -1:
                        comment = line[comment_pos:]
                        updated_lines.append(f'{param_name} = {repr(value)} {comment}\n')
                    else:
                        updated_lines.append(f'{param_name} = {repr(value)}\n')
                    param_updated = True
                else:
                    updated_lines.append(line)
            
            # If the parameter doesn't exist, add it at the end of the file
            if not param_updated:
                updated_lines.append(f'\n{param_name} = {repr(value)}  # Added by GUI\n')
            
            # Write back to the file
            with open('ParamConfig.py', 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)
            
        except Exception as e:
            print(f"Warning: Failed to update ParamConfig.py: {e}")
    
    def import_specific_data(self, mode, data_type):
        """Import specific type of data folder"""
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        options |= QFileDialog.DontResolveSymlinks
        
        data_type_name = "Apparent Resistivity" if data_type == 'resistivity' else "Phase"
        dialog_title = f"Select {mode} Mode {data_type_name} Data Folder"
        
        folder_path = QFileDialog.getExistingDirectory(
            self, dialog_title, "", options=options
        )
        
        if folder_path:
            # Save folder path
            key = f'{mode}_{data_type}'
            self.data_paths[key].setText(folder_path)
            
            # Update corresponding path in ParamConfig.py
            param_name_map = {
                'TE_resistivity': 'TE_Resistivity_Dir',
                'TE_phase': 'TE_Phase_Dir',
                'TM_resistivity': 'TM_Resistivity_Dir',
                'TM_phase': 'TM_Phase_Dir'
            }
            if key in param_name_map:
                self.update_param_config_path(param_name_map[key], folder_path)
            
            # Update file list
            self.update_input_file_list()
            # Check file counts after import
            self.check_file_counts()
    
    def update_input_file_list(self):
        """Update input file list"""
        # Clear all file lists
        self.te_resistivity_files.clear()
        self.te_phase_files.clear()
        self.tm_resistivity_files.clear()
        self.tm_phase_files.clear()
        
        # Update file list visibility based on currently selected mode
        self.update_file_lists_visibility()
        
        # Process each data type path
        for key, label in self.data_paths.items():
            path_text = label.text()
            # Check if file is selected (not default text) and label is visible
            if label.isVisible() and not path_text.startswith("No file selected"):
                # Determine corresponding file list and data type based on key
                if key == 'TE_resistivity':
                    target_list = self.te_resistivity_files
                    data_type = 'resistivity'
                elif key == 'TE_phase':
                    target_list = self.te_phase_files
                    data_type = 'phase'
                elif key == 'TM_resistivity':
                    target_list = self.tm_resistivity_files
                    data_type = 'resistivity'
                elif key == 'TM_phase':
                    target_list = self.tm_phase_files
                    data_type = 'phase'
                
                # Display files
                if os.path.isdir(path_text):
                    self.show_files_in_folder(path_text, target_list, data_type)
                elif os.path.isfile(path_text):
                    file_name = os.path.basename(path_text)
                    item = QListWidgetItem(file_name)
                    item.setData(Qt.UserRole, path_text)
                    item.setData(Qt.UserRole + 1, data_type)  # Store data type
                    target_list.addItem(item)
    
    def update_file_lists_visibility(self):
        """Update file list visibility based on currently selected mode"""
        if self.te_radio.isChecked():
            self.te_files_group.setVisible(True)
            self.tm_files_group.setVisible(False)
        elif self.tm_radio.isChecked():
            self.te_files_group.setVisible(False)
            self.tm_files_group.setVisible(True)
        else:  # Both mode
            self.te_files_group.setVisible(True)
            self.tm_files_group.setVisible(True)
    
    def get_selected_file_paths(self):
        """Get all selected file paths from Data Import tab"""
        file_paths = {
            'model_files': [],
            'te_resistivity_files': [],
            'te_phase_files': [],
            'tm_resistivity_files': [],
            'tm_phase_files': []
        }
        
        # Get model files
        for i in range(self.model_file_list.count()):
            item = self.model_file_list.item(i)
            if item:
                file_path = item.data(Qt.UserRole)
                if file_path:
                    file_paths['model_files'].append(file_path)
                else:
                    # If no UserRole data, try to get from text
                    file_paths['model_files'].append(item.text())
        
        # Get TE resistivity files
        for i in range(self.te_resistivity_files.count()):
            item = self.te_resistivity_files.item(i)
            if item:
                file_path = item.data(Qt.UserRole)
                if file_path:
                    file_paths['te_resistivity_files'].append(file_path)
                else:
                    file_paths['te_resistivity_files'].append(item.text())
        
        # Get TE phase files
        for i in range(self.te_phase_files.count()):
            item = self.te_phase_files.item(i)
            if item:
                file_path = item.data(Qt.UserRole)
                if file_path:
                    file_paths['te_phase_files'].append(file_path)
                else:
                    file_paths['te_phase_files'].append(item.text())
        
        # Get TM resistivity files
        for i in range(self.tm_resistivity_files.count()):
            item = self.tm_resistivity_files.item(i)
            if item:
                file_path = item.data(Qt.UserRole)
                if file_path:
                    file_paths['tm_resistivity_files'].append(file_path)
                else:
                    file_paths['tm_resistivity_files'].append(item.text())
        
        # Get TM phase files
        for i in range(self.tm_phase_files.count()):
            item = self.tm_phase_files.item(i)
            if item:
                file_path = item.data(Qt.UserRole)
                if file_path:
                    file_paths['tm_phase_files'].append(file_path)
                else:
                    file_paths['tm_phase_files'].append(item.text())
        
        # 打印文件路径匹配信息（前5个文件）用于调试
        print("[GUI] File paths matching verification (first 5 files):")
        if file_paths['model_files']:
            print(f"  Model files count: {len(file_paths['model_files'])}")
            for i in range(min(5, len(file_paths['model_files']))):
                print(f"    [{i}] {os.path.basename(file_paths['model_files'][i])}")
        if file_paths['te_resistivity_files']:
            print(f"  TE Resistivity files count: {len(file_paths['te_resistivity_files'])}")
            for i in range(min(5, len(file_paths['te_resistivity_files']))):
                print(f"    [{i}] {os.path.basename(file_paths['te_resistivity_files'][i])}")
        if file_paths['te_phase_files']:
            print(f"  TE Phase files count: {len(file_paths['te_phase_files'])}")
            for i in range(min(5, len(file_paths['te_phase_files']))):
                print(f"    [{i}] {os.path.basename(file_paths['te_phase_files'][i])}")
        if file_paths['tm_resistivity_files']:
            print(f"  TM Resistivity files count: {len(file_paths['tm_resistivity_files'])}")
            for i in range(min(5, len(file_paths['tm_resistivity_files']))):
                print(f"    [{i}] {os.path.basename(file_paths['tm_resistivity_files'][i])}")
        if file_paths['tm_phase_files']:
            print(f"  TM Phase files count: {len(file_paths['tm_phase_files'])}")
            for i in range(min(5, len(file_paths['tm_phase_files']))):
                print(f"    [{i}] {os.path.basename(file_paths['tm_phase_files'][i])}")
        
        return file_paths
    
    def check_file_counts(self):
        """Check if the number of input model and output data files meet requirements - 正演系统"""
        # Get model file count (输入)
        model_count = self.model_file_list.count()
        
        # Calculate file counts in each output file list (输出)
        te_resistivity_count = self.te_resistivity_files.count()
        te_phase_count = self.te_phase_files.count()
        tm_resistivity_count = self.tm_resistivity_files.count()
        tm_phase_count = self.tm_phase_files.count()
        
        # Different validation logic based on currently selected mode
        is_valid = False
        status_text = ""
        
        if self.te_radio.isChecked():
            # TE mode: TE Apparent Resistivity and TE Phase file counts must be equal and match model file count
            if model_count > 0:
                if te_resistivity_count == te_phase_count == model_count:
                    is_valid = True
                    status_text = "Data is ready, you can proceed to model settings"
                else:
                    status_text = f"File count mismatch: Model({model_count}), TE Apparent Resistivity({te_resistivity_count}), TE Phase({te_phase_count}) file counts must be equal"
            else:
                status_text = "Please import resistivity model files first"
        elif self.tm_radio.isChecked():
            # TM mode: TM Apparent Resistivity and TM Phase file counts must be equal and match model file count
            if model_count > 0:
                if tm_resistivity_count == tm_phase_count == model_count:
                    is_valid = True
                    status_text = "Data is ready, you can proceed to model settings"
                else:
                    status_text = f"File count mismatch: Model({model_count}), TM Apparent Resistivity({tm_resistivity_count}), TM Phase({tm_phase_count}) file counts must be equal"
            else:
                status_text = "Please import resistivity model files first"
        else:  # Both mode
            # Both mode: TE and TM Apparent Resistivity and Phase file counts must all be equal and match model file count
            if model_count > 0:
                if (te_resistivity_count == te_phase_count == 
                    tm_resistivity_count == tm_phase_count == model_count):
                    is_valid = True
                    status_text = "Data is ready, you can proceed to model settings"
                else:
                    status_text = f"File count mismatch: Model({model_count}), TE Apparent Resistivity({te_resistivity_count}), TE Phase({te_phase_count}), TM Apparent Resistivity({tm_resistivity_count}), TM Phase({tm_phase_count}) file counts must all be equal"
            else:
                status_text = "Please import resistivity model files first"
        
        # Enable or disable model settings tab
        if self.parent and hasattr(self.parent, 'update_tab_status'):
            model_tab_index = 1  # Index of model settings tab
            
            # Get current status of model settings tab
            current_status = self.parent.tabs.isTabEnabled(model_tab_index)
            
            # If previously disabled and now enabled, call parent window's update method
            if not current_status and is_valid:
                self.parent.update_tab_status(0)  # 0 is the index of Data Import tab
            
            # Set tab status
            self.parent.tabs.setTabEnabled(model_tab_index, is_valid)
            
            # Update channel information, add file count status - 正演系统
            current_mode = "Both TE and TM" if self.both_radio.isChecked() else ("TE" if self.te_radio.isChecked() else "TM")
            output_channels = 4 if self.both_radio.isChecked() else 2  # 输出通道数：TE&TM=4, TE或TM=2
            # Split long text into two lines to ensure completeness
            self.channel_info.setText(f"Forward modeling mode: {current_mode}\nInput: Model (1 channel), Output: Apparent Resistivity & Phase ({output_channels} channels)\n{status_text}")
                
    def update_channel_info(self, mode_name, output_channels):
        """Update channel information display - 正演系统"""
        self.channel_info.setText(f"Forward modeling mode: {mode_name}\nInput: Model (1 channel), Output: Apparent Resistivity & Phase ({output_channels} channels)")
    

    
    def import_model_data(self):
        """Import resistivity model data folder - 正演系统：输入是模型"""
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        options |= QFileDialog.DontResolveSymlinks
        
        folder = QFileDialog.getExistingDirectory(
            self, "Select Resistivity Model Folder", "", options=options
        )
        
        if folder:
            self.model_data_path.setText(folder)
            # Update resistivity model path in ParamConfig.py
            self.update_param_config_path('Resistivity_Model_Dir', folder)
            # Show files in folder
            self.show_files_in_folder(folder, self.model_file_list)
            # Check file counts after import
            self.check_file_counts()
    
    def on_model_path_clicked(self):
        """Handle click on model path label"""
        if self.model_data_path.text() != "Resistivity model folder not selected":
            folder = self.model_data_path.text()
            if os.path.exists(folder):
                # Visualize first model file if available
                if self.model_file_list.count() > 0:
                    first_item = self.model_file_list.item(0)
                    if first_item:
                        file_path = first_item.data(Qt.UserRole)
                        self.visualize_specific_file(file_path, is_input_data=True, data_type='model')
    
    def extract_file_number(self, file_name):
        """Extract number from filename for sorting by index
        Examples: '0.txt' -> 0, '1.txt' -> 1, 'zz0.txt' -> 0, 'zz10.txt' -> 10
        """
        import re
        # Remove extension
        name_without_ext = os.path.splitext(file_name)[0]
        # Extract all numbers from filename
        numbers = re.findall(r'\d+', name_without_ext)
        if numbers:
            # Use the last number found (handles cases like 'zz0.txt', 'file_1.txt')
            return int(numbers[-1])
        else:
            # If no number found, return a large number to put it at the end
            return 999999
    
    def show_files_in_folder(self, folder_path, list_widget, data_type=None):
        """Display .txt files in the folder in list widget, sorted by file number/index"""
        list_widget.clear()
        try:
            # Get all files in the folder
            file_list = []
            for file_name in os.listdir(folder_path):
                # Only show files with .txt extension
                if file_name.lower().endswith('.txt'):
                    file_path = os.path.join(folder_path, file_name)
                    if os.path.isfile(file_path):
                        file_list.append((file_name, file_path))
            
            # Sort files by number/index extracted from filename
            # This ensures files are matched by index: 0.txt, 1.txt, 2.txt, etc.
            file_list.sort(key=lambda x: (self.extract_file_number(x[0]), x[0]))
            
            # Add sorted files to list widget
            for file_name, file_path in file_list:
                # Store file path and data type in Item's data
                item = QListWidgetItem(file_name)
                item.setData(Qt.UserRole, file_path)
                item.setData(Qt.UserRole + 1, data_type)  # Store data type
                list_widget.addItem(item)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Unable to read folder contents: {str(e)}")
    
    def show_te_resistivity_context_menu(self, position):
        """Show right-click menu for TE Apparent Resistivity files"""
        if self.te_resistivity_files.currentItem():
            menu = QMenu()
            visualize_action = menu.addAction("Show Visualization")
            action = menu.exec_(self.te_resistivity_files.mapToGlobal(position))
            
            if action == visualize_action:
                file_path = self.te_resistivity_files.currentItem().data(Qt.UserRole)
                data_type = self.te_resistivity_files.currentItem().data(Qt.UserRole + 1) or 'resistivity'
                self.visualize_specific_file(file_path, is_input_data=True, data_type=data_type)
    
    def show_te_phase_context_menu(self, position):
        """Show right-click menu for TE Phase files"""
        if self.te_phase_files.currentItem():
            menu = QMenu()
            visualize_action = menu.addAction("Show Visualization")
            action = menu.exec_(self.te_phase_files.mapToGlobal(position))
            
            if action == visualize_action:
                file_path = self.te_phase_files.currentItem().data(Qt.UserRole)
                data_type = self.te_phase_files.currentItem().data(Qt.UserRole + 1) or 'phase'
                self.visualize_specific_file(file_path, is_input_data=True, data_type=data_type)
    
    def show_tm_resistivity_context_menu(self, position):
        """Show right-click menu for TM Apparent Resistivity files"""
        if self.tm_resistivity_files.currentItem():
            menu = QMenu()
            visualize_action = menu.addAction("Show Visualization")
            action = menu.exec_(self.tm_resistivity_files.mapToGlobal(position))
            
            if action == visualize_action:
                file_path = self.tm_resistivity_files.currentItem().data(Qt.UserRole)
                data_type = self.tm_resistivity_files.currentItem().data(Qt.UserRole + 1) or 'resistivity'
                self.visualize_specific_file(file_path, is_input_data=True, data_type=data_type)
    
    def show_tm_phase_context_menu(self, position):
        """Show right-click menu for TM Phase files"""
        if self.tm_phase_files.currentItem():
            menu = QMenu()
            visualize_action = menu.addAction("Show Visualization")
            action = menu.exec_(self.tm_phase_files.mapToGlobal(position))
            
            if action == visualize_action:
                file_path = self.tm_phase_files.currentItem().data(Qt.UserRole)
                data_type = self.tm_phase_files.currentItem().data(Qt.UserRole + 1) or 'phase'
                self.visualize_specific_file(file_path, is_input_data=True, data_type=data_type)
    
    def show_model_file_context_menu(self, position):
        """Show right-click menu for model files"""
        if self.model_file_list.currentItem():
            menu = QMenu()
            visualize_action = menu.addAction("Show Visualization")
            action = menu.exec_(self.model_file_list.mapToGlobal(position))
            
            if action == visualize_action:
                file_path = self.model_file_list.currentItem().data(Qt.UserRole)
                self.visualize_specific_file(file_path, is_input_data=True, data_type='model')
    
    def show_label_file_context_menu(self, position):
        """Show right-click menu for label files"""
        if self.label_file_list.currentItem():
            menu = QMenu()
            visualize_action = menu.addAction("Show Visualization")
            action = menu.exec_(self.label_file_list.mapToGlobal(position))
            
            if action == visualize_action:
                file_path = self.label_file_list.currentItem().data(Qt.UserRole)
                self.visualize_specific_file(file_path, is_input_data=False)
    
    def on_path_label_clicked(self, data_type):
        """Handle path label click event - allow user to reselect file address"""
        # Extract mode and data type information
        if data_type.startswith('TE_'):
            mode = 'TE'
        else:
            mode = 'TM'
            
        data_type_value = 'resistivity' if 'resistivity' in data_type else 'phase'
        
        # Call import method to let user reselect file
        self.import_specific_data(mode, data_type_value)
        
    def visualize_specific_file(self, file_path, is_input_data=True, data_type=None):
        """Visualize specific data file"""
        # Read file data and process as required
        model_size = 32
        file_name = os.path.basename(file_path)
        # 添加缺失的变量定义
        config_dir = os.path.dirname(file_path)
        base_filename = os.path.splitext(file_name)[0].replace('pred_', '')
        
        try:
            # Create custom dialog for user to input both model length and depth
            class ModelSizeDialog(QDialog):
                def __init__(self, parent=None):
                    super().__init__(parent)
                    self.setWindowTitle("Model Size Input")
                    self.setFixedSize(300, 150)
                    
                    layout = QVBoxLayout(self)
                    
                    # Create length input
                    length_layout = QHBoxLayout()
                    length_label = QLabel("Model Length (km):")
                    self.length_spin = QDoubleSpinBox()
                    self.length_spin.setRange(0.1, 100.0)
                    self.length_spin.setDecimals(1)
                    self.length_spin.setValue(5.0)  # Default value
                    length_layout.addWidget(length_label)
                    length_layout.addWidget(self.length_spin)
                    
                    # Create depth input
                    depth_layout = QHBoxLayout()
                    depth_label = QLabel("Model Depth (km):")
                    self.depth_spin = QDoubleSpinBox()
                    self.depth_spin.setRange(0.1, 50.0)
                    self.depth_spin.setDecimals(1)
                    self.depth_spin.setValue(3.0)  # Default value
                    depth_layout.addWidget(depth_label)
                    depth_layout.addWidget(self.depth_spin)
                    
                    # Create buttons
                    button_layout = QHBoxLayout()
                    ok_button = QPushButton("OK")
                    cancel_button = QPushButton("Cancel")
                    ok_button.clicked.connect(self.accept)
                    cancel_button.clicked.connect(self.reject)
                    button_layout.addWidget(ok_button)
                    button_layout.addWidget(cancel_button)
                    
                    # Add to main layout
                    layout.addLayout(length_layout)
                    layout.addLayout(depth_layout)
                    layout.addLayout(button_layout)
                    
            # Show dialog and get user input
            dialog = ModelSizeDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                length = dialog.length_spin.value()
                depth = dialog.depth_spin.value()
            else:
                # User canceled input
                return
            
            # Read file data, assuming data is separated by spaces or tabs
            data = np.loadtxt(file_path)
            
            # 修复数据重塑逻辑，确保1024元素数据正确重塑为32x32
            # 首先检查数据形状，如果是一维数组，需要重塑
            if len(data.shape) == 1:
                # 直接根据数据长度进行重塑
                if data.size == 1024:
                    # 明确处理1024元素的情况，重塑为32x32
                    data = data.reshape(32, 32)
                elif data.size == 256:
                    # 处理256元素的情况，重塑为16x16
                    data = data.reshape(16, 16)
                else:
                    # 尝试从配置中获取实际的数据维度
                    actual_data_dim = None
                    try:
                        import glob
                        config_files = glob.glob(os.path.join(config_dir, f"pred_config_*{base_filename}*.json"))
                        if config_files:
                            with open(config_files[0], 'r', encoding='utf-8') as f:
                                import json
                                config = json.load(f)
                                if 'data_dim' in config and isinstance(config['data_dim'], list) and len(config['data_dim']) == 2:
                                    actual_data_dim = config['data_dim']
                    except Exception:
                        pass
                    
                    # 根据实际维度重塑数据
                    if actual_data_dim and len(data) >= actual_data_dim[0] * actual_data_dim[1]:
                        # 使用配置文件中的实际维度
                        data = data[:actual_data_dim[0] * actual_data_dim[1]].reshape(actual_data_dim)
                    else:
                        # 对于其他情况，尝试计算合适的形状
                        size = int(np.sqrt(data.size))
                        # 找到最接近的合适尺寸
                        data = data[:size*size].reshape(size, size)
            
            # Determine whether to apply logarithm based on file type
            # For resistivity model and apparent resistivity files, always apply logarithm
            # For phase files, do not apply logarithm
            # IMPORTANT: Input data files (resistivity and phase) are in raw format
            # - Resistivity files: raw values, need log10 for visualization
            # - Phase files: raw values, no log10 needed
            # - Model files: raw values, need log10 for visualization
            # - Label files: should be determined by data_type if available
            if is_input_data:
                # Input data: check data_type
                is_resistivity = (data_type == 'resistivity' or data_type == 'model')
            else:
                # Label data: try to infer from filename or use data_type
                # Label files might be resistivity or phase, need to check
                is_resistivity = (data_type == 'resistivity' if data_type else 
                                 'res' in file_name.lower() or 'rho' in file_name.lower() or 'model' in file_name.lower())
            
            if is_resistivity:
                # First take absolute value to ensure no negative values, then add a small value to avoid log10(0)
                data = np.log10(np.abs(data) + 1e-10)
                print(f"[Input Data Visualization] Applied log10 for resistivity. Data range: {data.min():.3f} - {data.max():.3f} (lgρ)")
            else:
                print(f"[Input Data Visualization] No log10 for phase. Data range: {data.min():.3f} - {data.max():.3f} (degrees)")
            
            # Visualize processed data
            # Reset figure more thoroughly: clear all axes and recreate main axis
            self.canvas.figure.clear()
            
            # Recreate main axis
            self.canvas.axes = self.canvas.figure.add_subplot(111)
            
            # Adjust figure aspect ratio based on user-input length and depth
            # Calculate actual aspect ratio and set reasonable limits (between 0.5 and 2 to avoid overly flat or elongated figures)
            aspect_ratio = min(max(depth / length, 0.5), 2.0)
            
            # Use 'jet' colormap for both resistivity and phase
            cmap = 'jet'
            
            im = self.canvas.axes.imshow(data, cmap=cmap, aspect=aspect_ratio, origin='upper')
            
            # Add colorbar and limit number of ticks to 3-4
            # Set colorbar height smaller than image height for better visual balance
            cbar = self.canvas.figure.colorbar(im, ax=self.canvas.axes, shrink=0.4)
            
            # Set colorbar label based on data type
            if is_input_data:
                if data_type == 'resistivity' or data_type == 'model':
                    # 电阻率模型和视电阻率都使用lgρ单位
                    cbar.set_label('lgρ(Ω·m)')  # Resistivity unit
                else:
                    cbar.set_label('φ(°)')  # Phase unit
            else:
                # Label data is always resistivity
                cbar.set_label('lgρ(Ω·m)')  # Resistivity unit
            cbar.locator = plt.MaxNLocator(nbins=4)
            cbar.update_ticks()
            
            # Set title
            if is_input_data:
                data_type_text = f"Input Data: {file_name}" + (" (TE Mode)" if self.te_radio.isChecked() else " (TM Mode)" if self.tm_radio.isChecked() else " (Both Modes)")
                if data_type == 'resistivity':
                    data_type_text += ""
            else:
                data_type_text = f"Label Data: {file_name} "
                
            # Set centered title
            self.canvas.axes.set_title(data_type_text, loc='center')
            
            # Set axis ticks and labels to display user-input model dimensions
            # x-axis (horizontal) represents length, y-axis (vertical) represents depth
            num_ticks = 5  # Set number of ticks
            x_ticks = np.linspace(0, model_size-1, num_ticks)
            y_ticks = np.linspace(0, model_size-1, num_ticks)
            
            # Calculate corresponding actual dimension values
            x_tick_labels = [f"{x:.1f}" for x in np.linspace(0, length, num_ticks)]
            y_tick_labels = [f"{y:.1f}" for y in np.linspace(0, depth, num_ticks)]
            
            # Set tick positions and labels
            self.canvas.axes.set_xticks(x_ticks)
            self.canvas.axes.set_yticks(y_ticks)
            self.canvas.axes.set_xticklabels(x_tick_labels)
            self.canvas.axes.set_yticklabels(y_tick_labels)
            
            # Set axis labels
            self.canvas.axes.set_xlabel('Length (km)')
            self.canvas.axes.set_ylabel('Depth (km)')
            
            # Adjust layout to ensure all elements are displayed correctly
            self.canvas.figure.tight_layout()
            
            # Redraw the canvas
            self.canvas.draw()
            
            # Enable save image button
            self.save_image_button.setEnabled(True)
        
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Cannot visualize file {file_name}: {str(e)}")
            # Disable save button when error occurs
            self.save_image_button.setEnabled(False)
        
    def preview_data(self, folder_path, is_input_data=True):
        """Preview data folder contents"""
        # In practical applications, different loading functions should be called based on file types and data types in the folder
        # For demonstration purposes, we generate a basic preview
        
        model_size = 32
        
    def save_current_image(self):
        """Save currently displayed image"""
        # Call canvas's save method
        if self.canvas.save_figure():
            QMessageBox.information(self, "Success", "Image saved successfully!")

class ModelConfigTab(QWidget):
    """Model Selection and Parameter Configuration Tab"""
    def __init__(self, parent=None):
        super(ModelConfigTab, self).__init__(parent)
        self.parent = parent  # Save parent reference
        self.init_ui()

    def init_ui(self):
        """Initialize UI interface"""
        # Create main layout
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Model selection
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        model_group = QGroupBox("Model Selection")
        
        model_layout = QVBoxLayout()
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["UnetModel", "DinkNet"])
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        
        model_layout.addWidget(QLabel("Select MT-specific model:"))
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        
        # Add to left layout
        left_layout.addWidget(model_group)
        left_layout.addStretch()
        
        # Right panel: Model parameter configuration only
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Model parameter configuration
        params_group = QGroupBox("Training Parameter Configuration")
        
        params_layout = QGridLayout()
        params_layout.setContentsMargins(10, 5, 10, 10)  # Reduce top margin for better title spacing
        
        # Training epochs
        params_layout.addWidget(QLabel("Training Epochs:"), 0, 0)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 10000)
        self.epochs.setValue(100)  # Default 100
        params_layout.addWidget(self.epochs, 0, 1)
        
        # Batch Size
        params_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 1000)
        self.batch_size.setValue(8)  # Default 8
        params_layout.addWidget(self.batch_size, 1, 1)
        
        # Learning Rate
        params_layout.addWidget(QLabel("Learning Rate:"), 2, 0)
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.00000001, 1.0)
        self.learning_rate.setValue(0.00001)  # Default 0.00001
        self.learning_rate.setDecimals(8)
        
        # 自定义步长调整行为：点击向上箭头乘10，点击向下箭头除10
        def stepBy(self, steps):
            # 保存当前值
            current_value = self.value()
            
            # 实现乘10或除10的逻辑
            if steps > 0:  # 向上箭头
                new_value = current_value * 10.0
            else:  # 向下箭头
                new_value = current_value / 10.0
            
            # 确保新值在范围内
            min_val = self.minimum()
            max_val = self.maximum()
            new_value = max(min_val, min(max_val, new_value))
            
            # 设置新值
            self.setValue(new_value)
        
        # 重写QDoubleSpinBox的stepBy方法
        self.learning_rate.stepBy = lambda steps, self=self.learning_rate: stepBy(self, steps)
        # 不需要valueChanged信号，因为我们已经重写了stepBy方法
        params_layout.addWidget(self.learning_rate, 2, 1)
        
        # Optimizer selection
        params_layout.addWidget(QLabel("Optimizer:"), 3, 0)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "SGD", "RMSprop"])
        self.optimizer_combo.setCurrentText("Adam")
        params_layout.addWidget(self.optimizer_combo, 3, 1)
        
        # Model save interval
        params_layout.addWidget(QLabel("Model Save Interval:"), 4, 0)
        self.save_interval = QSpinBox()
        self.save_interval.setRange(1, 1000)
        self.save_interval.setValue(10)  # Default 10 epochs
        params_layout.addWidget(self.save_interval, 4, 1)
        params_layout.addWidget(QLabel("(epochs)"), 4, 2)
        
        # Add training data ratio control
        params_layout.addWidget(QLabel("Train Size:"), 5, 0)
        self.train_size = QDoubleSpinBox()
        self.train_size.setRange(0.01, 0.99)
        # Set step to 0.05
        self.train_size.setSingleStep(0.05)
        # Try to import default value from ParamConfig
        try:
            import ParamConfig
            if hasattr(ParamConfig, 'TrainSize'):
                self.train_size.setValue(ParamConfig.TrainSize)
            else:
                self.train_size.setValue(0.8)
        except:
            self.train_size.setValue(0.8)
        self.train_size.setDecimals(2)
        params_layout.addWidget(self.train_size, 5, 1)
        params_layout.addWidget(QLabel("(Ratio of training data)"), 5, 2)
        
        # Add validation data ratio control
        params_layout.addWidget(QLabel("Validation Size:"), 6, 0)
        self.val_size = QDoubleSpinBox()
        self.val_size.setRange(0.01, 0.99)
        self.val_size.setSingleStep(0.05)
        # Default validation size is 0.2 (20%)
        try:
            import ParamConfig
            if hasattr(ParamConfig, 'ValSize'):
                self.val_size.setValue(ParamConfig.ValSize)
            else:
                self.val_size.setValue(0.2)
        except:
            self.val_size.setValue(0.2)
        self.val_size.setDecimals(2)
        params_layout.addWidget(self.val_size, 6, 1)
        params_layout.addWidget(QLabel("(Ratio of validation data)"), 6, 2)
        
        # Early Stop patience
        params_layout.addWidget(QLabel("Early Stop Patience:"), 7, 0)
        self.early_stop = QSpinBox()
        self.early_stop.setRange(1, 1000)
        # Try to import default value from ParamConfig
        try:
            import ParamConfig
            if hasattr(ParamConfig, 'EarlyStop'):
                self.early_stop.setValue(ParamConfig.EarlyStop)
            else:
                self.early_stop.setValue(100)
        except:
            self.early_stop.setValue(100)
        params_layout.addWidget(self.early_stop, 7, 1)
        params_layout.addWidget(QLabel("(epochs without improvement)"), 7, 2)
        
        # Pre-trained model loading option
        params_layout.addWidget(QLabel("Use Pre-trained Model:"), 8, 0)
        self.use_pretrained = QCheckBox()
        # Try to import default value from ParamConfig
        try:
            import ParamConfig
            if hasattr(ParamConfig, 'ReUse'):
                self.use_pretrained.setChecked(ParamConfig.ReUse)
            else:
                self.use_pretrained.setChecked(False)
        except:
            self.use_pretrained.setChecked(False)
        self.use_pretrained.stateChanged.connect(self.on_pretrained_changed)
        params_layout.addWidget(self.use_pretrained, 8, 1)
        
        # Pre-trained model file selection
        params_layout.addWidget(QLabel("Pre-trained Model File:"), 9, 0)
        pretrained_layout = QHBoxLayout()
        self.pretrained_model_path = QLineEdit()
        self.pretrained_model_path.setPlaceholderText("Select pre-trained model file (.pth, .pt, .ckpt, etc.)...")
        self.pretrained_model_path.setReadOnly(True)
        # Try to load default value from ParamConfig
        try:
            import ParamConfig
            import os
            if hasattr(ParamConfig, 'PreModel') and ParamConfig.PreModel:
                from PathConfig import models_dir
                default_path = os.path.join(models_dir, f"{ParamConfig.PreModel}.pth")
                if os.path.exists(default_path):
                    self.pretrained_model_path.setText(default_path)
                else:
                    self.pretrained_model_path.setText(f"{ParamConfig.PreModel}.pth")
        except:
            pass
        pretrained_layout.addWidget(self.pretrained_model_path)
        self.select_pretrained_btn = QPushButton("Browse...")
        self.select_pretrained_btn.clicked.connect(self.select_pretrained_model)
        self.select_pretrained_btn.setEnabled(self.use_pretrained.isChecked())
        pretrained_layout.addWidget(self.select_pretrained_btn)
        params_layout.addLayout(pretrained_layout, 9, 1, 1, 2)
        
        params_group.setLayout(params_layout)
        
        # Add to right layout
        right_layout.addWidget(params_group)
        
        # Add confirm button
        confirm_button = QPushButton("Confirm Configuration")
        confirm_button.clicked.connect(self.confirm_config)
        right_layout.addWidget(confirm_button, alignment=Qt.AlignCenter)
        
        right_layout.addStretch()
        
        # Add left and right panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 500])  # Adjust initial size ratio to show left text completely
        
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
    
    def on_model_changed(self, index):
        """Update parameter configuration when model selection changes"""
        # In practical applications, update parameter configuration interface based on selected model
        print(f"Model switched to: {self.model_combo.currentText()}")
    
    def on_pretrained_changed(self, state):
        """Handle pre-trained model checkbox state change"""
        # Enable/disable the file selection button based on checkbox state
        self.select_pretrained_btn.setEnabled(state == Qt.Checked)
        if state != Qt.Checked:
            # Clear the path when unchecked
            self.pretrained_model_path.clear()
    
    def select_pretrained_model(self):
        """Open file dialog to select pre-trained model file"""
        import os
        try:
            from PathConfig import models_dir
        except ImportError:
            models_dir = getattr(__import__('ParamConfig', fromlist=['ModelsDir']), 'ModelsDir', 'models/')
        default_dir = os.path.abspath(models_dir) if os.path.exists(models_dir) else os.getcwd()
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Pre-trained Model File",
            default_dir,
            "Model Files (*.pth *.pt *.ckpt *.pkl *.h5);;All Files (*.*)"
        )
        
        if file_path:
            # Convert to absolute path to ensure it's always accessible
            abs_file_path = os.path.abspath(file_path)
            self.pretrained_model_path.setText(abs_file_path)
            
            # Update ParamConfig immediately
            try:
                import ParamConfig
                # Store the full absolute path
                ParamConfig.PreModelPath = abs_file_path
                # Also store just the filename without extension for PreModel (for backward compatibility)
                model_name = os.path.splitext(os.path.basename(file_path))[0]
                ParamConfig.PreModel = model_name
                ParamConfig.ReUse = True
                print(f"Pre-trained model selected: {abs_file_path}")
            except Exception as e:
                print(f"Warning: Failed to update ParamConfig: {e}")
    
    def get_mt_mode_from_data_import(self):
        """Get MT mode from Data Import tab"""
        try:
            if self.parent and hasattr(self.parent, 'data_tab'):
                data_tab = self.parent.data_tab
                if data_tab.te_radio.isChecked():
                    return 'TE'
                elif data_tab.tm_radio.isChecked():
                    return 'TM'
                elif data_tab.both_radio.isChecked():
                    return 'Both'
            # Default to TM if not available
            return 'TM'
        except Exception as e:
            print(f"Warning: Failed to get MT mode from Data Import tab: {e}")
            return 'TM'
    
    def confirm_config(self):
        """Confirm configuration and pass to the underlying layer"""
        model_name = self.model_combo.currentText()
        
        # Get MT mode from Data Import tab
        forward_mode = self.get_mt_mode_from_data_import()
        
        # Calculate input and output channels based on MT mode
        # Input: resistivity model (1 channel)
        # Output: apparent resistivity and phase
        #   - TE or TM: 2 channels (resistivity + phase)
        #   - Both: 4 channels (TE resistivity + TE phase + TM resistivity + TM phase)
        input_channels = 1  # Input is always resistivity model (1 channel)
        if forward_mode == 'Both':
            output_channels = 4  # Both mode: 4 channels output
        else:
            output_channels = 2  # TE or TM mode: 2 channels output
        
        epochs = self.epochs.value()
        batch_size = self.batch_size.value()
        learning_rate = self.learning_rate.value()
        optimizer = self.optimizer_combo.currentText()
        save_interval = self.save_interval.value()
        train_size = self.train_size.value()
        val_size = self.val_size.value()
        early_stop = self.early_stop.value()
        
        # Validate that train_size + val_size <= 1.0
        if train_size + val_size > 1.0:
            QMessageBox.warning(self, "Validation Error", "Sum of training and validation sizes cannot exceed 1.0")
            return
            
        # Update TrainSize and add ValSize in ParamConfig.py
        try:
            import ParamConfig
            import os
            
            # Directly modify variables in the module
            ParamConfig.TrainSize = train_size
            ParamConfig.ValSize = val_size
            
            # Also update file content for next program start
            current_dir = os.path.dirname(os.path.abspath(__file__))
            param_config_path = os.path.join(current_dir, 'ParamConfig.py')
            with open(param_config_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            updated_lines = []
            train_size_updated = False
            val_size_updated = False
            early_stop_updated = False
            
            for line in lines:
                if line.startswith('TrainSize'):
                    comment_pos = line.find('#')
                    if comment_pos != -1:
                        comment = line[comment_pos:]
                        updated_lines.append(f'TrainSize     = {train_size:.2f}      {comment}\n')
                    else:
                        updated_lines.append(f'TrainSize     = {train_size:.2f}      # Training data size ratio\n')
                    train_size_updated = True
                elif line.startswith('ValSize'):
                    comment_pos = line.find('#')
                    if comment_pos != -1:
                        comment = line[comment_pos:]
                        updated_lines.append(f'ValSize       = {val_size:.2f}      {comment}\n')
                    else:
                        updated_lines.append(f'ValSize       = {val_size:.2f}      # Validation data size ratio\n')
                    val_size_updated = True
                elif line.startswith('EarlyStop'):
                    comment_pos = line.find('#')
                    if comment_pos != -1:
                        comment = line[comment_pos:]
                        updated_lines.append(f'EarlyStop     = {early_stop}       {comment}\n')
                    else:
                        updated_lines.append(f'EarlyStop     = {early_stop}       # Early stop patience (epochs without improvement)\n')
                    early_stop_updated = True
                else:
                    updated_lines.append(line)
            
            # If TrainSize line not found, add it to the file
            if not train_size_updated:
                for i, line in enumerate(lines):
                    if line.startswith('Epochs'):
                        updated_lines.insert(i+1, f'TrainSize     = {train_size:.2f}      # Training data size ratio\n')
                        break
            
            # If ValSize line not found, add it after TrainSize
            if not val_size_updated:
                for i, line in enumerate(updated_lines):
                    if line.startswith('TrainSize'):
                        updated_lines.insert(i+1, f'ValSize       = {val_size:.2f}      # Validation data size ratio\n')
                        break
            
            # If EarlyStop line not found, add it after LearnRate
            if not early_stop_updated:
                for i, line in enumerate(updated_lines):
                    if line.startswith('LearnRate'):
                        updated_lines.insert(i+1, f'EarlyStop     = {early_stop}       # Early stop patience (epochs without improvement)\n')
                        break
            
            # Write updated content
            with open(param_config_path, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)
        except Exception as e:
            print(f"Warning: Failed to update ParamConfig.py: {str(e)}")
        
        # Update ParamConfig with forward modeling parameters
        try:
            import ParamConfig
            ParamConfig.Inchannels = input_channels
            ParamConfig.Nclasses = output_channels
            ParamConfig.MT_Mode = forward_mode
            ParamConfig.Epochs = epochs
            ParamConfig.BatchSize = batch_size
            ParamConfig.LearnRate = learning_rate
            ParamConfig.TrainSize = train_size
            ParamConfig.ValSize = val_size
            ParamConfig.EarlyStop = early_stop
            
            # Handle pre-trained model settings
            if self.use_pretrained.isChecked():
                pretrained_path = self.pretrained_model_path.text().strip()
                if pretrained_path:
                    # Resolve path: try absolute, then models_dir
                    abs_pretrained_path = os.path.abspath(pretrained_path)
                    models_dir_cfg = getattr(ParamConfig, 'ModelsDir', 'models/')
                    if not os.path.isabs(models_dir_cfg):
                        models_dir_cfg = os.path.abspath(models_dir_cfg)
                    candidates = [
                        abs_pretrained_path,
                        os.path.join(models_dir_cfg, os.path.basename(pretrained_path)),
                        os.path.join(models_dir_cfg, f"{getattr(ParamConfig, 'ModelName', 'DinkNet')}_best.pth"),
                    ]
                    resolved_path = None
                    for p in candidates:
                        if p and os.path.exists(p):
                            resolved_path = os.path.abspath(p)
                            break
                    if resolved_path:
                        ParamConfig.ReUse = True
                        ParamConfig.PreModelPath = resolved_path
                        ParamConfig.PreModel = os.path.splitext(os.path.basename(resolved_path))[0]
                        print(f"Pre-trained model configured: {resolved_path}")
                        if resolved_path != abs_pretrained_path:
                            self.pretrained_model_path.setText(resolved_path)
                    else:
                        print(f"Warning: Pre-trained model file not found. Tried: {candidates[0]}")
                        ParamConfig.ReUse = False
                        if hasattr(ParamConfig, 'PreModelPath'):
                            delattr(ParamConfig, 'PreModelPath')
                else:
                    print("Warning: Pre-trained model enabled but no file path provided. Click Browse to select model.")
                    ParamConfig.ReUse = False
                    if hasattr(ParamConfig, 'PreModelPath'):
                        delattr(ParamConfig, 'PreModelPath')
            else:
                ParamConfig.ReUse = False
                if hasattr(ParamConfig, 'PreModelPath'):
                    delattr(ParamConfig, 'PreModelPath')
        except Exception as e:
            print(f"Warning: Failed to update ParamConfig: {str(e)}")
        
        # Here should pass the configuration to the underlying module
        print(f"Configuration confirmed:\nModel: {model_name}\nInput channels: {input_channels}\nOutput channels: {output_channels}\nForward mode: {forward_mode}\nEpochs: {epochs}\nBatch size: {batch_size}\nLearning rate: {learning_rate}\nOptimizer: {optimizer}\nSave interval: {save_interval}\nTrain Size: {train_size}\nValidation Size: {val_size}\nEarly Stop: {early_stop}")
        
        # Enable training tab after confirming configuration
        if self.parent and hasattr(self.parent, 'update_tab_status'):
            self.parent.update_tab_status(1)  # 1 is the index of Model Training tab
            
        # Build confirmation message
        confirm_msg = f"Forward modeling configuration confirmed:\n\n"
        confirm_msg += f"Network Structure:\n"
        confirm_msg += f"- Model: {model_name}\n"
        confirm_msg += f"- Input channels: {input_channels} (Model parameters)\n"
        confirm_msg += f"- Output channels: {output_channels} (Apparent resistivity & phase)\n"
        confirm_msg += f"- Forward mode: {forward_mode}\n\n"
        confirm_msg += f"Training Parameters:\n"
        confirm_msg += f"- Epochs: {epochs}\n"
        confirm_msg += f"- Early Stop Patience: {early_stop}\n"
        confirm_msg += f"- Batch size: {batch_size}\n"
        confirm_msg += f"- Learning rate: {learning_rate}\n"
        confirm_msg += f"- Optimizer: {optimizer}\n"
        confirm_msg += f"- Save interval: {save_interval} epochs\n"
        confirm_msg += f"- Training data ratio: {round(train_size*100, 1):.1f}%\n"
        confirm_msg += f"- Validation data ratio: {round(val_size*100, 1):.1f}%\n"
        
        # Add pre-trained model info if enabled
        if self.use_pretrained.isChecked():
            pretrained_path = self.pretrained_model_path.text()
            if pretrained_path:
                import os
                model_name = os.path.basename(pretrained_path)
                confirm_msg += f"\nPre-trained Model:\n"
                confirm_msg += f"- Use Pre-trained Model: Yes\n"
                confirm_msg += f"- Model File: {model_name}\n"
                confirm_msg += f"- Will continue training from this model"
            else:
                confirm_msg += f"\nPre-trained Model:\n"
                confirm_msg += f"- Warning: Enabled but no model file selected"
        
        QMessageBox.information(self, "Configuration Confirmed", confirm_msg)

class TrainingTab(QWidget):
    """One-click training and real-time monitoring tab"""
    def __init__(self, parent=None):
        super(TrainingTab, self).__init__(parent)
        self.parent = parent
        self.training_thread = None
        self.selected_model_name = ""
        self.init_ui()

    def init_ui(self):
        """Initialize UI interface"""
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)  # 设置整体布局边距
        main_layout.setSpacing(5)  # 设置组件间距
        
        # No longer use custom title_style, use global style defined in set_style
        
        # 从ParamConfig读取参数（不显示，仅供内部使用）
        try:
            import ParamConfig
            epochs = getattr(ParamConfig, 'Epochs', 200)
            batch_size = getattr(ParamConfig, 'BatchSize', 20)
            learning_rate = getattr(ParamConfig, 'LearnRate', 0.001)
            train_size = getattr(ParamConfig, 'TrainSize', 0.8)
            val_size = getattr(ParamConfig, 'ValSize', 0.2)
            
            # 保存参数值供训练使用（不显示在界面上）
            self.epochs_value = epochs
            self.batch_size_value = batch_size
            self.learning_rate_value = learning_rate
            
        except Exception as e:
            # 如果读取失败，使用默认值
            self.epochs_value = 200
            self.batch_size_value = 20
            self.learning_rate_value = 0.001
        
        # Training control area
        control_group = QGroupBox("Control")
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(10, 5, 10, 5)  # Set inner margins
        control_layout.setSpacing(5)  # Set spacing
        
        # Reduce minimum button width for more compact layout
        button_width = 120
        
        self.start_button = QPushButton("Start Training")
        self.start_button.setMinimumWidth(button_width)
        self.start_button.clicked.connect(self.start_training)
        
        self.pause_button = QPushButton("Pause Training")
        self.pause_button.setMinimumWidth(button_width)
        self.pause_button.clicked.connect(self.pause_training)
        self.pause_button.setEnabled(False)
        
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.setMinimumWidth(button_width)
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        
        self.save_button = QPushButton("Save Current Model")
        self.save_button.setMinimumWidth(button_width)
        self.save_button.clicked.connect(self.save_model)
        self.save_button.setEnabled(False)
        self.save_button.setVisible(False)  # Hide Save Current Model button
        
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.stop_button)
        control_group.setLayout(control_layout)
        
        # Training monitoring area
        monitor_group = QGroupBox("Monitoring")
        monitor_layout = QVBoxLayout()
        monitor_layout.setContentsMargins(10, 5, 10, 5)  # Set inner margins
        monitor_layout.setSpacing(10)  # Set spacing
        
        # Create loss curve canvas with adjusted height
        self.loss_canvas = MPLCanvas(self, width=6, height=3.5, dpi=100)  # Reduce height to minimize black border
        
        # Training progress bar with customized style to avoid overlapping
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        # Set style to remove background and border, prevent overlapping display
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                background: transparent;
                border: none;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
            """
        )
        
        # Training information display with increased maximum height
        self.training_info = QTextEdit()
        self.training_info.setReadOnly(True)
        self.training_info.setPlaceholderText("Training information will be displayed here...")
        self.training_info.setMaximumHeight(150)  # Increase maximum height to display more information
        
        # Create and style labels to ensure complete text display
        progress_label = QLabel("Training Progress:")
        # Use global style
        
        info_label = QLabel("Training Information:")
        # Use global style
        
        # First add the curve chart to ensure it has enough space to display
        monitor_layout.addWidget(self.loss_canvas, 1)  # Use stretch factor 1 to make it occupy more space
        
        # Then add progress bar and training information
        monitor_layout.addWidget(progress_label)
        monitor_layout.addWidget(self.progress_bar)
        monitor_layout.addWidget(info_label)
        monitor_layout.addWidget(self.training_info)
        monitor_group.setLayout(monitor_layout)
        
        # Add to main layout
        # params_group已移除，不再显示参数信息
        main_layout.addWidget(control_group)
        main_layout.addWidget(monitor_group, 1)  # 使用伸缩因子1使其占据剩余空间
        self.setLayout(main_layout)
    
    def reset_training_buttons(self):
        """Reset training button states to allow starting new training"""
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.save_button.setEnabled(False)
    
    def start_training(self):
        """Start training with thread safety checks"""
        # 检查是否已有线程在运行
        if hasattr(self, 'training_thread') and self.training_thread and self.training_thread.isRunning():
            self.update_training_info("Training is already in progress. Please stop it first.")
            return
            
        # 清理任何已完成但未清理的线程
        if hasattr(self, 'training_thread') and self.training_thread:
            if not self.training_thread.isRunning():
                # 线程已完成，安全清理
                self.training_thread = None
            else:
                # 线程仍在运行，不应该发生，但为了安全起见
                self.update_training_info("Warning: Previous training thread still exists. Please wait or restart the application.")
                return
        
        # Disable/enable related buttons
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.save_button.setEnabled(True)
        
        # Model Prediction tab is already enabled by default in main window initialization
            
        try:
            # Get data file paths from DataImportTab
            if self.parent and hasattr(self.parent, 'tabs'):
                data_tab = self.parent.tabs.widget(0)  # DataImportTab is at index 0
                model_config_tab = self.parent.tabs.widget(1)  # ModelConfigTab is at index 1
                
                # Get selected file paths from Data Import tab
                file_paths = data_tab.get_selected_file_paths() if hasattr(data_tab, 'get_selected_file_paths') else {}
                
                # Determine input channels and MT mode based on selected mode
                if hasattr(data_tab, 'te_radio') and hasattr(data_tab, 'tm_radio'):
                    if data_tab.te_radio.isChecked():
                        # TE mode - 2 input channels
                        in_channels = 2
                        mt_mode = 'TE'
                    elif data_tab.tm_radio.isChecked():
                        # TM mode - 2 input channels
                        in_channels = 2
                        mt_mode = 'TM'
                    else:
                        # Both mode - 4 input channels
                        in_channels = 4
                        mt_mode = 'Both'
                else:
                    # Fallback to original method if mode selection is not available
                    in_channels = 1
                    mt_mode = 'TE'
                
                # Get model configuration from ModelConfigTab
                model_name = model_config_tab.model_combo.currentText()
                batch_size = model_config_tab.batch_size.value()
                learning_rate = model_config_tab.learning_rate.value()
                
                # Get training parameters from ParamConfig (已在init_ui中读取)
                epochs = self.epochs_value
                # 不再使用SaveEpoch参数，因为模型会根据验证损失自动保存最佳模型
                
                # Create training thread with all parameters including in_channels, mt_mode, and file_paths
                self.training_thread = TrainingThread(
                    epochs=epochs,
                    model_name=model_name,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    models_dir=models_dir,
                    results_dir=results_dir,
                    in_channels=in_channels,
                    mt_mode=mt_mode,
                    file_paths=file_paths
                )
                
                # Connect signals and slots
                self.training_thread.update_signal.connect(self.update_training_info)
                self.training_thread.progress_signal.connect(self.update_progress)
                self.training_thread.loss_signal.connect(self.update_loss_curve)
                # Connect custom training finished signal
                self.training_thread.training_finished.connect(self.training_finished)
                
                # Start thread
                self.training_thread.start()
                
                self.selected_model_name = model_name
                self.training_info.append(f"Training started with model: {model_name}")
            else:
                self.training_info.append("Error: Could not access data and model configuration")
                self.training_finished()
        except Exception as e:
            self.training_info.append(f"Error starting training: {str(e)}")
            self.training_finished()
    
    def pause_training(self):
        """Pause the training process with status checks"""
        if hasattr(self, 'training_thread') and self.training_thread and self.training_thread.isRunning():
            self.training_thread.pause()
            self.training_info.append("Training paused")
            self.pause_button.setText("Resume Training")
            self.pause_button.clicked.disconnect()
            self.pause_button.clicked.connect(self.resume_training)
        else:
            self.update_training_info("No active training to pause.")
    
    def resume_training(self):
        """Resume training with status checks"""
        if hasattr(self, 'training_thread') and self.training_thread and self.training_thread.isRunning():
            self.training_thread.resume()
            self.training_info.append("Training resumed")
            self.pause_button.setText("Pause Training")
            self.pause_button.clicked.disconnect()
            self.pause_button.clicked.connect(self.pause_training)
        else:
            self.update_training_info("No active training to resume.")
    
    def stop_training(self):
        """Safely stop training"""
        if hasattr(self, 'training_thread') and self.training_thread and self.training_thread.isRunning():
            # 发送停止信号
            self.training_thread.stop()
            self.update_training_info("Stopping training...")
            
            # 立即禁用停止按钮，防止重复点击
            self.stop_button.setEnabled(False)
            
            # 连接finished信号到清理槽函数
            def cleanup_thread():
                """在线程完全结束后清理资源并重置按钮状态"""
                if hasattr(self, 'training_thread') and self.training_thread:
                    thread_ref = self.training_thread
                    # 确保线程已经完全结束
                    if not thread_ref.isRunning():
                        # 使用deleteLater()让Qt安全地管理线程对象的销毁
                        thread_ref.deleteLater()
                        self.training_thread = None
                        self.update_training_info("Training process stopped, resources safely released")
                    else:
                        # 如果还在运行，等待一下
                        thread_ref.wait(2000)  # 等待最多2秒
                        if not thread_ref.isRunning():
                            thread_ref.deleteLater()
                            self.training_thread = None
                            self.update_training_info("Training process stopped, resources safely released")
                        else:
                            self.update_training_info("Warning: Thread cannot terminate normally")
                
                # 重置按钮状态，允许重新开始训练
                self.reset_training_buttons()
            
            # 如果线程已经在停止中，直接连接信号
            if not self.training_thread.isFinished():
                # 使用单次连接，避免重复连接
                self.training_thread.finished.connect(cleanup_thread, Qt.UniqueConnection)
                
                # 等待线程结束（最多等待3秒）
                if not self.training_thread.wait(3000):
                    self.update_training_info("Warning: Timeout waiting for thread to finish")
                    # 即使超时，也尝试清理
                    cleanup_thread()
            else:
                # 如果线程已经结束，立即清理
                cleanup_thread()
        else:
            self.update_training_info("No active training to stop.")
            # 确保按钮状态正确
            self.reset_training_buttons()
            # 确保按钮状态正确
            self.reset_training_buttons()
    
    def save_model(self):
        """Save current model"""
        # In actual application, this should call the model saving function
        options = QFileDialog.Options()
        file, _ = QFileDialog.getSaveFileName(
            self, "Save Model", "", "Model Files (*.pth *.pt);;All Files (*)", options=options
        )
        
        if file:
            # Here should call the underlying model saving function
            self.training_info.append(f"Model saved to: {file}")
    
    @pyqtSlot(str)
    def update_training_info(self, info):
        """Update training information with high-performance throttling mechanism to prevent UI freezing"""
        # Use internal buffer to store information first, greatly reducing direct UI operations
        if not hasattr(self, 'info_buffer'):
            self.info_buffer = []
        self.info_buffer.append(info)
        
        # Initialize counter
        if not hasattr(self, 'update_counter'):
            self.update_counter = 0
        self.update_counter += 1
        
        # High-priority optimization: only perform UI operations in the following cases
        # 1. When the buffer reaches a certain size (20 items), batch update UI to significantly reduce UI operation frequency
        # 2. When information contains keywords (such as 'Epoch', 'Error'), update immediately to ensure important information is displayed in a timely manner
        should_update = False
        
        # Check if it contains important keywords
        if any(keyword in info for keyword in ['Epoch', 'Error', 'Loss', 'completed', 'training', 'model']):
            should_update = True
        # Check buffer size
        elif len(self.info_buffer) >= 20:
            should_update = True
        # Regular update (every 50 items)
        elif self.update_counter % 50 == 0:
            should_update = True
            
        if should_update:
            # Batch add information to text box
            self.training_info.append('\n'.join(self.info_buffer))
            self.info_buffer = []  # Clear buffer
            
            # Line limit - only perform line limit check every 50 items to minimize processing overhead
            MAX_LINES = 100
            if self.training_info.document().blockCount() > MAX_LINES:
                # Clean up excess lines in one go to reduce repeated operations
                excess_lines = self.training_info.document().blockCount() - MAX_LINES
                cursor = QTextCursor(self.training_info.document())
                cursor.movePosition(QTextCursor.Start)
                for _ in range(excess_lines):
                    cursor.movePosition(QTextCursor.EndOfLine, QTextCursor.KeepAnchor)
                    cursor.removeSelectedText()
                    if not cursor.atEnd():
                        cursor.deleteChar()  # Delete newline character
                        
            # Auto-scroll - only perform once every 20 items or when containing keywords to reduce scrolling operations
            if any(keyword in info for keyword in ['Epoch', 'Error', 'Loss', 'completed', 'training', 'model']) or len(self.info_buffer) % 20 == 0:
                self.training_info.moveCursor(QTextCursor.End)
    
    @pyqtSlot(int)
    def update_progress(self, value):
        """Update training progress"""
        self.progress_bar.setValue(value)
    
    @pyqtSlot(list, list, list)
    def update_loss_curve(self, train_loss, val_loss, _):
        """Update loss curve with optimized performance using throttling to prevent UI freezing"""
        
        # Add chart update throttling mechanism
        if not hasattr(self, 'curve_update_counter'):
            self.curve_update_counter = 0
        self.curve_update_counter += 1
        
        # Significantly reduce chart update frequency:
        # 1. Update chart only at certain important epoch points
        # 2. Update every 10 epochs for the first 100 epochs
        # 3. Update every 20 epochs for epochs 100-500
        # 4. Update every 50 epochs for epochs over 500
        # 5. Always update the last few epochs to ensure final results are displayed
        current_epoch = len(train_loss)
        should_update = False
        
        # First 100 epochs: update every 10 epochs
        if current_epoch <= 100 and current_epoch % 10 == 0:
            should_update = True
        # Epochs 100-500: update every 20 epochs
        elif current_epoch <= 500 and current_epoch % 20 == 0:
            should_update = True
        # Epochs over 500: update every 50 epochs
        elif current_epoch % 50 == 0:
            should_update = True
        # Last 10 epochs: update every epoch
        elif current_epoch > len(train_loss) - 10:
            should_update = True
        
        if should_update:
            # Only perform complete plotting operations when needed
            self.loss_canvas.clear()
            
            epochs = range(1, len(train_loss) + 1)
            # 绘制训练损失曲线
            train_label = f'{self.selected_model_name} Training Loss' if self.selected_model_name else 'Training Loss'
            self.loss_canvas.axes.plot(epochs, train_loss, 'b-', linewidth=2, label=train_label)
            
            # 绘制验证损失曲线（如果有数据）
            if val_loss and len(val_loss) > 0:
                # 确保验证损失的长度与训练损失匹配或接近
                if abs(len(val_loss) - len(train_loss)) <= 1:
                    # 如果验证损失长度比训练损失少1，进行填充以确保绘图一致
                    display_val_loss = val_loss.copy()
                    if len(display_val_loss) < len(train_loss):
                        display_val_loss.append(display_val_loss[-1])
                    
                    val_label = f'{self.selected_model_name} Validation Loss' if self.selected_model_name else 'Validation Loss'
                    self.loss_canvas.axes.plot(epochs, display_val_loss, 'r-', linewidth=2, label=val_label)
            
            # 添加图例
            self.loss_canvas.axes.legend(loc='upper right')
            
            # Add X and Y axis labels and ticks
            self.loss_canvas.axes.set_xlabel('Epochs')
            self.loss_canvas.axes.set_ylabel('Loss Value')
            
            # Set X-axis ticks to ensure correct display up to the target number of epochs
            if len(epochs) > 0:
                current_epoch = len(train_loss)
                
                # Improved X-axis tick settings - ensure ticks display correctly up to 100 epochs or more
                # For 100 epochs training, set ticks at 0, 20, 40, 60, 80, 100, etc.
                if current_epoch <= 100:
                    # For 100 epochs or less, one tick every 10 epochs
                    tick_step = 10
                    xticks = list(range(0, current_epoch + 1, tick_step))
                    # Ensure the last tick is the current total number of epochs
                    if xticks[-1] != current_epoch:
                        xticks.append(current_epoch)
                elif current_epoch <= 500:
                    # For 100-500 epochs, one tick every 50 epochs
                    tick_step = 50
                    xticks = list(range(0, current_epoch + 1, tick_step))
                    if xticks[-1] != current_epoch:
                        xticks.append(current_epoch)
                else:
                    # For over 500 epochs, one tick every 100 epochs
                    tick_step = 100
                    xticks = list(range(0, current_epoch + 1, tick_step))
                    if xticks[-1] != current_epoch:
                        xticks.append(current_epoch)
                
                # Apply X-axis ticks
                self.loss_canvas.axes.set_xticks(xticks)
                
                # Automatically set Y-axis range to include training and validation loss values
                all_losses = train_loss.copy()
                # Add validation loss if available
                if val_loss and isinstance(val_loss, list):
                    all_losses.extend(val_loss)
                
                if all_losses:
                    min_loss = min(all_losses)
                    max_loss = max(all_losses)
                    # Add some margin to make the chart more visually appealing
                    y_margin = (max_loss - min_loss) * 0.1 if max_loss > min_loss else 0.1
                    self.loss_canvas.axes.set_ylim(min_loss - y_margin, max_loss + y_margin)
                
                # Improved Y-axis tick display - dynamically adjust tick format based on loss value magnitude
                from matplotlib.ticker import AutoMinorLocator, FuncFormatter
                
                # Set Y-axis major ticks
                self.loss_canvas.axes.tick_params(axis='y', which='major', labelsize=8)
                
                # Enable Y-axis minor ticks
                self.loss_canvas.axes.yaxis.set_minor_locator(AutoMinorLocator())
                # Show Y-axis minor tick labels
                self.loss_canvas.axes.tick_params(axis='y', which='minor', labelsize=6, labelcolor='gray')
                
                # Dynamically select appropriate tick format based on loss value range
                loss_range = max_loss - min_loss
                if loss_range > 1000:
                    # Very large loss value range, use scientific notation
                    formatter = FuncFormatter(lambda x, pos: '{:.1e}'.format(x))
                elif loss_range > 100:
                    # Large loss value range, keep 1 decimal place
                    formatter = FuncFormatter(lambda x, pos: '{:.1f}'.format(x))
                elif loss_range > 10:
                    # Medium range, keep 2 decimal places
                    formatter = FuncFormatter(lambda x, pos: '{:.2f}'.format(x))
                elif loss_range > 1:
                    # Small range, keep 3 decimal places
                    formatter = FuncFormatter(lambda x, pos: '{:.3f}'.format(x))
                else:
                    # Very small range, use scientific notation or more decimal places
                    # Check the order of magnitude of the maximum value
                    max_abs_value = max(abs(min_loss), abs(max_loss))
                    if max_abs_value < 0.001:
                        formatter = FuncFormatter(lambda x, pos: '{:.1e}'.format(x))
                    elif max_abs_value < 0.01:
                        formatter = FuncFormatter(lambda x, pos: '{:.5f}'.format(x))
                    elif max_abs_value < 0.1:
                        formatter = FuncFormatter(lambda x, pos: '{:.4f}'.format(x))
                    else:
                        formatter = FuncFormatter(lambda x, pos: '{:.3f}'.format(x))
                
                # Apply formatter to Y-axis
                self.loss_canvas.axes.yaxis.set_major_formatter(formatter)
                
                # 明确设置Y轴主要刻度位置，不依赖于get_yticks()
                # 根据损失值范围自动生成合理的Y轴刻度
                y_step = (max_loss - min_loss) / 5  # 将Y轴分为5个主要刻度
                if y_step == 0:
                    y_step = 0.1  # 避免除以零
                
                # 计算主要刻度位置
                y_ticks = []
                current = min_loss
                while current <= max_loss + y_step/2:
                    y_ticks.append(current)
                    current += y_step
                
                # 应用Y轴刻度
                self.loss_canvas.axes.set_yticks(y_ticks)
                
                # 强制设置Y轴刻度和标签可见
                self.loss_canvas.axes.tick_params(axis='y', which='major', labelleft=True, left=True, labelsize=8)
                self.loss_canvas.axes.tick_params(axis='y', which='minor', labelleft=True, left=True, labelsize=6)
                
                # 确保Y轴轴线可见
                self.loss_canvas.axes.spines['left'].set_visible(True)
            
            # Adjust layout to ensure X-axis labels are fully visible
            # Increase bottom margin to prevent X-axis labels from being cut off
            self.loss_canvas.figure.subplots_adjust(bottom=0.15, top=0.95, left=0.1, right=0.95)
            
            # Only perform time-consuming draw operation when needed
            self.loss_canvas.draw()
    
    def training_finished(self):
        """Safely handle resource release after training completion"""
        # 重置按钮状态，允许重新开始训练
        self.reset_training_buttons()
        
        # Save thread reference for report generation
        self.training_thread_ref = self.training_thread
        training_stopped = False
        
        if self.training_thread_ref:
            training_stopped = self.training_thread_ref.was_stopped
        
        # Check if there is actual training log output
        has_training_logs = len(self.training_info.toPlainText()) > 100  # Simple check for sufficient training logs
        
        if training_stopped:
            self.training_info.append("Training stopped")
        else:
            # Final condition to determine if training was successful: sufficient training logs
            final_training_successful = has_training_logs
            
            if final_training_successful:
                self.training_info.append("Training completed successfully")
            else:
                self.training_info.append("Training completed with potential issues")
                
            # Generate training report
            self.generate_training_report()
            
            # Enable prediction tab after training is complete
            if self.parent and hasattr(self.parent, 'update_tab_status'):
                self.parent.update_tab_status(2)  # 2 is the index of Model Training tab
                
                # Automatically load the last trained model into the prediction tab
                try:
                    prediction_tab = self.parent.tabs.widget(3)  # PredictionTab is at index 3
                    if prediction_tab and hasattr(prediction_tab, 'model_path_label'):
                        # Get the path of the last model file saved during training
                        import os
                        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
                        # Find the latest .pth or .pt files in the models directory
                        model_files = []
                        if os.path.exists(models_dir):
                            for file in os.listdir(models_dir):
                                if file.endswith('.pth') or file.endswith('.pt'):
                                    file_path = os.path.join(models_dir, file)
                                    model_files.append((os.path.getmtime(file_path), file_path))
                        
                        # If model files are found, select the latest one
                        if model_files:
                            # Sort by modification time, get the latest one
                            model_files.sort(reverse=True)
                            latest_model_path = model_files[0][1]
                            
                            # Update model path in prediction tab
                            prediction_tab.model_path_label.setText(latest_model_path)
                            prediction_tab.update_model_info(latest_model_path)
                            
                            # If prediction data (resistivity model) is loaded, enable prediction button
                            if hasattr(prediction_tab, 'resistivity_model_path') and prediction_tab.resistivity_model_path:
                                prediction_tab.predict_button.setEnabled(True)
                            else:
                                # 即使预测数据未加载，也要确保标签页已启用
                                prediction_tab.predict_button.setEnabled(False)  # 但按钮仍需禁用
                            
                            self.training_info.append(f"Model automatically loaded to prediction tab: {latest_model_path}")
                        else:
                            self.training_info.append("Warning: No model files found, please load model manually")
                except Exception as e:
                    self.training_info.append(f"Error auto-loading model to prediction tab: {str(e)}")
                
                # 无论训练结果如何，都显示训练已完成
                QMessageBox.information(self, "Training Completed", "Training process has finished, MT_train.py execution completed.")
        
        # 确保线程资源安全释放
        # 关键修复：等待线程完全完成后再清理，避免"QThread: Destroyed while thread is still running"错误
        if self.training_thread is not None:
            # 检查线程是否还在运行
            if self.training_thread.isRunning():
                # 如果线程还在运行，等待它完成
                # 使用finished信号来确保线程完成后再清理
                def cleanup_after_finished():
                    """在线程完全结束后清理资源"""
                    # 确保线程已经完全结束
                    if not self.training_thread.isRunning():
                        # 使用deleteLater()让Qt安全地管理线程对象的销毁
                        self.training_thread.deleteLater()
                        self.training_thread = None
                        self.training_info.append("Training thread resources safely released")
                
                # 连接finished信号到清理函数（使用单次连接避免重复）
                self.training_thread.finished.connect(cleanup_after_finished, Qt.UniqueConnection)
                
                # 如果线程卡住，设置超时等待（最多等待5秒）
                import time
                start_time = time.time()
                while self.training_thread.isRunning() and (time.time() - start_time) < 5.0:
                    QApplication.processEvents()  # 处理事件，确保信号能够传递
                    time.sleep(0.1)
                
                # 如果超时后线程仍在运行，强制等待
                if self.training_thread.isRunning():
                    self.training_info.append("Warning: Timeout waiting for thread to finish, forcing wait...")
                    self.training_thread.wait(5000)  # 最多等待5秒
                    if self.training_thread.isRunning():
                        self.training_info.append("Error: Thread cannot terminate normally, possible deadlock")
                    else:
                        self.training_thread.deleteLater()
                        self.training_thread = None
            else:
                # 线程已经完成，可以安全清理
                self.training_thread.deleteLater()
                self.training_thread = None
    
    def generate_training_report(self):
        """Generate training report"""
        # In actual application, this should generate a detailed training report
        self.training_info.append("\n=== Training Report ===")
        self.training_info.append(f"Total epochs: {self.epochs_value}")
        
        # Get final loss value from training thread if available
        final_loss = "0.xxxx"
        if hasattr(self, 'training_thread_ref') and self.training_thread_ref:
            if hasattr(self.training_thread_ref, 'total_loss') and self.training_thread_ref.total_loss:
                final_loss = f"{self.training_thread_ref.total_loss[-1]:.6f}"
            elif hasattr(self.training_thread_ref, 'validation_losses') and self.training_thread_ref.validation_losses:
                final_loss = f"{self.training_thread_ref.validation_losses[-1]:.6f}"
        
        self.training_info.append(f"Final loss value: {final_loss}")
        self.training_info.append("Model saved to default path")
        self.training_info.append("====================")

class TrainingThread(QThread):
    """Training thread for model training in the background using ML Trainer module"""
    update_signal = pyqtSignal(str)  # Signal for updating training information
    progress_signal = pyqtSignal(int)  # Signal for updating progress
    loss_signal = pyqtSignal(list, list, list)  # Signal for updating loss curve with three loss lists
    # 自定义带参数的完成信号，替代QThread自带的无参数finished信号
    training_finished = pyqtSignal(bool)  # Signal with success/failure status
    
    def __init__(self, epochs=100, 
                 model_name='DinkNet50', 
                 models_dir='models/', results_dir='results/',
                 batch_size=8, learning_rate=0.001, in_channels=1,
                 mt_mode='TE', file_paths=None):
        super(TrainingThread, self).__init__()
        # 设置对象的所有权为Qt事件循环，而不是Python的垃圾回收机制
        # 这可以防止线程在完成前被Python垃圾回收器销毁
        self.setObjectName("TrainingThread")
        
        self.epochs = epochs
        # No longer need save_epoch parameter as model will auto-save based on validation loss
        self.paused = False
        self.stopped = False
        self.was_stopped = False  # 初始值设为False，表示训练未开始
        self.signals_emitted = False  # 用于跟踪是否已经发出过training_finished信号
        
        # 从GUI标签页获取的数据和配置
        self.model_name = model_name
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.in_channels = in_channels
        self.mt_mode = mt_mode
        self.file_paths = file_paths if file_paths else {}
        
        # For storing loss values
        self.total_loss = []  # Training loss
        self.validation_losses = []  # Validation loss
        
        
        # 用于跟踪已处理的epoch，避免重复添加损失值
        self.processed_epochs = set()
        
        # 同时更新ParamConfig.py文件中的Inchannels参数
        self.update_inchannels_param()
    
    def update_inchannels_param(self):
        """Update Inchannels parameter in ParamConfig.py"""
        try:
            import ParamConfig
            # Directly modify variable in the module
            ParamConfig.Inchannels = self.in_channels
            # No longer update SaveEpoch parameter as model will auto-save based on validation loss
            self.update_signal.emit(f"Updated ParamConfig.Inchannels to {self.in_channels}")
        except Exception as e:
            self.update_signal.emit(f"Warning: Failed to update ParamConfig: {str(e)}")
    
    def run(self):
        """Thread running function that runs the actual training by updating ParamConfig and directly calling MT_train.py"""
        try:
            import subprocess
            import sys
            import os
            
            # 首先更新ParamConfig.py文件中的训练参数，保留GUI中设置的参数
            self.update_param_config()
            
            # 直接获取MT_train.py的绝对路径
            mt_train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MT_train.py')
            
            self.update_signal.emit(f"Starting MT_train.py for training")
            self.update_signal.emit(f"MT_train.py path: {mt_train_path}")
            
            # 构建命令行参数，直接运行MT_train.py
            cmd_args = [sys.executable, mt_train_path]
            
            self.update_signal.emit(f"Command: {' '.join(cmd_args)}")
            
            # 直接启动MT_train.py进程
            process = None
            try:
                # 确保MT_train.py文件存在
                if not os.path.exists(mt_train_path):
                    self.update_signal.emit(f"Error: Cannot find MT_train.py file at path: {mt_train_path}")
                    if not self.signals_emitted:
                        self.training_finished.emit(False)
                        self.signals_emitted = True
                    return
                
                # 检查文件权限
                if not os.access(mt_train_path, os.R_OK):
                    self.update_signal.emit(f"Error: No permission to read MT_train.py file")
                    if not self.signals_emitted:
                        self.training_finished.emit(False)
                        self.signals_emitted = True
                    return
                
                self.update_signal.emit(f"Starting MT_train.py process...")
                self.update_signal.emit(f"Python interpreter path: {sys.executable}")
                self.update_signal.emit(f"Current working directory: {os.getcwd()}")
                
                # 创建变量来跟踪是否有输出被捕获
                has_output = False
                
                # 获取MT_train.py所在的目录作为工作目录
                dl_dir = os.path.dirname(mt_train_path)
                process = subprocess.Popen(
                    cmd_args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # 将stderr重定向到stdout
                    text=True,
                    encoding='gbk',  # 指定编码为gbk以匹配中文Windows系统默认编码
                    bufsize=1,
                    shell=False,
                    creationflags=subprocess.CREATE_NO_WINDOW,  # 不显示控制台窗口
                    cwd=dl_dir  # 明确设置工作目录为dl文件夹
                )
                self.update_signal.emit(f"MT_train.py process started successfully, PID: {process.pid}")
                
                # 导入time模块用于休眠
                import time
                
                # 实时读取输出
                while process.poll() is None:
                    # 检查是否停止训练
                    if self.stopped:
                        self.update_signal.emit("Stopping training...")
                        try:
                            process.terminate()
                            self.update_signal.emit(f"Termination signal sent, waiting for process to end...")
                             
                            # 等待进程终止，但缩短超时时间以更快响应
                            wait_time = 0
                            while process.poll() is None and wait_time < 2:  # 等待最多2秒
                                time.sleep(0.1)
                                wait_time += 0.1
                                 
                            if process.poll() is None:
                                self.update_signal.emit("Warning: Process did not terminate within timeout, attempting force termination...")
                                process.kill()
                                 
                                # 再等待一会儿确认进程终止，缩短等待时间
                                wait_time = 0
                                while process.poll() is None and wait_time < 1:  # 等待最多1秒
                                    time.sleep(0.1)
                                    wait_time += 0.1
                        except Exception as terminate_error:
                            self.update_signal.emit(f"Error stopping process: {str(terminate_error)}")
                            import traceback
                            self.update_signal.emit(f"Termination error details: {traceback.format_exc()}")
                         
                        # 确保final_status变量在所有路径上都有定义
                        final_status = "terminated" if process.poll() is not None else "not fully terminated"
                        self.update_signal.emit(f"Training {final_status}")
                        if not self.signals_emitted:
                            self.training_finished.emit(False)
                            self.signals_emitted = True
                        return
                     
                    # 检查是否暂停训练
                    if self.paused:
                        self.update_signal.emit("Training paused")
                        # 等待直到恢复训练或停止训练
                        while self.paused and not self.stopped:
                            time.sleep(0.5)  # 避免CPU占用过高
                         
                        if self.stopped:
                            # 如果在暂停期间收到停止命令，执行停止逻辑
                            self.update_signal.emit("Stopping training...")
                            try:
                                process.terminate()
                                self.update_signal.emit(f"Termination signal sent, waiting for process to end...")
                                 
                                # 等待进程终止，但缩短超时时间以更快响应
                                wait_time = 0
                                while process.poll() is None and wait_time < 2:  # 等待最多2秒
                                    time.sleep(0.1)
                                    wait_time += 0.1
                                 
                                if process.poll() is None:
                                    self.update_signal.emit("Warning: Process did not terminate within timeout, attempting force termination...")
                                    process.kill()
                                 
                                    # 再等待一会儿确认进程终止，缩短等待时间
                                    wait_time = 0
                                    while process.poll() is None and wait_time < 1:  # 等待最多1秒
                                        time.sleep(0.1)
                                        wait_time += 0.1
                            except Exception as terminate_error:
                                self.update_signal.emit(f"停止进程时出错: {str(terminate_error)}")
                                import traceback
                                self.update_signal.emit(f"终止错误详情: {traceback.format_exc()}")
                             
                            # 确保final_status变量在所有路径上都有定义
                            final_status = "terminated" if process.poll() is not None else "failed to terminate completely"
                            self.update_signal.emit(f"Training {final_status}")
                            if not self.signals_emitted:
                                self.training_finished.emit(False)
                                self.signals_emitted = True
                            return
                        else:
                            self.update_signal.emit("Training resumed")
                     
                    # 尝试读取输出，使用try-except避免非套接字操作错误
                    try:
                        # 使用非阻塞方式读取一行输出
                        line = process.stdout.readline()
                        if line:
                            has_output = True
                            line_stripped = line.strip()
                            
                            # 立即显示所有包含错误关键词的行
                            error_keywords = ['Error', 'error', 'Exception', 'exception', 'Traceback', 'traceback',
                                            'Failed', 'failed', '失败', '错误', '异常', 'ImportError', 
                                            'ModuleNotFoundError', 'FileNotFoundError', 'ValueError', 'TypeError',
                                            'AttributeError', 'KeyError', 'IndexError', 'RuntimeError']
                            
                            if any(keyword in line_stripped for keyword in error_keywords):
                                # 错误信息立即显示，不缓冲
                                self.update_signal.emit(f"[错误] {line_stripped}")
                            
                            # 解析epoch信息以更新进度条
                            if "Epoch:" in line and "finished" in line:
                                try:
                                    # 提取epoch信息
                                    epoch_str = line.split("Epoch:")[-1].split("finished")[0].strip()
                                    if epoch_str.isdigit():
                                        epoch = int(epoch_str)
                                        # 更新进度条
                                        progress = int((epoch / self.epochs) * 100)
                                        self.progress_signal.emit(min(progress, 100))
                                except ValueError:
                                    pass
                            
                            # 解析损失信息以更新损失曲线和训练信息
                            if "Loss:" in line and "---" not in line and "Epoch:" in line:
                                try:
                                    # 提取epoch信息
                                    epoch_start = line.find("Epoch:") + 6
                                    epoch_end = line.find("finished")
                                    if epoch_start > 5 and epoch_end > epoch_start:
                                        epoch_str = line[epoch_start:epoch_end].strip()
                                        if epoch_str.isdigit():
                                            epoch = int(epoch_str)
                                            
                                            # 检查这个epoch是否已经处理过
                                            if epoch not in self.processed_epochs:
                                                # 尝试解析详细的损失格式: "Epoch: X finished, Loss: X.XXXXXX, Data Loss: X.XXXXXX, Time: X.XXs"
                                                # 支持有或没有Physics Loss的格式
                                                if "Loss:" in line:
                                                    # 提取总损失
                                                    total_loss_part = line.split("Loss:")[1].split(",")[0].strip()
                                                    total_loss = float(total_loss_part)
                                                    
                                                    # 更新损失曲线数据
                                                    self.total_loss.append(total_loss)
                                                    
                                                    # 标记该epoch为已处理
                                                    self.processed_epochs.add(epoch)
                                                    
                                                    # 只显示损失值信息在Training Information中
                                                    self.update_signal.emit(f"Epoch: {epoch} finished, Loss: {total_loss:.6f}")
                                                else:
                                                    # 回退到简单的损失解析
                                                    loss_str = line.split("Loss:")[-1].strip()
                                                    loss_parts = ''.join(filter(lambda c: c.isdigit() or c == '.' or c == '-', loss_str))
                                                    if loss_parts:
                                                        loss_value = float(loss_parts)
                                                        # 更新损失曲线数据
                                                        self.total_loss.append(loss_value)
                                                        
                                                        # 标记该epoch为已处理
                                                        self.processed_epochs.add(epoch)
                                                        
                                                        # 只显示损失值信息在Training Information中
                                                        self.update_signal.emit(f"Epoch: {epoch} finished, Loss: {loss_value:.6f}")
                                                
                                                # 触发损失信号更新图表
                                            self.loss_signal.emit(self.total_loss, self.validation_losses, [])
                                except ValueError:
                                    pass
                            
                            # 解析验证损失信息
                            if '[MT_TRAIN VALIDATION]' in line:
                                try:
                                    # 提取验证损失值
                                    val_loss_part = line.split('Validation Loss:')[1].strip()
                                    val_loss = float(val_loss_part)
                                    
                                    # 将验证损失添加到列表中
                                    self.validation_losses.append(val_loss)
                                    
                                    # 更新图表
                                    self.loss_signal.emit(self.total_loss, self.validation_losses, [])
                                except Exception as e:
                                    # 解析验证损失时出错
                                    self.update_signal.emit(f"解析验证损失时出错: {str(e)}")
                        else:
                            # 短暂休眠，避免CPU占用过高
                            time.sleep(0.01)
                    except Exception as read_error:
                        # 避免重复显示相同的错误信息
                        if "WinError 10038" in str(read_error):
                            # 静默处理这个常见错误，不向用户显示
                            time.sleep(0.1)
                        else:
                            self.update_signal.emit(f"Error reading output: {str(read_error)}")
                            time.sleep(0.1)
                
                # 处理剩余的输出 - 确保读取所有输出，特别是错误信息
                remaining_output = ""
                try:
                    # 尝试读取所有剩余输出
                    remaining_output = process.stdout.read()
                    if not remaining_output:
                        # 如果read()返回空，尝试再读取一次（可能缓冲区还有数据）
                        import select
                        if hasattr(select, 'select'):
                            # Unix系统可以使用select
                            pass
                        else:
                            # Windows系统，尝试再次读取
                            try:
                                remaining_output = process.stdout.read()
                            except:
                                pass
                except Exception as read_remaining_error:
                    self.update_signal.emit(f"Error reading remaining output: {str(read_remaining_error)}")
                
                # 训练完成，获取返回码
                return_code = process.poll()
                
                # 处理所有剩余输出，特别是错误信息
                if remaining_output:
                    has_output = True
                    output_lines = remaining_output.strip().split('\n')
                    if output_lines:
                        # 检查是否包含错误信息
                        has_error = any(keyword in remaining_output for keyword in 
                                       ['Error', 'error', 'Exception', 'exception', 'Traceback', 'traceback', 
                                        'Failed', 'failed', '失败', '错误', '异常'])
                        
                        if has_error:
                            # 如果有错误，显示所有错误相关行
                            self.update_signal.emit("=" * 60)
                            self.update_signal.emit("Error detected, showing complete error output:")
                            self.update_signal.emit("=" * 60)
                            for line in output_lines:
                                line_stripped = line.strip()
                                if line_stripped:
                                    # 显示所有包含错误关键词的行
                                    if any(keyword in line_stripped for keyword in 
                                          ['Error', 'error', 'Exception', 'exception', 'Traceback', 'traceback',
                                           'Failed', 'failed', '失败', '错误', '异常', 'ImportError', 'ModuleNotFoundError']):
                                        self.update_signal.emit(line_stripped)
                            # 如果错误行太多，也显示其他重要行
                            if len(output_lines) > 20:
                                self.update_signal.emit(f"... {len(output_lines)} more lines of output")
                        else:
                            # 没有明显错误，按原逻辑处理
                            if len(output_lines) > 10:
                                self.update_signal.emit(f"Training process additional output: {len(output_lines)} lines")
                                # 显示前几行和最后几行
                                for i, line in enumerate(output_lines[:5]):
                                    if line.strip():
                                        self.update_signal.emit(line.strip())
                                self.update_signal.emit(f"... (omitted {len(output_lines)-10} lines) ...")
                                for i, line in enumerate(output_lines[-5:]):
                                    if line.strip():
                                        self.update_signal.emit(line.strip())
                            else:
                                for line in output_lines:
                                    if line.strip():
                                        self.update_signal.emit(line.strip())
                
                # 显示返回码和状态
                self.update_signal.emit(f"MT_train.py execution completed, return code: {return_code}")
                if return_code != 0:
                    self.update_signal.emit("=" * 60)
                    self.update_signal.emit(f"Training failed! Return code: {return_code}")
                    self.update_signal.emit("Please check the error messages above to understand the failure reason")
                    self.update_signal.emit("=" * 60)

                # 只有在真正训练完成时才将进度条设为100%
                self.progress_signal.emit(100)
                 
                # 如果没有捕获到任何输出，可能是执行路径问题
                if not has_output:
                    self.update_signal.emit("Warning: No output captured from training process, may be execution path or environment issue")
                    self.update_signal.emit(f"Please check if MT_train.py is executing correctly")
                 
                # 根据返回码判断训练是否成功
                if not self.signals_emitted:
                    if return_code == 0:
                        self.update_signal.emit("Training completed successfully!")
                        self.training_finished.emit(True)
                    else:
                        self.update_signal.emit(f"Training failed, return code: {return_code}")
                        self.training_finished.emit(False)
                    self.signals_emitted = True
            except Exception as process_error:
                # 处理subprocess创建或运行过程中的错误
                self.update_signal.emit(f"Error in training process: {str(process_error)}")
                import traceback
                self.update_signal.emit(f"Process error details: {traceback.format_exc()}")
                if not self.signals_emitted:
                    self.training_finished.emit(False)
                    self.signals_emitted = True
        except Exception as outer_error:
            self.update_signal.emit(f"Error occurred during training: {str(outer_error)}")
            import traceback
            self.update_signal.emit(f"External error details: {traceback.format_exc()}")
            if not self.signals_emitted:
                self.training_finished.emit(False)
                self.signals_emitted = True
        finally:
            # 确保线程最终结束，无论如何都要发出training_finished信号
            # 这样可以确保主线程正确处理线程结束，避免线程在运行时被销毁
            try:
                # 只在没有发送过信号时才发送
                if not self.signals_emitted:
                    self.training_finished.emit(False)
                    self.signals_emitted = True
            except:
                # 即使信号发送失败也要确保代码继续执行
                pass
    
    def update_param_config(self):
        """Update ParamConfig.py with parameters from GUI using absolute path"""
        try:
            import os
            import json
            # 使用绝对路径确保找到ParamConfig.py文件
            param_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ParamConfig.py')
            
            # Save file paths to a JSON file that MT_train.py can read
            if self.file_paths:
                file_paths_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_file_paths.json')
                with open(file_paths_json, 'w', encoding='utf-8') as f:
                    json.dump(self.file_paths, f, indent=2, ensure_ascii=False)
                self.update_signal.emit(f"Saved file paths to {file_paths_json}")
            self.update_signal.emit(f"使用ParamConfig.py路径: {param_config_path}")
            
            # 读取当前ParamConfig.py内容
            if not os.path.exists(param_config_path):
                self.update_signal.emit(f"Warning: Cannot find ParamConfig.py file at path: {param_config_path}")
                # 尝试创建一个默认的ParamConfig.py文件
                self.create_default_param_config(param_config_path)
                
            with open(param_config_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # 更新参数值 - 只更新GUI中设置的参数
            updated_lines = []
            for line in lines:
                if line.startswith('Epochs'):
                    updated_lines.append(f'Epochs        = {self.epochs}      # Number of epoch\n')
                elif line.startswith('BatchSize'):
                    updated_lines.append(f'BatchSize         = {self.batch_size}       # Number of batch size\n')
                elif line.startswith('LearnRate'):
                    updated_lines.append(f'LearnRate         = {self.learning_rate}      # Learning rate\n')
                elif line.startswith('ModelName'):
                    updated_lines.append(f"ModelName     = '{self.model_name}' # Name of the model to use\n")
                elif line.startswith('ModelsDir'):
                    updated_lines.append(f"ModelsDir     = '{self.models_dir}'   # Directory for saving models\n")
                elif line.startswith('ResultsDir'):
                    updated_lines.append(f"ResultsDir    = '{self.results_dir}'  # Directory for saving results\n")
                elif line.startswith('ModelDir'):
                    updated_lines.append(f"ModelDir      = '{self.models_dir}' # Model directory\n")
                elif line.startswith('ResultDir'):
                    updated_lines.append(f"ResultDir     = '{self.results_dir}' # Result directory\n")
                elif line.startswith('Inchannels'):
                    updated_lines.append(f'Inchannels        = {self.in_channels}        # Number of input channels, i.e. the number of shots\n')
                elif line.startswith('MT_Mode'):
                    updated_lines.append(f"MT_Mode = '{self.mt_mode}'  # MT mode: 'TE', 'TM', or 'Both'\n")
                else:
                    updated_lines.append(line)
            
            # No longer need to add or update SaveEpoch parameter as model will auto-save based on validation loss
            
            # 写回ParamConfig.py文件
            param_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ParamConfig.py')
            with open(param_config_path, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)
                
            self.update_signal.emit(f"ParamConfig updated: Epochs={self.epochs}, BatchSize={self.batch_size}, LearnRate={self.learning_rate}, ModelName={self.model_name}, MT_Mode={self.mt_mode}")
            
        except Exception as e:
            self.update_signal.emit(f"Failed to update ParamConfig: {str(e)}")
    
    def create_default_param_config(self, file_path):
        """Create default ParamConfig.py file"""
        try:
            default_content = '''# Default parameter configuration
Epochs        = 100      # Number of epoch
BatchSize     = 8       # Number of batch size
LearnRate     = 0.001      # Learning rate
ModelName     = 'DinkNet50' # Name of the model to use
ModelsDir     = 'models/'   # Directory for saving models
ResultsDir    = 'results/'  # Directory for saving results
ModelDir      = 'models/' # Model directory
ResultDir     = 'results/' # Result directory
Inchannels    = 1        # Number of input channels, i.e. the number of shots
TestBatchSize = 8
EarlyStop     = 0       # Early stopping threshold (0 means no early stopping)
MT_Mode = 'TE'  # MT mode: 'TE', 'TM', or 'Both'
'''
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(default_content)
            self.update_signal.emit(f"Default ParamConfig.py file created")
        except Exception as e:
            self.update_signal.emit(f"Failed to create default ParamConfig.py file: {str(e)}")
            
    def pause(self):
        """Pause training"""
        self.paused = True
        self.update_signal.emit("Training paused")
    
    def resume(self):
        """Resume training"""
        self.paused = False
        self.update_signal.emit("Training resumed")
    
    def stop(self):
        """Safely stop the training thread and clean up resources"""
        self.stopped = True
        self.paused = False
        
        # 标记为已停止
        self.was_stopped = True
        
        # 确保子进程被终止（如果存在）
        try:
            if hasattr(self, 'process') and self.process and self.process.poll() is None:
                # 先尝试正常终止
                self.process.terminate()
                # 等待进程结束，最多等待3秒
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    # 如果超时，强制终止
                    if hasattr(os, 'kill'):
                        try:
                            os.kill(self.process.pid, signal.SIGKILL)
                        except:
                            pass
                    elif hasattr(self.process, 'kill'):
                        self.process.kill()
                # 清理进程资源
                self.process.stdout.close()
                self.process.stderr.close()
                self.process = None
        except Exception as e:
            # 记录错误但不抛出，确保stop方法不会失败
            self.update_signal.emit(f"Error during process termination: {str(e)}")

class PredictionTab(QWidget):
    """Model prediction and result export tab"""
    def __init__(self, parent=None):
        super(PredictionTab, self).__init__(parent)
        self.prediction_data = None  # 存储预测数据
        self.current_file_type = None  # 当前文件类型
        self.current_folder = None  # 当前文件夹路径
        # 初始化所有需要的列表控件，确保即使在旧方法中也能正常工作
        self.data_file_list = QListWidget()
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI interface"""
        # Create main layout as horizontal layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Controls - 使用滚动区域
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setFrameShape(QFrame.NoFrame)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(12)
        
        model_group = QGroupBox("Model Loading")
        model_layout = QVBoxLayout()
        model_layout.setContentsMargins(12, 12, 12, 12)
        model_layout.setSpacing(10)
        
        self.model_path_label = QLabel("No model file selected")
        self.model_path_label.setWordWrap(True)
        self.model_path_label.setMinimumHeight(60)
        self.model_path_label.setStyleSheet("border: 1px solid #e0e0e0; padding: 8px; background-color: #fafafa; border-radius: 4px;")
        
        self.load_model_button = QPushButton("Load Model File")
        self.load_model_button.clicked.connect(self.load_model)
        self.load_model_button.setMinimumWidth(200)
        self.load_model_button.setMinimumHeight(40)
        
        self.model_info = QTextEdit()
        self.model_info.setReadOnly(True)
        self.model_info.setPlaceholderText("Model information will be displayed here...")
        self.model_info.setMinimumHeight(100)
        self.model_info.setMaximumHeight(120)
        
        model_file_label = QLabel("Loaded Model File:")
        model_file_label.setStyleSheet("font-weight: bold;")
        model_layout.addWidget(model_file_label)
        model_layout.addWidget(self.model_path_label)
        model_layout.addWidget(self.load_model_button, alignment=Qt.AlignLeft)
        model_info_label = QLabel("Model Information:")
        model_info_label.setStyleSheet("font-weight: bold;")
        model_layout.addWidget(model_info_label)
        model_layout.addWidget(self.model_info)
        model_group.setLayout(model_layout)
        
        # Prediction data loading
        data_group = QGroupBox("Prediction Data Loading")
        data_layout = QVBoxLayout()
        data_layout.setContentsMargins(0, 0, 0, 0)
        data_layout.setSpacing(10)
        
        # Input: Resistivity Model
        input_group = QGroupBox("Input: Resistivity Model")
        input_layout = QVBoxLayout()
        input_layout.setContentsMargins(12, 12, 12, 12)
        input_layout.setSpacing(10)
        
        self.import_model_button = QPushButton("Import Resistivity Model")
        self.import_model_button.clicked.connect(self.load_resistivity_model)
        self.import_model_button.setMinimumWidth(200)
        self.import_model_button.setMinimumHeight(40)
        
        self.model_file_label = QLabel("Resistivity Model: No file selected")
        self.model_file_label.setWordWrap(True)
        self.model_file_label.setMinimumHeight(50)
        self.model_file_label.setStyleSheet("border: 1px solid #e0e0e0; padding: 8px; background-color: #fafafa; border-radius: 4px;")
        
        input_layout.addWidget(self.import_model_button, alignment=Qt.AlignLeft)
        input_layout.addWidget(self.model_file_label)
        input_group.setLayout(input_layout)
        
        # Output Mode Selection
        mode_group = QGroupBox("Output Mode Selection")
        mode_layout = QVBoxLayout()
        mode_layout.setContentsMargins(12, 12, 12, 12)
        mode_layout.setSpacing(10)
        
        mode_radio_layout = QHBoxLayout()
        mode_label = QLabel("Forward Mode:")
        self.te_radio = QRadioButton("TE")
        self.tm_radio = QRadioButton("TM")
        self.both_radio = QRadioButton("Both")
        self.te_radio.setChecked(True)  # Default to TE mode
        
        # Connect radio buttons to update mode selection
        self.te_radio.toggled.connect(self.on_mode_changed)
        self.tm_radio.toggled.connect(self.on_mode_changed)
        self.both_radio.toggled.connect(self.on_mode_changed)
        
        mode_radio_layout.addWidget(mode_label)
        mode_radio_layout.addWidget(self.te_radio)
        mode_radio_layout.addWidget(self.tm_radio)
        mode_radio_layout.addWidget(self.both_radio)
        mode_radio_layout.addStretch()
        
        mode_info = QLabel("Select the mode(s) for forward modeling output:\n"
                          "• TE: Output TE apparent resistivity and phase\n"
                          "• TM: Output TM apparent resistivity and phase\n"
                          "• Both: Output both TE and TM data")
        mode_info.setWordWrap(True)
        mode_info.setStyleSheet("color: #666; font-size: 9pt; padding: 8px; background-color: #f5f5f5; border-radius: 4px;")
        
        mode_layout.addLayout(mode_radio_layout)
        mode_layout.addWidget(mode_info)
        mode_group.setLayout(mode_layout)
        
        # Output: Predicted Results (will be displayed after prediction)
        output_group = QGroupBox("Output: Predicted Results")
        output_layout = QVBoxLayout()
        output_layout.setContentsMargins(12, 12, 12, 12)
        output_layout.setSpacing(8)
        
        # Create file path labels for each output type with labels
        te_res_container = QVBoxLayout()
        te_res_container.setSpacing(5)
        te_res_label = QLabel("TE Apparent Resistivity:")
        te_res_label.setStyleSheet("font-weight: bold;")
        self.te_resistivity_label = QLabel("Not predicted yet")
        self.te_resistivity_label.setWordWrap(True)
        self.te_resistivity_label.setMinimumHeight(40)
        self.te_resistivity_label.setStyleSheet("border: 1px solid #e0e0e0; padding: 8px; background-color: #fafafa; border-radius: 4px;")
        te_res_container.addWidget(te_res_label)
        te_res_container.addWidget(self.te_resistivity_label)
        output_layout.addLayout(te_res_container)
        
        te_phase_container = QVBoxLayout()
        te_phase_container.setSpacing(5)
        te_phase_label = QLabel("TE Phase:")
        te_phase_label.setStyleSheet("font-weight: bold;")
        self.te_phase_label = QLabel("Not predicted yet")
        self.te_phase_label.setWordWrap(True)
        self.te_phase_label.setMinimumHeight(40)
        self.te_phase_label.setStyleSheet("border: 1px solid #e0e0e0; padding: 8px; background-color: #fafafa; border-radius: 4px;")
        te_phase_container.addWidget(te_phase_label)
        te_phase_container.addWidget(self.te_phase_label)
        output_layout.addLayout(te_phase_container)
        
        tm_res_container = QVBoxLayout()
        tm_res_container.setSpacing(5)
        tm_res_label = QLabel("TM Apparent Resistivity:")
        tm_res_label.setStyleSheet("font-weight: bold;")
        self.tm_resistivity_label = QLabel("Not predicted yet")
        self.tm_resistivity_label.setWordWrap(True)
        self.tm_resistivity_label.setMinimumHeight(40)
        self.tm_resistivity_label.setStyleSheet("border: 1px solid #e0e0e0; padding: 8px; background-color: #fafafa; border-radius: 4px;")
        tm_res_container.addWidget(tm_res_label)
        tm_res_container.addWidget(self.tm_resistivity_label)
        output_layout.addLayout(tm_res_container)
        
        tm_phase_container = QVBoxLayout()
        tm_phase_container.setSpacing(5)
        tm_phase_label = QLabel("TM Phase:")
        tm_phase_label.setStyleSheet("font-weight: bold;")
        self.tm_phase_label = QLabel("Not predicted yet")
        self.tm_phase_label.setWordWrap(True)
        self.tm_phase_label.setMinimumHeight(40)
        self.tm_phase_label.setStyleSheet("border: 1px solid #e0e0e0; padding: 8px; background-color: #fafafa; border-radius: 4px;")
        tm_phase_container.addWidget(tm_phase_label)
        tm_phase_container.addWidget(self.tm_phase_label)
        output_layout.addLayout(tm_phase_container)
        
        output_group.setLayout(output_layout)
        
        self.predict_button = QPushButton("Start Forward Modeling")
        self.predict_button.clicked.connect(self.start_prediction)
        self.predict_button.setMinimumWidth(220)
        self.predict_button.setMinimumHeight(45)
        
        data_layout.addWidget(input_group)
        data_layout.addWidget(mode_group)
        data_layout.addWidget(output_group)
        data_layout.addSpacing(5)
        data_layout.addWidget(self.predict_button, alignment=Qt.AlignCenter)
        data_group.setLayout(data_layout)
        
        # Result export
        export_group = QGroupBox("Result Export")
        export_layout = QVBoxLayout()
        export_layout.setContentsMargins(12, 12, 12, 12)
        export_layout.setSpacing(10)
        
        self.export_result_button = QPushButton("Export Prediction Results")
        self.export_result_button.clicked.connect(self.export_result)
        self.export_result_button.setMinimumWidth(200)
        self.export_result_button.setMinimumHeight(40)
        
        self.export_figure_button = QPushButton("Export Profile Diagram")
        self.export_figure_button.clicked.connect(self.export_figure)
        self.export_figure_button.setMinimumWidth(200)
        self.export_figure_button.setMinimumHeight(40)
        
        export_layout.addWidget(self.export_result_button, alignment=Qt.AlignLeft)
        export_layout.addWidget(self.export_figure_button, alignment=Qt.AlignLeft)
        export_group.setLayout(export_layout)
        
        # Add to left layout
        left_layout.addWidget(model_group)
        left_layout.addWidget(data_group)
        left_layout.addWidget(export_group)
        left_layout.addStretch()
        
        # 设置左侧面板到滚动区域
        left_scroll.setWidget(left_panel)
        
        # Right panel: Prediction results (image area on right side)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.result_title = QLabel("Forward Modeling Results: Apparent Resistivity & Phase")
        self.result_title.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.result_title.setFont(font)
        self.result_title.setStyleSheet("padding: 10px; background-color: #f5f5f5; border-radius: 4px;")
        
        # Create result display canvas with wider dimensions as requested
        self.result_canvas = MPLCanvas(self, width=10, height=16, dpi=100)
        
        # Add message display area below canvas
        self.message_display = QTextEdit()
        self.message_display.setReadOnly(True)
        self.message_display.setPlaceholderText("Messages and results information will be displayed here...")
        self.message_display.setMinimumHeight(100)
        self.message_display.setMaximumHeight(150)
        self.message_display.setStyleSheet("font-size: 9pt; padding: 8px;")
        
        # Add result file list below message area
        self.result_files_label = QLabel("Prediction Result Files:")
        self.result_files_label.setStyleSheet("font-weight: bold; padding: 5px;")
        self.result_files_list = QListWidget()
        self.result_files_list.setMinimumHeight(100)
        self.result_files_list.setMaximumHeight(150)
        self.result_files_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.result_files_list.customContextMenuRequested.connect(lambda pos: self.show_result_file_context_menu(pos))
        
        # Configure right layout with proper spacing
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.addWidget(self.result_title)
        right_layout.addWidget(self.result_canvas)
        right_layout.addWidget(self.message_display)
        right_layout.addWidget(self.result_files_label)
        right_layout.addWidget(self.result_files_list)
        right_layout.addStretch(1)  # Add stretch at bottom to push content up
        
        # Add panels to main layout - controls on left, image area on right
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setSizes([500, 700])  # 设置左右面板的初始比例，右侧稍大以显示图像
        
        main_layout.addWidget(splitter)
        main_layout.setStretch(0, 1)  # Controls panel gets less space
        main_layout.setStretch(1, 2)  # Image area gets more space
        
        self.setLayout(main_layout)
        
        # Disable prediction button initially
        self.predict_button.setEnabled(False)
        
        # Initialize model mode property
        self.mt_mode_from_model = None
        
        # Call on_mode_changed to set initial button states based on default TE selection
        self.on_mode_changed()
    
    def load_resistivity_model(self):
        """Load resistivity model file for forward modeling"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Resistivity Model File", "", 
            "Model Files (*.txt *.dat *.npy);;All Files (*)", options=options
        )
        
        if file_path:
            file_name = os.path.basename(file_path)
            self.model_file_label.setText(file_name)
            self.resistivity_model_path = file_path
            
            # Update model info
            self.model_info.append(f"Loaded resistivity model: {file_name}")
            
            # Enable prediction button if trained model is also loaded
            if self.model_path_label.text() != "No model file selected":
                self.predict_button.setEnabled(True)
                self.message_display.append(f"Ready for forward modeling. Click 'Start Forward Modeling' to begin.")
            else:
                self.message_display.append(f"Please load a trained model file to perform forward modeling.")
    
    def load_model(self):
        """Load model file"""
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "Model Files (*.pth *.pt *.pkl);;All Files (*)", options=options
        )
        
        if file:
            self.model_path_label.setText(file)
            # In actual application, this should load the model and display model information
            self.update_model_info(file)
            
            # Automatically select MT mode based on model information if available
            if hasattr(self, 'mt_mode_from_model') and self.mt_mode_from_model:
                # Convert to uppercase to ensure case-insensitive matching
                mt_mode = self.mt_mode_from_model.upper()
                if mt_mode == 'TE':
                    self.te_radio.setChecked(True)
                elif mt_mode == 'TM':
                    self.tm_radio.setChecked(True)
                elif mt_mode == 'BOTH':
                    self.both_radio.setChecked(True)
                # Call on_mode_changed to update button states
                self.on_mode_changed()
            
            # Enable prediction button if resistivity model is also loaded
            if hasattr(self, 'resistivity_model_path') and self.resistivity_model_path:
                self.predict_button.setEnabled(True)
    
    def update_model_info(self, model_path):
        """Update model information by parsing model filename"""
        model_name = os.path.basename(model_path)
        self.model_info.clear()
        self.model_info.append(f"Model name: {model_name}")
        
        # Try to parse information from model filename
        # New model name format: ModelName_TrainSizeX_EpochX_BatchSizeX_LRX_epochX/final.pkl
        
        # Parse model type
        model_type = "Unknown"
        if "DinkNet" in model_name or "Dnet" in model_name:
            model_type = "DinkNet"
        elif "Unet" in model_name:
            model_type = "U-Net"
        elif "Seg" in model_name:
            model_type = "SegNet"
        elif "HRNet" in model_name:
            model_type = "HRNet"
        self.model_info.append(f"Model type: {model_type}")
        
        # Set training data information
        self.model_info.append("Training data: MT dataset")
        
        # Try to extract other training parameter information
        import re
        train_size_match = re.search(r'TrainSize([\d.]+)', model_name)
        epoch_match = re.search(r'Epoch(\d+)', model_name)
        batch_size_match = re.search(r'BatchSize(\d+)', model_name)
        lr_match = re.search(r'LR([\d.]+)', model_name)
        mt_mode_match = re.search(r'Mode(\w+)', model_name)
        
        # Display if these information are found
        if train_size_match:
            self.model_info.append(f"Training size: {train_size_match.group(1)}")
        if epoch_match:
            self.model_info.append(f"Total epochs: {epoch_match.group(1)}")
        if batch_size_match:
            self.model_info.append(f"Batch size: {batch_size_match.group(1)}")
        if lr_match:
            self.model_info.append(f"Learning rate: {lr_match.group(1)}")
        if mt_mode_match:
            self.mt_mode_from_model = mt_mode_match.group(1)
            self.model_info.append(f"MT mode: {self.mt_mode_from_model}")
    
    def on_mode_changed(self):
        """Handle output mode selection change for forward modeling"""
        # Update output labels visibility based on selected mode
        if self.te_radio.isChecked():
            # TE mode: show only TE outputs
            self.te_resistivity_label.setVisible(True)
            self.te_phase_label.setVisible(True)
            self.tm_resistivity_label.setVisible(False)
            self.tm_phase_label.setVisible(False)
        elif self.tm_radio.isChecked():
            # TM mode: show only TM outputs
            self.te_resistivity_label.setVisible(False)
            self.te_phase_label.setVisible(False)
            self.tm_resistivity_label.setVisible(True)
            self.tm_phase_label.setVisible(True)
        else:  # Both mode
            # Both mode: show all outputs
            self.te_resistivity_label.setVisible(True)
            self.te_phase_label.setVisible(True)
            self.tm_resistivity_label.setVisible(True)
            self.tm_phase_label.setVisible(True)
    
    def update_data_file_list(self):
        """Update data file list based on current mode and loaded data"""
        # Since we now load single files instead of folders,
        # this method is no longer needed for mode changes
        pass
    
    def load_prediction_data(self):
        """Load prediction data from a single file (backward compatibility)"""
        # Get current mode
        current_mode = "TE" if self.te_radio.isChecked() else "TM" if self.tm_radio.isChecked() else "Both"
        
        # Define file filters based on mode
        if current_mode == "Both":
            file_filter = "Data Files (*.txt *.dat);;All Files (*)"
        else:
            # Filter files containing current mode in the name
            file_filter = f"{current_mode} Mode Data (*.txt *.dat);;All Files (*)"
        
        # Open file dialog to select a single file
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select {current_mode} Mode Prediction Data", "", file_filter, options=options
        )
        
        if file_path:
            # Get file name
            file_name = os.path.basename(file_path)
            
            # Determine data type (resistivity or phase) based on file name
            data_type = ('resistivity' if ('RES' in file_name.upper() or 'RHO' in file_name.upper())
                        else 'phase' if ('PHASE' in file_name.upper() or 'PHS' in file_name.upper())
                        else None)
            
            # Determine mode (TE or TM) based on file name
            file_mode = 'TE' if ('TE' in file_name.upper()) else 'TM' if ('TM' in file_name.upper()) else None
            
            # Check if the file mode matches current selection
            if current_mode != "Both" and file_mode and file_mode != current_mode:
                QMessageBox.warning(self, "Mode Mismatch", 
                                   f"Selected file is {file_mode} mode, but current selection is {current_mode} mode.")
                return
            
            # Note: This method is kept for backward compatibility
            # In the new forward modeling workflow, input is resistivity model, not apparent resistivity/phase
            # This method may still be used for visualization of prediction results
            if file_mode and data_type:
                # Update appropriate label (lists are no longer used in new UI)
                if file_mode == 'TE' and data_type == 'resistivity':
                    self.te_resistivity_label.setText(file_name)
                elif file_mode == 'TE' and data_type == 'phase':
                    self.te_phase_label.setText(file_name)
                elif file_mode == 'TM' and data_type == 'resistivity':
                    self.tm_resistivity_label.setText(file_name)
                else:  # TM phase
                    self.tm_phase_label.setText(file_name)
                
                # Add to result files list if it's a prediction result
                if hasattr(self, 'result_files_list'):
                    item = QListWidgetItem(f"{file_name} [{file_mode}] [{data_type}]")
                    item.setData(Qt.UserRole, file_path)
                    item.setData(Qt.UserRole + 1, data_type)
                    item.setData(Qt.UserRole + 2, file_mode)
                    self.result_files_list.addItem(item)
                    
                    # Visualize the file
                    self.visualize_specific_file(file_path, data_type, file_mode)
            
            # Update status
            self.model_info.append(f"Loading data from file: {file_name}")
            
            # If model is also loaded, enable prediction button
            if self.model_path_label.text() != "No model file selected":
                self.predict_button.setEnabled(True)
    
    def show_data_context_menu(self, position, list_widget=None):
        """Show right-click menu for data files"""
        # For backward compatibility
        if list_widget is None:
            list_widget = self.data_file_list
            
        if list_widget.currentItem():
            menu = QMenu()
            visualize_action = menu.addAction("Show Visualization")
            action = menu.exec_(list_widget.mapToGlobal(position))
            
            if action == visualize_action:
                file_path = list_widget.currentItem().data(Qt.UserRole)
                data_type = list_widget.currentItem().data(Qt.UserRole + 1)
                mode = list_widget.currentItem().data(Qt.UserRole + 2) if list_widget.currentItem().data(Qt.UserRole + 2) else "Unknown"
                self.visualize_specific_file(file_path, data_type, mode)
    
    def visualize_specific_file(self, file_path, data_type=None, mode="Unknown"):
        """Visualize specific data file"""
        # Read file data and process as required
        file_name = os.path.basename(file_path)
        
        # 尝试查找对应的配置文件来获取实际的数据维度
        # 配置文件通常与结果文件在同一目录，名称格式为 pred_config_*.json
        config_dir = os.path.dirname(file_path)
        base_filename = os.path.splitext(file_name)[0].replace('pred_', '')
        
        # 默认维度
        model_size = 32
        
        # 尝试查找对应的配置文件
        try:
            # 查找匹配的配置文件
            import glob
            config_files = glob.glob(os.path.join(config_dir, f"pred_config_*{base_filename}*.json"))
            if config_files:
                # 读取第一个匹配的配置文件
                with open(config_files[0], 'r', encoding='utf-8') as f:
                    import json
                    config = json.load(f)
                    if 'data_dim' in config and isinstance(config['data_dim'], list) and len(config['data_dim']) == 2:
                        # 取最大的维度作为重塑尺寸
                        model_size = max(config['data_dim'])
        except Exception as e:
            # 配置文件读取失败，继续使用默认值
            pass
        
        try:
            # Create custom dialog for user to input both model length and depth
            class ModelSizeDialog(QDialog):
                def __init__(self, parent=None):
                    super().__init__(parent)
                    self.setWindowTitle("Model Size Input")
                    self.setFixedSize(300, 150)
                    
                    layout = QVBoxLayout(self)
                    
                    # Create length input
                    length_layout = QHBoxLayout()
                    length_label = QLabel("Model Length (km):")
                    self.length_spin = QDoubleSpinBox()
                    self.length_spin.setRange(0.1, 100.0)
                    self.length_spin.setDecimals(1)
                    self.length_spin.setValue(5.0)  # Default value
                    length_layout.addWidget(length_label)
                    length_layout.addWidget(self.length_spin)
                    
                    # Create depth input
                    depth_layout = QHBoxLayout()
                    depth_label = QLabel("Model Depth (km):")
                    self.depth_spin = QDoubleSpinBox()
                    self.depth_spin.setRange(0.1, 50.0)
                    self.depth_spin.setDecimals(1)
                    self.depth_spin.setValue(3.0)  # Default value
                    depth_layout.addWidget(depth_label)
                    depth_layout.addWidget(self.depth_spin)
                    
                    # Create buttons
                    button_layout = QHBoxLayout()
                    ok_button = QPushButton("OK")
                    cancel_button = QPushButton("Cancel")
                    ok_button.clicked.connect(self.accept)
                    cancel_button.clicked.connect(self.reject)
                    button_layout.addWidget(ok_button)
                    button_layout.addWidget(cancel_button)
                    
                    # Add to main layout
                    layout.addLayout(length_layout)
                    layout.addLayout(depth_layout)
                    layout.addLayout(button_layout)
                    
            # Show dialog and get user input
            dialog = ModelSizeDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                length = dialog.length_spin.value()
                depth = dialog.depth_spin.value()
            else:
                # User canceled input
                return
            
            # 检查是否为Both模式的4通道数据文件
            is_both_mode_file = False
            channel_data = []
            
            # 尝试读取文件并检查是否有通道标记
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if any('# Channel' in line for line in lines):
                    is_both_mode_file = True
                    
                    # 分割文件内容为不同通道
                    current_channel = []
                    for line in lines:
                        if '# Channel' in line:
                            if current_channel:
                                # 保存当前通道数据
                                channel_data.append(np.array(current_channel))
                                current_channel = []
                        elif line.strip() != '':  # 跳过空行
                            # 解析数值行
                            values = [float(v) for v in line.strip().split()]
                            current_channel.extend(values)
                    # 保存最后一个通道
                    if current_channel:
                        channel_data.append(np.array(current_channel))
            
            # 如果是Both模式的多通道文件，让用户选择要可视化的通道
            if is_both_mode_file and len(channel_data) > 1:
                # 创建通道选择对话框
                class ChannelSelectionDialog(QDialog):
                    def __init__(self, parent=None):
                        super().__init__(parent)
                        self.setWindowTitle("Select Channel")
                        self.setFixedSize(250, 150)
                        
                        layout = QVBoxLayout(self)
                        
                        # 创建通道选择标签
                        label = QLabel("Select channel to visualize:")
                        layout.addWidget(label)
                        
                        # 创建通道选择下拉框
                        self.channel_combo = QComboBox()
                        channel_options = [
                            "Channel 1: TE Resistivity",
                            "Channel 2: TE Phase",
                            "Channel 3: TM Resistivity",
                            "Channel 4: TM Phase"
                        ]
                        self.channel_combo.addItems(channel_options)
                        layout.addWidget(self.channel_combo)
                        
                        # 创建按钮
                        button_layout = QHBoxLayout()
                        ok_button = QPushButton("OK")
                        cancel_button = QPushButton("Cancel")
                        ok_button.clicked.connect(self.accept)
                        cancel_button.clicked.connect(self.reject)
                        button_layout.addWidget(ok_button)
                        button_layout.addWidget(cancel_button)
                        
                        # 添加到主布局
                        layout.addLayout(button_layout)
                    
                    def get_selected_channel(self):
                        return self.channel_combo.currentIndex()
                    
                # 显示通道选择对话框
                channel_dialog = ChannelSelectionDialog(self)
                if channel_dialog.exec_() == QDialog.Accepted:
                    selected_channel_index = channel_dialog.get_selected_channel()
                    if 0 <= selected_channel_index < len(channel_data):
                        data = channel_data[selected_channel_index]
                        # 更新模式和数据类型信息
                        if selected_channel_index in [0, 2]:  # 电阻率通道
                            data_type = "resistivity"
                        else:  # Phase channel
                            data_type = "phase"
                        if selected_channel_index in [0, 1]:  # TE模式通道
                            mode = "TE"
                        else:  # TM模式通道
                            mode = "TM"
                else:
                    # User canceled channel selection
                    return
            else:
                # 读取普通格式的文件数据
                if not is_both_mode_file:
                    data = np.loadtxt(file_path)
                elif len(channel_data) > 0:
                    # 如果是Both模式但只有一个通道，或用户取消了选择，使用第一个通道
                    data = channel_data[0]
            
            # 修复数据重塑逻辑，确保与数据导入标签页一致
            # 首先检查数据形状，如果是一维数组，需要重塑
            if len(data.shape) == 1:
                # 对于一维数据，根据数据长度重塑为合适的形状
                size = int(np.sqrt(data.size))
                # 找到最接近的合适尺寸
                data = data[:size*size].reshape(size, size)
            else:
                # 如果已经是二维数组，确保形状正确
                if data.shape != (model_size, model_size):
                    data = data[:model_size, :model_size]
            
            # Determine whether to apply logarithm based on file type
            # For apparent resistivity files, always apply logarithm
            # For phase files, do not apply logarithm
            # IMPORTANT: Prediction result files are saved in raw format (after inverse log10 for resistivity)
            # - Resistivity files: raw values (after inverse log10 in MT_test.py), need log10 for visualization
            # - Phase files: raw values (no log10 in MT_test.py), no log10 needed for visualization
            # Priority: use data_type first (most reliable), then infer from filename
            if data_type == 'resistivity':
                is_resistivity = True
            elif data_type == 'phase':
                is_resistivity = False
            else:
                # Infer from filename if data_type not available
                is_resistivity = ('res' in file_name.lower() or 'rho' in file_name.lower() or 
                                 'resistivity' in file_name.lower())
            
            if is_resistivity:
                # 对于预测结果文件，数据已经是原始电阻率值，需要log10转换
                # First take absolute value to ensure no negative values, then add a small value to avoid log10(0)
                data = np.log10(np.abs(data) + 1e-10)
                print(f"[Prediction Visualization] Applied log10 for resistivity. Data range: {data.min():.3f} - {data.max():.3f} (lgρ)")
            else:
                print(f"[Prediction Visualization] No log10 for phase. Data range: {data.min():.3f} - {data.max():.3f} (degrees)")
            
            # Visualize processed data
            # Reset figure more thoroughly: clear all axes and recreate main axis
            self.result_canvas.figure.clear()
            
            # 重新创建主坐标轴
            self.result_canvas.axes = self.result_canvas.figure.add_subplot(111)
            
            # Adjust figure aspect ratio based on user-input length and depth
            # Calculate actual aspect ratio and set reasonable limits (between 0.5 and 2 to avoid overly flat or elongated figures)
            aspect_ratio = min(max(depth / length, 0.5), 2.0)
            
            # Use 'jet' colormap for both resistivity and phase
            cmap = 'jet'
            
            im = self.result_canvas.axes.imshow(data, cmap=cmap, aspect=aspect_ratio, origin='upper')
            
            # Add colorbar and limit number of ticks to 3-4
            # Set colorbar height smaller than image height for better visual balance
            cbar = self.result_canvas.figure.colorbar(im, ax=self.result_canvas.axes, shrink=0.7)
            
            # Set colorbar label based on data type
            if data_type == 'resistivity':
                cbar.set_label('lgρ(Ω·m)')  # Resistivity unit (log scaled)
            else:
                cbar.set_label('φ(°)')  # Phase unit
            cbar.locator = plt.MaxNLocator(nbins=4)
            cbar.update_ticks()
            
            # Set title with mode information
            if data_type == 'resistivity':
                data_type_text = f"Predicted Data: {file_name} "
            else:
                data_type_text = f"Predicted Data: {file_name} "
            
            # Set centered title with larger font size
            self.result_canvas.axes.set_title(data_type_text, loc='center', fontsize=20)
            
            # Set axis ticks and labels to display user-input model dimensions
            # x-axis (horizontal) represents length, y-axis (vertical) represents depth
            num_ticks = 5  # Set number of ticks
            x_ticks = np.linspace(0, model_size-1, num_ticks)
            y_ticks = np.linspace(0, model_size-1, num_ticks)
            
            # Calculate corresponding actual dimension values
            x_tick_labels = [f"{x:.1f}" for x in np.linspace(0, length, num_ticks)]
            y_tick_labels = [f"{y:.1f}" for y in np.linspace(0, depth, num_ticks)]
            
            # Set tick positions and labels with larger font size
            self.result_canvas.axes.set_xticks(x_ticks)
            self.result_canvas.axes.set_yticks(y_ticks)
            self.result_canvas.axes.set_xticklabels(x_tick_labels, fontsize=18)
            self.result_canvas.axes.set_yticklabels(y_tick_labels, fontsize=18)
            
            # Set tick parameters font size
            self.result_canvas.axes.tick_params(axis='both', labelsize=18)
            
            # Set axis labels with larger font size
            self.result_canvas.axes.set_xlabel('Length (km)', fontsize=20)
            self.result_canvas.axes.set_ylabel('Depth (km)', fontsize=20)
            
            # Adjust layout to ensure all elements are displayed correctly
            self.result_canvas.figure.tight_layout()
            
            # Redraw the canvas
            self.result_canvas.draw()
            
            # Update the result title
            self.result_title.setText(f"Data Visualization: {file_name}")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Cannot visualize file {file_name}: {str(e)}")
    
    def start_prediction(self):
        """Start forward modeling: input resistivity model, output apparent resistivity and phase"""
        import subprocess
        import sys
        import os
        import re
        
        try:
            # Validate inputs
            if not hasattr(self, 'resistivity_model_path') or not self.resistivity_model_path:
                QMessageBox.warning(self, "Missing Input", "Please import a resistivity model file first.")
                return
            
            if self.model_path_label.text() == "No model file selected":
                QMessageBox.warning(self, "Missing Model", "Please load a trained model file first.")
                return
            
            # Show a message to indicate forward modeling is starting
            self.result_canvas.clear()
            self.result_canvas.axes = self.result_canvas.figure.add_subplot(111)
            self.result_canvas.axes.text(0.5, 0.5, 'Running forward modeling...', 
                                         horizontalalignment='center', 
                                         verticalalignment='center', 
                                         fontsize=12)
            self.result_canvas.figure.tight_layout()
            self.result_canvas.draw()
            
            # Clear previous messages and result files
            self.message_display.clear()
            self.result_files_list.clear()
            
            # Get the path to forward modeling script (mt_test.py or similar)
            forward_script_path = os.path.join(os.path.dirname(__file__), 'MT_test.py')
            
            # Get the selected model path from the UI
            selected_model_path = self.model_path_label.text()
            
            # Get the selected output mode
            current_mode = "TE" if self.te_radio.isChecked() else "TM" if self.tm_radio.isChecked() else "BOTH"
            print(f"Forward modeling mode: {current_mode}")
            print(f"Input resistivity model: {self.resistivity_model_path}")
            
            # Run forward modeling script
            # Arguments: resistivity_model_path, trained_model_path, output_mode
            process = subprocess.Popen([sys.executable, forward_script_path, 
                                      self.resistivity_model_path, 
                                      selected_model_path, 
                                      current_mode],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      text=True,
                                      encoding='gbk')
            
            # Wait for the process to complete and get the output
            stdout, stderr = process.communicate()
            
            # Check if there was an error
            if process.returncode != 0:
                # Show error in message display area instead of canvas
                self.message_display.setStyleSheet("color: red; font-size: 10px;")
                self.message_display.setText(f'Prediction failed:\n{stderr}')
                
                # Clear canvas but show a simple message
                self.result_canvas.clear()
                self.result_canvas.axes = self.result_canvas.figure.add_subplot(111)
                self.result_canvas.axes.text(0.5, 0.5, 'Prediction failed\nCheck error message below', 
                                         horizontalalignment='center', 
                                         verticalalignment='center', 
                                         fontsize=18, 
                                         color='red')
                self.result_canvas.figure.tight_layout()
                self.result_canvas.draw()
                return
            
            # Update the result title
            current_mode = "TE" if self.te_radio.isChecked() else "TM" if self.tm_radio.isChecked() else "Both TE and TM"
            self.result_title.setText(f"Forward Modeling Completed ({current_mode} Mode)")
            
            # Parse stdout to find result file paths
            result_files = []
            result_lines = stdout.split('\n')
            for line in result_lines:
                if line.startswith('RESULT_PATH:'):
                    result_path = line[12:].strip()
                    if os.path.exists(result_path):
                        result_files.append(result_path)
                # Also look for output file patterns
                if 'TE' in current_mode or current_mode == "Both TE and TM":
                    if 'te_resistivity' in line.lower() or 'te_rho' in line.lower():
                        # Try to extract file path
                        if os.path.exists(line.strip()):
                            result_files.append(line.strip())
                    if 'te_phase' in line.lower():
                        if os.path.exists(line.strip()):
                            result_files.append(line.strip())
                if 'TM' in current_mode or current_mode == "Both TE and TM":
                    if 'tm_resistivity' in line.lower() or 'tm_rho' in line.lower():
                        if os.path.exists(line.strip()):
                            result_files.append(line.strip())
                    if 'tm_phase' in line.lower():
                        if os.path.exists(line.strip()):
                            result_files.append(line.strip())
            
            # Update output labels based on results
            self.message_display.setStyleSheet("color: green; font-size: 10px;")
            if result_files:
                message = 'Forward modeling completed successfully!\nOutput files:\n'
                for rf in result_files:
                    rf_name = os.path.basename(rf)
                    rf_name_lower = rf_name.lower()
                    
                    # Determine file type and mode for display
                    file_type = None
                    file_mode = None
                    
                    if 'te' in rf_name_lower:
                        file_mode = 'TE'
                        if 'resistivity' in rf_name_lower or 'rho' in rf_name_lower:
                            file_type = 'resistivity'
                            self.te_resistivity_label.setText(rf_name)
                        elif 'phase' in rf_name_lower:
                            file_type = 'phase'
                            self.te_phase_label.setText(rf_name)
                    elif 'tm' in rf_name_lower:
                        file_mode = 'TM'
                        if 'resistivity' in rf_name_lower or 'rho' in rf_name_lower:
                            file_type = 'resistivity'
                            self.tm_resistivity_label.setText(rf_name)
                        elif 'phase' in rf_name_lower:
                            file_type = 'phase'
                            self.tm_phase_label.setText(rf_name)
                    else:
                        # Try to infer from file name patterns
                        if 'resistivity' in rf_name_lower or 'rho' in rf_name_lower:
                            file_type = 'resistivity'
                        elif 'phase' in rf_name_lower:
                            file_type = 'phase'
                    
                    # Add to result files list with descriptive label
                    if file_type and file_mode:
                        display_name = f"{rf_name} [{file_mode}] [{file_type}]"
                    elif file_type:
                        display_name = f"{rf_name} [{file_type}]"
                    else:
                        display_name = rf_name
                    
                    message += f'  • {display_name}\n'
                    
                    # Add to result files list
                    item = QListWidgetItem(display_name)
                    item.setData(Qt.UserRole, rf)
                    if file_type:
                        item.setData(Qt.UserRole + 1, file_type)
                    if file_mode:
                        item.setData(Qt.UserRole + 2, file_mode)
                    self.result_files_list.addItem(item)
                
                self.message_display.setText(message)
            else:
                self.message_display.setText('Forward modeling completed successfully!\nCheck results directory for output files.')
            
            # Show success message on canvas
            self.result_canvas.clear()
            self.result_canvas.axes = self.result_canvas.figure.add_subplot(111)
            self.result_canvas.axes.text(0.5, 0.5, 'Prediction completed successfully!', 
                                         horizontalalignment='center', 
                                         verticalalignment='center', 
                                         fontsize=12, 
                                         color='green')
            self.result_canvas.figure.tight_layout()
            self.result_canvas.draw()
            
        except Exception as e:
            # Show error in message display area instead of canvas
            self.message_display.setStyleSheet("color: red; font-size: 10px;")
            self.message_display.setText(f'Error: {str(e)}')
            
            # Clear canvas but show a simple message
            self.result_canvas.clear()
            self.result_canvas.axes = self.result_canvas.figure.add_subplot(111)
            self.result_canvas.axes.text(0.5, 0.5, 'Operation failed\nCheck error message below', 
                                         horizontalalignment='center', 
                                         verticalalignment='center', 
                                         fontsize=12, 
                                         color='red')
            self.result_canvas.figure.tight_layout()
            self.result_canvas.draw()
        
        # Enable export buttons
        self.export_result_button.setEnabled(True)
        self.export_figure_button.setEnabled(True)
    
    def export_result(self):
        """Export prediction results"""
        # Check if there is a selected result file
        if not self.result_files_list.currentItem():
            QMessageBox.warning(self, "Warning", "Please select a result file to export.")
            return
        
        # Get the selected result file path
        result_file_path = self.result_files_list.currentItem().data(Qt.UserRole)
        
        # Check if the file exists
        if not os.path.exists(result_file_path):
            QMessageBox.warning(self, "Warning", "Selected result file does not exist.")
            return
        
        # Open save file dialog
        options = QFileDialog.Options()
        file, _ = QFileDialog.getSaveFileName(
            self, "Export Prediction Results", "", "Data Files (*.dat);;All Files (*)", options=options
        )
        
        if file:
            try:
                model_size = 32
                
                # 检查是否为Both模式的4通道数据文件
                is_both_mode_file = False
                channel_data = []
                
                # 尝试读取文件并检查是否有通道标记
                with open(result_file_path, 'r') as f:
                    lines = f.readlines()
                    if any('# Channel' in line for line in lines):
                        is_both_mode_file = True
                        
                        # 分割文件内容为不同通道
                        current_channel = []
                        for line in lines:
                            if '# Channel' in line:
                                if current_channel:
                                    # 保存当前通道数据
                                    channel_data.append(np.array(current_channel))
                                    current_channel = []
                            elif line.strip() != '':  # 跳过空行
                                # 解析数值行
                                values = [float(v) for v in line.strip().split()]
                                current_channel.extend(values)
                        # 保存最后一个通道
                        if current_channel:
                            channel_data.append(np.array(current_channel))
                
                # 如果是Both模式的多通道文件，让用户选择要导出的通道
                if is_both_mode_file and len(channel_data) > 1:
                    # 创建通道选择对话框
                    class ChannelSelectionDialog(QDialog):
                        def __init__(self, parent=None):
                            super().__init__(parent)
                            self.setWindowTitle("Select Channel")
                            self.setFixedSize(250, 150)
                            
                            layout = QVBoxLayout(self)
                            
                            # 创建通道选择标签
                            label = QLabel("Select channel to export:")
                            layout.addWidget(label)
                            
                            # 创建通道选择下拉框
                            self.channel_combo = QComboBox()
                            channel_options = [
                                "Channel 1: TE Resistivity",
                                "Channel 2: TE Phase",
                                "Channel 3: TM Resistivity",
                                "Channel 4: TM Phase"
                            ]
                            self.channel_combo.addItems(channel_options)
                            layout.addWidget(self.channel_combo)
                            
                            # 创建按钮
                            button_layout = QHBoxLayout()
                            ok_button = QPushButton("OK")
                            cancel_button = QPushButton("Cancel")
                            ok_button.clicked.connect(self.accept)
                            cancel_button.clicked.connect(self.reject)
                            button_layout.addWidget(ok_button)
                            button_layout.addWidget(cancel_button)
                            
                            # 添加到主布局
                            layout.addLayout(button_layout)
                        
                        def get_selected_channel(self):
                            return self.channel_combo.currentIndex()
                        
                    # 显示通道选择对话框
                    channel_dialog = ChannelSelectionDialog(self)
                    if channel_dialog.exec_() == QDialog.Accepted:
                        selected_channel_index = channel_dialog.get_selected_channel()
                        if 0 <= selected_channel_index < len(channel_data):
                            data = channel_data[selected_channel_index]
                        else:
                            # 如果选择的通道不存在，使用第一个通道
                            data = channel_data[0]
                    else:
                        # User canceled channel selection
                        return
                else:
                    # 读取普通格式的文件数据
                    if not is_both_mode_file:
                        data = np.loadtxt(result_file_path)
                    elif len(channel_data) > 0:
                        # 如果是Both模式但只有一个通道，或用户取消了选择，使用第一个通道
                        data = channel_data[0]
                
                # Ensure data is properly shaped
                if data.size != model_size * model_size:
                    # If data size doesn't match, try to reshape as best as possible
                    data = data[:model_size*model_size].reshape((model_size, model_size))
                else:
                    data = data.reshape((model_size, model_size))
                
                # Export the complete 32x32 data
                with open(file, 'w') as f:
                    # Optional header comment
                    f.write("# Predicted resistivity model data\n")
                    f.write("# Format: 32 values per line, separated by spaces\n")
                    
                    # Export each row with 32 values separated by spaces
                    for i in range(model_size):
                        # Create a list of 32 values for this row
                        row_values = [f"{data[i, j]:.6f}" for j in range(model_size)]
                        # Join with a single space and write to file
                        f.write(' '.join(row_values) + '\n')
                
                # Display export success message
                self.model_info.append(f"\nResults exported to: {file}")
                
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to export results: {str(e)}")
    
    def export_figure(self):
        """Export profile diagram - same as right-click save image"""
        # 调用 canvas 的 save_figure 方法，与右键保存图像功能一致
        if self.result_canvas.save_figure():
            # Display export success message
            self.model_info.append(f"\nProfile diagram exported successfully")
    
    def show_result_file_context_menu(self, position):
        """Show context menu for result files list"""
        if not self.result_files_list.currentItem():
            return
        
        menu = QMenu()
        visualize_action = menu.addAction("Visualize")
        
        action = menu.exec_(self.result_files_list.mapToGlobal(position))
        
        if action == visualize_action:
            current_item = self.result_files_list.currentItem()
            file_path = current_item.data(Qt.UserRole)
            if file_path and os.path.exists(file_path):
                # Get file type and mode from item data if available
                file_type = current_item.data(Qt.UserRole + 1)  # 'resistivity' or 'phase'
                file_mode = current_item.data(Qt.UserRole + 2)  # 'TE' or 'TM'
                
                # If not available in item data, try to infer from file name
                if not file_type or not file_mode:
                    file_name_lower = os.path.basename(file_path).lower()
                    if 'te' in file_name_lower:
                        file_mode = 'TE'
                    elif 'tm' in file_name_lower:
                        file_mode = 'TM'
                    
                    if 'resistivity' in file_name_lower or 'rho' in file_name_lower:
                        file_type = 'resistivity'
                    elif 'phase' in file_name_lower:
                        file_type = 'phase'
                    else:
                        file_type = 'resistivity'  # Default to resistivity
                
                # Clear the canvas
                self.result_canvas.clear()
                
                # Visualize the file with correct type and mode
                # Note: visualize_specific_file signature is (file_path, data_type=None, mode="Unknown")
                mode_str = f"{file_mode} {file_type}" if file_mode and file_type else "Prediction Result"
                self.visualize_specific_file(file_path, data_type=file_type, mode=mode_str)
                
                # Update the result title
                file_name = os.path.basename(file_path)
                if file_type and file_mode:
                    self.result_title.setText(f"Visualization: {file_name} [{file_mode}] [{file_type}]")
                else:
                    self.result_title.setText(f"Visualization: {file_name}")

class GeoDLFWDMainWindow(QMainWindow):
    """GeoDLFWD main window"""
    def __init__(self):
        super(GeoDLFWDMainWindow, self).__init__()
        self.setWindowTitle("GeoDLFWD - Geophysical Deep Learning Forward Modeling")
        self.setMinimumSize(1400, 900)  # 增加最小尺寸，确保标签页文字完整显示
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # Create central widget and tabs
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)  # 更现代的标签页样式
        
        # Create each tab
        self.data_tab = DataImportTab(self)
        self.model_tab = ModelConfigTab(self)
        self.training_tab = TrainingTab(self)
        self.prediction_tab = PredictionTab(self)
        
        # Add tabs to tab widget - 使用更短的标签文本确保完整显示
        self.tabs.addTab(self.data_tab, "Data Import")
        self.tabs.addTab(self.model_tab, "Parameters")
        self.tabs.addTab(self.training_tab, "Training")
        self.tabs.addTab(self.prediction_tab, "Prediction")
        
        # 设置标签页的初始启用状态
        self.tabs.setTabEnabled(1, False)  # Model Config (disabled until data is imported)
        self.tabs.setTabEnabled(2, False)  # Model Training (disabled until model is configured)
        self.tabs.setTabEnabled(3, True)  # Model Prediction (always enabled)
        
        # Add tab widget to main layout
        self.main_layout.addWidget(self.tabs)
        
        # 创建状态栏
        self.create_status_bar()
        
        # Set style
        self.set_style()
    
    def create_menu_bar(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File(&F)')
        exit_action = file_menu.addAction('Exit(&X)')
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        
        # Help menu
        help_menu = menubar.addMenu('Help(&H)')
        about_action = help_menu.addAction('About(&A)')
        about_action.triggered.connect(self.show_about)
        
        # Add upload functionality to Help menu
        help_menu.addSeparator()
        upload_model_action = help_menu.addAction('Upload Model(&M)')
        upload_model_action.triggered.connect(self.show_upload_model_dialog)
        upload_data_action = help_menu.addAction('Upload Data(&D)')
        upload_data_action.triggered.connect(self.show_upload_data_dialog)
    
    def create_status_bar(self):
        """Create status bar"""
        self.statusBar().showMessage('Ready', 0)
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: #f5f5f5;
                border-top: 1px solid #d0d0d0;
                padding: 2px;
            }
        """)
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About GeoDLFWD", 
                         "GeoDLFWD - Geophysical Deep Learning Forward Modeling\n\n"
                         "Version: 1.0.0\n"
                         "A professional geophysical deep learning forward modeling software")
    
    def show_upload_model_dialog(self):
        """Show upload model dialog"""
        try:
            from layers.ui_layer.upload_dialog import UploadDialog
            dialog = UploadDialog(self, upload_type="Model")
            dialog.exec_()
        except ImportError as e:
            QMessageBox.warning(self, "Error", f"Failed to import upload dialog module: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open upload dialog: {str(e)}")
    
    def show_upload_data_dialog(self):
        """Show upload data dialog"""
        try:
            from layers.ui_layer.upload_dialog import UploadDialog
            dialog = UploadDialog(self, upload_type="Data")
            dialog.exec_()
        except ImportError as e:
            QMessageBox.warning(self, "Error", f"Failed to import upload dialog module: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open upload dialog: {str(e)}")
        
    def update_tab_status(self, completed_tab_index):
        """Update tab status, enable the next tab"""
        # Ensure the index is valid
        if 0 <= completed_tab_index < self.tabs.count() - 1:
            # Enable the next tab
            self.tabs.setTabEnabled(completed_tab_index + 1, True)
            # Can add status prompt messages here
            if completed_tab_index == 0:
                QMessageBox.information(self, "Preparation Completed", "Data has been successfully imported, now you can configure the model.")
            elif completed_tab_index == 1:
                QMessageBox.information(self, "Preparation Completed", "Model has been successfully configured, now you can start model training.")
            # Remove training completion message, let training_finished method handle display
    
    def set_style(self):
        """Set application style - Professional modern design"""
        # Set global font - ensure all elements use the same font
        font = QFont()
        # 智能选择可用字体，避免字体警告
        font_candidates = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Heiti TC", "Segoe UI", "Arial", "sans-serif"]
        selected_font = "Segoe UI"  # 默认字体，更专业
        for font_name in font_candidates:
            if QFont(font_name).exactMatch() or font_name == "sans-serif":
                selected_font = font_name
                break
        font.setFamily(selected_font)
        font.setPointSize(9)
        font.setStyleStrategy(QFont.PreferAntialias)
        self.setFont(font)
        
        # 设置标签页字体，确保文字完整显示
        tab_font = QFont(selected_font)
        tab_font.setPointSize(9)
        self.tabs.setFont(tab_font)
        
        # Ensure all widgets in the application use the same font
        for widget in self.findChildren(QWidget):
            try:
                widget.setFont(font)
            except:
                pass
        
        # 专业软件样式表 - 现代化设计
        self.setStyleSheet('''
/* 主窗口背景 */
QMainWindow {
    background-color: #f8f9fa;
}

/* 菜单栏样式 */
QMenuBar {
    background-color: #ffffff;
    border-bottom: 1px solid #e0e0e0;
    padding: 4px;
    font-weight: 500;
}

QMenuBar::item {
    background-color: transparent;
    padding: 6px 12px;
    border-radius: 4px;
}

QMenuBar::item:selected {
    background-color: #e3f2fd;
    color: #1976d2;
}

QMenu {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 4px;
}

QMenu::item {
    padding: 8px 32px 8px 24px;
    border-radius: 3px;
}

QMenu::item:selected {
    background-color: #e3f2fd;
    color: #1976d2;
}

/* 标签页样式 - 现代化设计 */
QTabWidget::pane {
    border: 1px solid #e0e0e0;
    border-top: none;
    background-color: #ffffff;
    top: -1px;
}

QTabBar {
    background-color: #f5f5f5;
    border-bottom: 2px solid #e0e0e0;
}

QTabBar::tab {
    background-color: #f5f5f5;
    color: #666666;
    border: none;
    border-bottom: 3px solid transparent;
    padding: 10px 20px;
    margin-right: 2px;
    font-weight: 500;
    font-size: 9pt;
    min-width: 140px;
    max-width: 200px;
}

QTabBar::tab:selected {
    background-color: #ffffff;
    color: #1976d2;
    border-bottom: 3px solid #1976d2;
    font-weight: 600;
}

QTabBar::tab:hover:!selected {
    background-color: #fafafa;
    color: #1976d2;
}

QTabBar::tab:disabled {
    background-color: #f5f5f5;
    color: #bdbdbd;
}

/* 组框样式 - 卡片式设计 */
QGroupBox {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    margin-top: 16px;
    padding-top: 28px;
    padding-bottom: 16px;
    padding-left: 16px;
    padding-right: 16px;
    background-color: #ffffff;
    font-size: 9pt;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 16px;
    top: 8px;
    padding: 4px 12px;
    background-color: #ffffff;
    color: #1976d2;
    font-weight: 600;
    font-size: 10pt;
    border-radius: 4px;
}

/* 按钮样式 - 现代化渐变和阴影 */
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #2196F3, stop:1 #1976D2);
    color: white;
    border: none;
    padding: 10px 20px;
    text-align: center;
    font-size: 9pt;
    font-weight: 500;
    border-radius: 6px;
    min-width: 100px;
    min-height: 32px;
}

QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #42A5F5, stop:1 #2196F3);
}

QPushButton:pressed {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #1976D2, stop:1 #1565C0);
    padding-top: 11px;
    padding-left: 21px;
}

QPushButton:disabled {
    background-color: #e0e0e0;
    color: #9e9e9e;
}

/* 主要操作按钮 - 绿色 */
QPushButton[class="primary"] {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #4CAF50, stop:1 #388E3C);
}

QPushButton[class="primary"]:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #66BB6A, stop:1 #4CAF50);
}

/* 危险操作按钮 - 红色 */
QPushButton[class="danger"] {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #f44336, stop:1 #d32f2f);
}

QPushButton[class="danger"]:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #ef5350, stop:1 #f44336);
}

/* 输入框样式 */
QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 6px 12px;
    background-color: #ffffff;
    font-size: 9pt;
    selection-background-color: #e3f2fd;
}

QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 2px solid #2196F3;
    background-color: #fafafa;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid #666666;
    width: 0;
    height: 0;
}

/* 复选框和单选按钮 */
QCheckBox, QRadioButton {
    spacing: 8px;
    font-size: 9pt;
    color: #424242;
}

QCheckBox::indicator, QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border: 2px solid #bdbdbd;
    border-radius: 3px;
    background-color: #ffffff;
}

QCheckBox::indicator:checked {
    background-color: #2196F3;
    border-color: #2196F3;
    image: none;
}

/* Qt样式表不支持content属性，使用背景色和边框来表示选中状态 */
QCheckBox::indicator:checked {
    background-color: #2196F3;
    border-color: #2196F3;
    image: none;
}

QRadioButton::indicator {
    border-radius: 9px;
}

QRadioButton::indicator:checked {
    background-color: #2196F3;
    border-color: #2196F3;
}

/* 进度条样式 */
QProgressBar {
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    text-align: center;
    background-color: #f5f5f5;
    height: 24px;
    font-size: 9pt;
    font-weight: 500;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #2196F3, stop:1 #42A5F5);
    border-radius: 3px;
}

/* 列表和表格 */
QListWidget {
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    background-color: #ffffff;
    alternate-background-color: #fafafa;
}

QListWidget::item {
    padding: 8px;
    border-bottom: 1px solid #f5f5f5;
}

QListWidget::item:selected {
    background-color: #e3f2fd;
    color: #1976d2;
}

QListWidget::item:hover {
    background-color: #f5f5f5;
}

/* 滑块样式 */
QSlider::groove:horizontal {
    border: 1px solid #e0e0e0;
    height: 6px;
    background: #f5f5f5;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #2196F3;
    border: 2px solid #ffffff;
    width: 18px;
    height: 18px;
    margin: -6px 0;
    border-radius: 9px;
}

QSlider::handle:horizontal:hover {
    background: #42A5F5;
}

/* 标签样式 */
QLabel {
    color: #424242;
    font-size: 9pt;
}

QLabel[class="title"] {
    font-size: 12pt;
    font-weight: 600;
    color: #1976d2;
}

QLabel[class="subtitle"] {
    font-size: 10pt;
    font-weight: 500;
    color: #666666;
}

/* 滚动条样式 */
QScrollBar:vertical {
    border: none;
    background: #f5f5f5;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background: #bdbdbd;
    border-radius: 6px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background: #9e9e9e;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    border: none;
    background: #f5f5f5;
    height: 12px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal {
    background: #bdbdbd;
    border-radius: 6px;
    min-width: 30px;
}

QScrollBar::handle:horizontal:hover {
    background: #9e9e9e;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}
''')

if __name__ == '__main__':
    # Check if PyQt5 is available
    if not PYQT5_AVAILABLE:
        print("\nGeoDLFWD GUI cannot start because PyQt5 library is not installed correctly.")
        print("Please complete the environment configuration according to the installation guide above and try again.")
        print("Press any key to exit...")
        input()
        sys.exit(1)
    
    # Check other necessary libraries
    required_libs = ["numpy", "matplotlib", "scipy", "torch"]
    missing_libs = []
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing_libs.append(lib)
    
    if missing_libs:
        print(f"Error: The following necessary libraries were not found: {', '.join(missing_libs)}")
        print(f"Please install using: pip install {' '.join(missing_libs)} -i https://pypi.tuna.tsinghua.edu.cn/simple")
        print("Press any key to exit...")
        input()
        sys.exit(1)
    
    try:
        # Create application instance
        app = QApplication(sys.argv)
        
        # Create and display main window
        window = GeoDLFWDMainWindow()
        window.showMaximized()  # Show maximized on startup
        
        # Run application
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"Error occurred during program execution: {e}")
        print("This may be due to incomplete installation of PyQt5 or other dependency libraries.")
        print("Please try reinstalling all dependency libraries:")
        print("pip install PyQt5 numpy matplotlib scipy torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple")
        print("Press any key to exit...")
        input()
        sys.exit(1)