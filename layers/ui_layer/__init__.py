# 用户交互层模块
"""
用户交互层 (UI Layer)
提供直观友好的操作入口，包括参数配置、训练任务管理、结果可视化及共享资源检索
"""

# 使用try-except来处理可能不存在的模块
try:
    from .parameter_config import ParameterConfigModule
except ImportError:
    ParameterConfigModule = None

try:
    from .task_manager import TaskManagerModule
except ImportError:
    TaskManagerModule = None

try:
    from .visualization import VisualizationModule
except ImportError:
    VisualizationModule = None

try:
    from .resource_search import ResourceSearchModule
except ImportError:
    ResourceSearchModule = None

# 导入实际存在的模块
from .collaboration_tab import CollaborationTab
from .upload_dialog import UploadDialog

__all__ = [
    'ParameterConfigModule',
    'TaskManagerModule',
    'VisualizationModule',
    'ResourceSearchModule',
    'CollaborationTab',
    'UploadDialog'
]
