# 数据服务层模块
"""
数据服务层 (Data Service Layer)
负责数据的存储、管理与预处理，涵盖本地数据存储模块与共享数据交互模块
"""

from .local_storage import LocalStorageModule
from .shared_data_interaction import SharedDataInteractionModule

__all__ = [
    'LocalStorageModule',
    'SharedDataInteractionModule'
]
