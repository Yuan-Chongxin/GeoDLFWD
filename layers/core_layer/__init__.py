# 核心功能层模块
"""
核心功能层 (Core Layer)
集成了数据驱动DL训练框架、网络结构适配模块及正演结果解析模块
"""

from .training_framework import DataDrivenDLTrainingFramework
from .network_adapter import NetworkAdapterModule
from .forward_result_parser import ForwardResultParserModule

__all__ = [
    'DataDrivenDLTrainingFramework',
    'NetworkAdapterModule',
    'ForwardResultParserModule'
]
