# 共享协同层模块
"""
共享协同层 (Collaboration Layer)
实现"全球协同训练、模型持续迭代"核心目标，提供GitHub集成适配、版本同步及权限管理
"""

from .github_adapter import GitHubAdapterModule
from .version_sync import VersionSyncModule
from .permission_manager import PermissionManagerModule

__all__ = [
    'GitHubAdapterModule',
    'VersionSyncModule',
    'PermissionManagerModule'
]
