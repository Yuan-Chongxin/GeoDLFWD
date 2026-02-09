#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
版本同步模块
跟踪GitHub仓库中共享资源的版本更新，及时向用户推送更新提醒
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path

class VersionSyncModule:
    """
    版本同步模块
    负责跟踪GitHub仓库中共享资源的版本更新
    """
    
    def __init__(self, 
                 repo_path: str,
                 sync_config_file: str = ".version_sync.json"):
        """
        初始化版本同步模块
        
        Args:
            repo_path: 本地仓库路径
            sync_config_file: 版本同步配置文件路径
        """
        self.repo_path = Path(repo_path)
        self.sync_config_file = self.repo_path / sync_config_file
        self.tracked_resources = {}  # 跟踪的资源列表
        self.update_callbacks = []  # 更新提醒回调函数列表
        self.load_config()
    
    def load_config(self):
        """加载版本同步配置"""
        if self.sync_config_file.exists():
            try:
                with open(self.sync_config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.tracked_resources = config.get('tracked_resources', {})
            except Exception as e:
                print(f"加载版本同步配置失败: {e}")
                self.tracked_resources = {}
        else:
            self.tracked_resources = {}
    
    def save_config(self):
        """保存版本同步配置"""
        try:
            config = {
                'tracked_resources': self.tracked_resources,
                'last_update': datetime.now().isoformat()
            }
            with open(self.sync_config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存版本同步配置失败: {e}")
    
    def track_resource(self, 
                      resource_path: str,
                      resource_type: str = "model",
                      description: str = ""):
        """
        开始跟踪资源
        
        Args:
            resource_path: 资源路径（相对于仓库根目录）
            resource_type: 资源类型（model, data, result等）
            description: 资源描述
        """
        resource_key = resource_path
        
        # 获取资源当前版本信息
        full_path = self.repo_path / resource_path
        if full_path.exists():
            mtime = os.path.getmtime(full_path)
            size = os.path.getsize(full_path) if full_path.is_file() else 0
            version_info = {
                'path': resource_path,
                'type': resource_type,
                'description': description,
                'last_modified': datetime.fromtimestamp(mtime).isoformat(),
                'size': size,
                'version': f"{mtime}_{size}"  # 简单版本标识
            }
        else:
            version_info = {
                'path': resource_path,
                'type': resource_type,
                'description': description,
                'last_modified': None,
                'size': 0,
                'version': None
            }
        
        self.tracked_resources[resource_key] = version_info
        self.save_config()
    
    def untrack_resource(self, resource_path: str):
        """
        停止跟踪资源
        
        Args:
            resource_path: 资源路径
        """
        resource_key = resource_path
        if resource_key in self.tracked_resources:
            del self.tracked_resources[resource_key]
            self.save_config()
    
    def check_updates(self) -> List[Dict]:
        """
        检查资源更新
        
        Returns:
            更新资源列表，每个元素包含资源信息和更新状态
        """
        updates = []
        
        for resource_key, version_info in self.tracked_resources.items():
            resource_path = version_info['path']
            full_path = self.repo_path / resource_path
            
            if full_path.exists():
                # 获取当前版本信息
                mtime = os.path.getmtime(full_path)
                size = os.path.getsize(full_path) if full_path.is_file() else 0
                current_version = f"{mtime}_{size}"
                
                # 比较版本
                if version_info['version'] != current_version:
                    # 有更新
                    old_version = version_info['version']
                    version_info['version'] = current_version
                    version_info['last_modified'] = datetime.fromtimestamp(mtime).isoformat()
                    version_info['size'] = size
                    
                    updates.append({
                        'resource_path': resource_path,
                        'resource_type': version_info['type'],
                        'description': version_info.get('description', ''),
                        'old_version': old_version,
                        'new_version': current_version,
                        'last_modified': version_info['last_modified'],
                        'has_update': True
                    })
            else:
                # 资源不存在
                if version_info['version'] is not None:
                    updates.append({
                        'resource_path': resource_path,
                        'resource_type': version_info['type'],
                        'description': version_info.get('description', ''),
                        'old_version': version_info['version'],
                        'new_version': None,
                        'last_modified': None,
                        'has_update': False,
                        'removed': True
                    })
        
        # 保存更新后的配置
        if updates:
            self.save_config()
        
        return updates
    
    def register_update_callback(self, callback: Callable[[Dict], None]):
        """
        注册更新提醒回调函数
        
        Args:
            callback: 回调函数，接收更新信息字典作为参数
        """
        if callback not in self.update_callbacks:
            self.update_callbacks.append(callback)
    
    def unregister_update_callback(self, callback: Callable[[Dict], None]):
        """
        取消注册更新提醒回调函数
        
        Args:
            callback: 回调函数
        """
        if callback in self.update_callbacks:
            self.update_callbacks.remove(callback)
    
    def notify_updates(self, updates: List[Dict]):
        """
        通知更新
        
        Args:
            updates: 更新资源列表
        """
        for update in updates:
            for callback in self.update_callbacks:
                try:
                    callback(update)
                except Exception as e:
                    print(f"执行更新回调失败: {e}")
    
    def get_tracked_resources(self) -> Dict:
        """
        获取所有跟踪的资源
        
        Returns:
            跟踪资源字典
        """
        return self.tracked_resources.copy()
    
    def get_resource_info(self, resource_path: str) -> Optional[Dict]:
        """
        获取资源信息
        
        Args:
            resource_path: 资源路径
            
        Returns:
            资源信息字典，如果不存在则返回None
        """
        resource_key = resource_path
        return self.tracked_resources.get(resource_key)
    
    def start_auto_sync(self, interval: int = 300):
        """
        启动自动同步检查（需要配合定时器使用）
        
        Args:
            interval: 检查间隔（秒），默认5分钟
        """
        # 注意：实际实现中需要使用QTimer或其他定时机制
        # 这里只是提供接口定义
        pass
