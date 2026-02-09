#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
权限管理模块
管理GitHub仓库中共享资源的公开/私有属性设置和访问权限分级管控
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from enum import Enum

class ResourceVisibility(Enum):
    """资源可见性枚举"""
    PUBLIC = "public"  # 公开资源
    PRIVATE = "private"  # 私有资源

class AccessLevel(Enum):
    """访问权限级别枚举"""
    READ_ONLY = "read_only"  # 只读
    READ_WRITE = "read_write"  # 读写
    ADMIN = "admin"  # 管理员

class PermissionManagerModule:
    """
    权限管理模块
    负责管理共享资源的权限设置
    """
    
    def __init__(self, 
                 repo_path: str,
                 permission_config_file: str = ".permissions.json"):
        """
        初始化权限管理模块
        
        Args:
            repo_path: 本地仓库路径
            permission_config_file: 权限配置文件路径
        """
        self.repo_path = Path(repo_path)
        self.permission_config_file = self.repo_path / permission_config_file
        self.resource_permissions = {}  # 资源权限配置
        self.user_permissions = {}  # 用户权限配置
        self.load_config()
    
    def load_config(self):
        """加载权限配置"""
        if self.permission_config_file.exists():
            try:
                with open(self.permission_config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.resource_permissions = config.get('resource_permissions', {})
                    self.user_permissions = config.get('user_permissions', {})
            except Exception as e:
                print(f"加载权限配置失败: {e}")
                self.resource_permissions = {}
                self.user_permissions = {}
        else:
            self.resource_permissions = {}
            self.user_permissions = {}
    
    def save_config(self):
        """保存权限配置"""
        try:
            config = {
                'resource_permissions': self.resource_permissions,
                'user_permissions': self.user_permissions
            }
            with open(self.permission_config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存权限配置失败: {e}")
    
    def set_resource_visibility(self, 
                               resource_path: str,
                               visibility: ResourceVisibility):
        """
        设置资源可见性
        
        Args:
            resource_path: 资源路径（相对于仓库根目录）
            visibility: 可见性（公开/私有）
        """
        if resource_path not in self.resource_permissions:
            self.resource_permissions[resource_path] = {}
        
        self.resource_permissions[resource_path]['visibility'] = visibility.value
        self.save_config()
    
    def get_resource_visibility(self, resource_path: str) -> ResourceVisibility:
        """
        获取资源可见性
        
        Args:
            resource_path: 资源路径
            
        Returns:
            资源可见性
        """
        if resource_path in self.resource_permissions:
            visibility_str = self.resource_permissions[resource_path].get('visibility', 'public')
            return ResourceVisibility(visibility_str)
        return ResourceVisibility.PUBLIC  # 默认公开
    
    def set_user_access_level(self,
                             resource_path: str,
                             user: str,
                             access_level: AccessLevel):
        """
        设置用户对资源的访问权限级别
        
        Args:
            resource_path: 资源路径
            user: 用户名（GitHub用户名）
            access_level: 访问权限级别
        """
        if resource_path not in self.resource_permissions:
            self.resource_permissions[resource_path] = {}
        
        if 'user_access' not in self.resource_permissions[resource_path]:
            self.resource_permissions[resource_path]['user_access'] = {}
        
        self.resource_permissions[resource_path]['user_access'][user] = access_level.value
        self.save_config()
    
    def get_user_access_level(self,
                              resource_path: str,
                              user: str) -> AccessLevel:
        """
        获取用户对资源的访问权限级别
        
        Args:
            resource_path: 资源路径
            user: 用户名
            
        Returns:
            访问权限级别
        """
        if resource_path in self.resource_permissions:
            user_access = self.resource_permissions[resource_path].get('user_access', {})
            if user in user_access:
                return AccessLevel(user_access[user])
        
        # 根据资源可见性返回默认权限
        visibility = self.get_resource_visibility(resource_path)
        if visibility == ResourceVisibility.PUBLIC:
            return AccessLevel.READ_ONLY
        else:
            return AccessLevel.READ_ONLY  # 私有资源默认只读，需要明确授权
    
    def check_access_permission(self,
                                resource_path: str,
                                user: str,
                                required_level: AccessLevel) -> bool:
        """
        检查用户是否有足够的访问权限
        
        Args:
            resource_path: 资源路径
            user: 用户名
            required_level: 所需权限级别
            
        Returns:
            是否有权限
        """
        user_level = self.get_user_access_level(resource_path, user)
        
        # 权限级别比较
        level_hierarchy = {
            AccessLevel.READ_ONLY: 1,
            AccessLevel.READ_WRITE: 2,
            AccessLevel.ADMIN: 3
        }
        
        return level_hierarchy[user_level] >= level_hierarchy[required_level]
    
    def list_accessible_resources(self, user: str) -> List[str]:
        """
        列出用户可访问的资源列表
        
        Args:
            user: 用户名
            
        Returns:
            可访问的资源路径列表
        """
        accessible = []
        
        for resource_path, permissions in self.resource_permissions.items():
            visibility = permissions.get('visibility', 'public')
            
            # 公开资源所有人都可以访问
            if visibility == 'public':
                accessible.append(resource_path)
            else:
                # 私有资源需要检查用户权限
                user_access = permissions.get('user_access', {})
                if user in user_access:
                    accessible.append(resource_path)
        
        return accessible
    
    def remove_user_access(self, resource_path: str, user: str):
        """
        移除用户对资源的访问权限
        
        Args:
            resource_path: 资源路径
            user: 用户名
        """
        if resource_path in self.resource_permissions:
            user_access = self.resource_permissions[resource_path].get('user_access', {})
            if user in user_access:
                del user_access[user]
                self.save_config()
    
    def get_resource_permissions(self, resource_path: str) -> Dict:
        """
        获取资源的所有权限配置
        
        Args:
            resource_path: 资源路径
            
        Returns:
            权限配置字典
        """
        return self.resource_permissions.get(resource_path, {}).copy()
    
    def set_default_visibility(self, visibility: ResourceVisibility):
        """
        设置默认资源可见性
        
        Args:
            visibility: 默认可见性
        """
        if 'defaults' not in self.resource_permissions:
            self.resource_permissions['defaults'] = {}
        self.resource_permissions['defaults']['visibility'] = visibility.value
        self.save_config()
