#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
共享数据交互模块
实现与平台共享资源的对接，支持上传本地训练数据、优化后模型至共享平台，
同时可下载平台上的共享资源用于本地训练
"""

import os
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path

# 导入GitHub适配器
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from layers.collaboration_layer.github_adapter import GitHubAdapterModule

class SharedDataInteractionModule:
    """
    共享数据交互模块
    负责与GitHub共享平台的数据交互
    """
    
    def __init__(self, github_adapter: Optional[GitHubAdapterModule] = None):
        """
        初始化共享数据交互模块
        
        Args:
            github_adapter: GitHub适配器实例（可选）
        """
        self.github_adapter = github_adapter
        self.upload_callbacks = []  # 上传进度回调
        self.download_callbacks = []  # 下载进度回调
    
    def set_github_adapter(self, github_adapter: GitHubAdapterModule):
        """
        设置GitHub适配器
        
        Args:
            github_adapter: GitHub适配器实例
        """
        self.github_adapter = github_adapter
    
    def upload_training_data(self,
                           data_path: str,
                           target_path: Optional[str] = None,
                           commit_message: Optional[str] = None,
                           progress_callback: Optional[Callable[[str], None]] = None) -> Tuple[bool, str]:
        """
        上传训练数据到共享平台
        
        Args:
            data_path: 本地训练数据路径
            target_path: 目标路径（相对于仓库根目录）
            commit_message: 提交信息
            progress_callback: 进度回调函数
            
        Returns:
            (success, message)
        """
        if not self.github_adapter:
            return False, "GitHub适配器未设置"
        
        if progress_callback:
            progress_callback(f"开始上传训练数据: {os.path.basename(data_path)}")
        
        if target_path is None:
            target_path = f"data/training_data/{os.path.basename(data_path)}"
        
        if commit_message is None:
            commit_message = f"Upload training data: {os.path.basename(data_path)}"
        
        success, message = self.github_adapter.upload_file(
            data_path,
            target_path,
            commit_message
        )
        
        if progress_callback:
            if success:
                progress_callback(f"训练数据上传成功: {target_path}")
            else:
                progress_callback(f"训练数据上传失败: {message}")
        
        return success, message
    
    def upload_model(self,
                    model_path: str,
                    target_path: Optional[str] = None,
                    commit_message: Optional[str] = None,
                    progress_callback: Optional[Callable[[str], None]] = None) -> Tuple[bool, str]:
        """
        上传训练模型到共享平台
        
        Args:
            model_path: 本地模型文件路径
            target_path: 目标路径（相对于仓库根目录）
            commit_message: 提交信息
            progress_callback: 进度回调函数
            
        Returns:
            (success, message)
        """
        if not self.github_adapter:
            return False, "GitHub适配器未设置"
        
        if progress_callback:
            progress_callback(f"开始上传模型: {os.path.basename(model_path)}")
        
        if target_path is None:
            target_path = f"models/{os.path.basename(model_path)}"
        
        if commit_message is None:
            commit_message = f"Upload trained model: {os.path.basename(model_path)}"
        
        success, message = self.github_adapter.upload_file(
            model_path,
            target_path,
            commit_message
        )
        
        if progress_callback:
            if success:
                progress_callback(f"模型上传成功: {target_path}")
            else:
                progress_callback(f"模型上传失败: {message}")
        
        return success, message
    
    def download_training_data(self,
                              resource_path: str,
                              local_path: str,
                              progress_callback: Optional[Callable[[str], None]] = None) -> Tuple[bool, str]:
        """
        从共享平台下载训练数据
        
        Args:
            resource_path: 共享平台中的资源路径
            local_path: 本地保存路径
            progress_callback: 进度回调函数
            
        Returns:
            (success, message)
        """
        if not self.github_adapter:
            return False, "GitHub适配器未设置"
        
        if progress_callback:
            progress_callback(f"开始下载训练数据: {resource_path}")
        
        success, message = self.github_adapter.download_file(
            resource_path,
            local_path
        )
        
        if progress_callback:
            if success:
                progress_callback(f"训练数据下载成功: {local_path}")
            else:
                progress_callback(f"训练数据下载失败: {message}")
        
        return success, message
    
    def download_model(self,
                      resource_path: str,
                      local_path: str,
                      progress_callback: Optional[Callable[[str], None]] = None) -> Tuple[bool, str]:
        """
        从共享平台下载模型
        
        Args:
            resource_path: 共享平台中的资源路径
            local_path: 本地保存路径
            progress_callback: 进度回调函数
            
        Returns:
            (success, message)
        """
        if not self.github_adapter:
            return False, "GitHub适配器未设置"
        
        if progress_callback:
            progress_callback(f"开始下载模型: {resource_path}")
        
        success, message = self.github_adapter.download_file(
            resource_path,
            local_path
        )
        
        if progress_callback:
            if success:
                progress_callback(f"模型下载成功: {local_path}")
            else:
                progress_callback(f"模型下载失败: {message}")
        
        return success, message
    
    def list_shared_resources(self,
                             resource_type: Optional[str] = None) -> List[Dict]:
        """
        列出共享平台上的可用资源
        
        Args:
            resource_type: 资源类型（training_data, model等），None表示所有类型
            
        Returns:
            资源列表，每个元素包含路径、类型等信息
        """
        # 注意：实际实现需要调用GitHub API获取文件列表
        # 这里只是提供接口定义
        resources = []
        
        if not self.github_adapter:
            return resources
        
        # TODO: 实现GitHub API调用获取文件列表
        # 可以使用GitHub API的Contents API来获取仓库文件列表
        
        return resources
    
    def search_shared_resources(self,
                               keyword: str,
                               resource_type: Optional[str] = None) -> List[Dict]:
        """
        搜索共享平台上的资源
        
        Args:
            keyword: 搜索关键词
            resource_type: 资源类型过滤
            
        Returns:
            匹配的资源列表
        """
        all_resources = self.list_shared_resources(resource_type)
        
        # 简单关键词匹配
        matched = []
        keyword_lower = keyword.lower()
        
        for resource in all_resources:
            resource_name = resource.get('name', '').lower()
            if keyword_lower in resource_name:
                matched.append(resource)
        
        return matched
    
    def register_upload_callback(self, callback: Callable[[str], None]):
        """
        注册上传进度回调函数
        
        Args:
            callback: 回调函数
        """
        if callback not in self.upload_callbacks:
            self.upload_callbacks.append(callback)
    
    def register_download_callback(self, callback: Callable[[str], None]):
        """
        注册下载进度回调函数
        
        Args:
            callback: 回调函数
        """
        if callback not in self.download_callbacks:
            self.download_callbacks.append(callback)
