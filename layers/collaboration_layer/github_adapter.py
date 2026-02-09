#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GitHub集成适配模块
实现GitHub账户授权、仓库关联、一键上传/下载功能
"""

import os
import subprocess
import json
import shutil
from typing import Optional, Dict, List, Tuple
from pathlib import Path

class GitHubAdapterModule:
    """
    GitHub集成适配模块
    提供Git命令封装和GitHub API调用接口
    """
    
    def __init__(self, 
                 repo_owner: str = "Yuan-Chongxin",
                 repo_name: str = "GeoDLFWD",
                 local_repo_path: Optional[str] = None):
        """
        初始化GitHub适配器
        
        Args:
            repo_owner: GitHub仓库所有者
            repo_name: 仓库名称
            local_repo_path: 本地仓库路径
        """
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.local_repo_path = local_repo_path or os.getcwd()
        self.git_path = self._find_git()
        self.token = None
        self.remote_url = f"https://github.com/{repo_owner}/{repo_name}.git"
        
    def _find_git(self) -> Optional[str]:
        """Find Git executable path"""
        # 1. Try shutil.which (uses current process PATH)
        git_path = shutil.which("git")
        if git_path:
            return git_path

        # 2. Check common Windows install locations (Git for Windows default paths)
        possible_paths = [
            r"C:\Program Files\Git\cmd\git.exe",
            r"C:\Program Files\Git\bin\git.exe",
            r"C:\Program Files (x86)\Git\cmd\git.exe",
            r"C:\Program Files (x86)\Git\bin\git.exe",
            os.path.expanduser(r"~\AppData\Local\Programs\Git\cmd\git.exe"),
            os.path.expanduser(r"~\AppData\Local\Programs\Git\bin\git.exe"),
            r"C:\Program Files\Portable Git\cmd\git.exe",
            r"C:\Program Files\Portable Git\bin\git.exe",
            os.path.expanduser(r"~\scoop\apps\git\current\cmd\git.exe"),
            os.path.expanduser(r"~\scoop\shims\git.exe"),
            r"C:\ProgramData\chocolatey\bin\git.exe",
            r"C:\Program Files\TortoiseGit\bin\git.exe",
        ]
        for path in possible_paths:
            if path and os.path.exists(path):
                return path

        # 3. Windows: use PowerShell with fresh PATH from registry (GUI apps often have stale PATH)
        if os.name == "nt":
            try:
                ps_cmd = (
                    "$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + "
                    "[Environment]::GetEnvironmentVariable('Path','User'); "
                    "(Get-Command git -ErrorAction SilentlyContinue).Source"
                )
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps_cmd],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip().split("\n")[0].strip()
            except Exception:
                pass

        # 4. Fallback: 'where git' in new cmd (may have refreshed PATH)
        try:
            result = subprocess.run(
                "where git",
                capture_output=True,
                text=True,
                shell=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split("\n")[0].strip()
        except Exception:
            pass

        return None
    
    def set_token(self, token: str):
        """设置GitHub Personal Access Token"""
        self.token = token
    
    def check_git_available(self) -> bool:
        """检查Git是否可用"""
        return self.git_path is not None
    
    def init_repository(self) -> Tuple[bool, str]:
        """
        初始化Git仓库
        
        Returns:
            (success, message)
        """
        if not self.check_git_available():
            return False, "Git未安装或未找到"
        
        try:
            # 检查是否已经是Git仓库
            if os.path.exists(os.path.join(self.local_repo_path, '.git')):
                return True, "仓库已初始化"
            
            # 初始化仓库
            result = subprocess.run(
                [self.git_path, 'init'],
                cwd=self.local_repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return True, "仓库初始化成功"
            else:
                return False, f"初始化失败: {result.stderr}"
        except Exception as e:
            return False, f"初始化异常: {str(e)}"
    
    def add_remote(self, remote_name: str = 'origin') -> Tuple[bool, str]:
        """
        添加远程仓库
        
        Args:
            remote_name: 远程仓库名称
            
        Returns:
            (success, message)
        """
        if not self.check_git_available():
            return False, "Git未安装或未找到"
        
        try:
            # 检查是否已存在
            result = subprocess.run(
                [self.git_path, 'remote', 'get-url', remote_name],
                cwd=self.local_repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # 已存在，更新URL
                result = subprocess.run(
                    [self.git_path, 'remote', 'set-url', remote_name, self.remote_url],
                    cwd=self.local_repo_path,
                    capture_output=True,
                    text=True
                )
            else:
                # 不存在，添加
                result = subprocess.run(
                    [self.git_path, 'remote', 'add', remote_name, self.remote_url],
                    cwd=self.local_repo_path,
                    capture_output=True,
                    text=True
                )
            
            if result.returncode == 0:
                return True, f"远程仓库已添加/更新: {remote_name}"
            else:
                return False, f"添加远程仓库失败: {result.stderr}"
        except Exception as e:
            return False, f"添加远程仓库异常: {str(e)}"
    
    def commit_and_push(self, 
                       commit_message: str = "Update training data and models",
                       branch: str = 'main',
                       files: Optional[List[str]] = None) -> Tuple[bool, str]:
        """
        提交并推送更改到GitHub
        
        Args:
            commit_message: 提交信息
            branch: 分支名称
            files: 要提交的文件列表（None表示所有更改）
            
        Returns:
            (success, message)
        """
        if not self.check_git_available():
            return False, "Git未安装或未找到"
        
        try:
            # 添加文件
            if files:
                for file in files:
                    subprocess.run(
                        [self.git_path, 'add', file],
                        cwd=self.local_repo_path,
                        capture_output=True
                    )
            else:
                subprocess.run(
                    [self.git_path, 'add', '.'],
                    cwd=self.local_repo_path,
                    capture_output=True
                )
            
            # 提交
            result = subprocess.run(
                [self.git_path, 'commit', '-m', commit_message],
                cwd=self.local_repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0 and 'nothing to commit' not in result.stdout.lower():
                return False, f"提交失败: {result.stderr}"
            
            # 推送
            # 使用token进行身份验证
            if self.token:
                # 修改remote URL以包含token
                auth_url = self.remote_url.replace(
                    'https://',
                    f'https://{self.token}@'
                )
                subprocess.run(
                    [self.git_path, 'remote', 'set-url', 'origin', auth_url],
                    cwd=self.local_repo_path,
                    capture_output=True
                )
            
            result = subprocess.run(
                [self.git_path, 'push', '-u', 'origin', branch],
                cwd=self.local_repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return True, "推送成功"
            else:
                return False, f"推送失败: {result.stderr}"
        except Exception as e:
            return False, f"推送异常: {str(e)}"
    
    def pull_latest(self, branch: str = 'main') -> Tuple[bool, str]:
        """
        从GitHub拉取最新更改
        
        Args:
            branch: 分支名称
            
        Returns:
            (success, message)
        """
        if not self.check_git_available():
            return False, "Git未安装或未找到"
        
        try:
            result = subprocess.run(
                [self.git_path, 'pull', 'origin', branch],
                cwd=self.local_repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return True, "拉取成功"
            else:
                return False, f"拉取失败: {result.stderr}"
        except Exception as e:
            return False, f"拉取异常: {str(e)}"
    
    def upload_training_data(self, 
                            data_path: str,
                            commit_message: str = "Upload training data") -> Tuple[bool, str]:
        """
        上传训练数据到GitHub
        
        Args:
            data_path: 训练数据路径
            commit_message: 提交信息
            
        Returns:
            (success, message)
        """
        return self.commit_and_push(
            commit_message=commit_message,
            files=[data_path] if os.path.exists(data_path) else None
        )
    
    def upload_model(self,
                    model_path: str,
                    commit_message: str = "Upload trained model") -> Tuple[bool, str]:
        """
        上传训练好的模型到GitHub
        
        Args:
            model_path: 模型文件路径
            commit_message: 提交信息
            
        Returns:
            (success, message)
        """
        return self.commit_and_push(
            commit_message=commit_message,
            files=[model_path] if os.path.exists(model_path) else None
        )
    
    def download_resource(self,
                         resource_path: str,
                         local_path: Optional[str] = None) -> Tuple[bool, str]:
        """
        从GitHub下载资源
        
        Args:
            resource_path: GitHub仓库中的资源路径
            local_path: 本地保存路径
            
        Returns:
            (success, message)
        """
        # 先拉取最新更改
        success, message = self.pull_latest()
        if not success:
            return False, f"拉取失败: {message}"
        
        # 检查资源是否存在
        full_path = os.path.join(self.local_repo_path, resource_path)
        if os.path.exists(full_path):
            if local_path and local_path != full_path:
                import shutil
                shutil.copy2(full_path, local_path)
            return True, f"资源已下载: {resource_path}"
        else:
            return False, f"资源不存在: {resource_path}"
    
    def upload_file(self, file_path: str, target_path: str, commit_message: str = "Upload file") -> Tuple[bool, str]:
        """
        上传文件到GitHub
        
        Args:
            file_path: 本地文件路径
            target_path: GitHub仓库中的目标路径（相对于仓库根目录）
            commit_message: 提交信息
            
        Returns:
            (success, message)
        """
        if not os.path.exists(file_path):
            return False, f"文件不存在: {file_path}"
        
        try:
            import shutil
            # 创建目标目录
            target_full_path = os.path.normpath(os.path.join(self.local_repo_path, target_path))
            src_path = os.path.normpath(os.path.abspath(file_path))
            os.makedirs(os.path.dirname(target_full_path), exist_ok=True)
            
            # 复制文件到目标位置（若源与目标是同一文件则跳过复制）
            if os.path.isdir(file_path):
                if os.path.exists(target_full_path):
                    shutil.rmtree(target_full_path)
                shutil.copytree(file_path, target_full_path)
            else:
                try:
                    if not os.path.samefile(src_path, target_full_path):
                        shutil.copy2(file_path, target_full_path)
                except OSError:
                    shutil.copy2(file_path, target_full_path)
            
            # 提交并推送
            return self.commit_and_push(
                commit_message=commit_message,
                files=[target_path]
            )
        except Exception as e:
            return False, f"上传失败: {str(e)}"
    
    def download_file(self, file_path: str, local_path: str) -> Tuple[bool, str]:
        """
        从GitHub下载文件
        
        Args:
            file_path: GitHub仓库中的文件路径（相对于仓库根目录）
            local_path: 本地保存路径
            
        Returns:
            (success, message)
        """
        return self.download_resource(file_path, local_path)
    
    def set_remote(self, remote_name: str, remote_url: str) -> Tuple[bool, str]:
        """
        设置远程仓库URL
        
        Args:
            remote_name: 远程仓库名称
            remote_url: 远程仓库URL
            
        Returns:
            (success, message)
        """
        if not self.check_git_available():
            return False, "Git未安装或未找到"
        
        try:
            # 检查是否已存在
            result = subprocess.run(
                [self.git_path, 'remote', 'get-url', remote_name],
                cwd=self.local_repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # 已存在，更新URL
                result = subprocess.run(
                    [self.git_path, 'remote', 'set-url', remote_name, remote_url],
                    cwd=self.local_repo_path,
                    capture_output=True,
                    text=True
                )
            else:
                # 不存在，添加
                result = subprocess.run(
                    [self.git_path, 'remote', 'add', remote_name, remote_url],
                    cwd=self.local_repo_path,
                    capture_output=True,
                    text=True
                )
            
            if result.returncode == 0:
                return True, f"远程仓库已设置: {remote_name}"
            else:
                return False, f"设置远程仓库失败: {result.stderr}"
        except Exception as e:
            return False, f"设置远程仓库异常: {str(e)}"
