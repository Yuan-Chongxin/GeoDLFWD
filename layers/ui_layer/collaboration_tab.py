#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
共享协同标签页
提供GitHub上传/下载、数据共享、模型共享功能
"""

import os
import sys
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QTextEdit, QGroupBox, QFileDialog, 
                             QMessageBox, QListWidget, QListWidgetItem, 
                             QProgressBar, QLineEdit, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDateTime
from pathlib import Path

# 添加父目录到路径以导入github_adapter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from layers.collaboration_layer.github_adapter import GitHubAdapterModule


class UploadThread(QThread):
    """上传线程"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, adapter, file_path, file_type, commit_message):
        super().__init__()
        self.adapter = adapter
        self.file_path = file_path
        self.file_type = file_type
        self.commit_message = commit_message
    
    def run(self):
        try:
            self.progress.emit(f"开始上传 {self.file_type}...")
            
            # 根据文件类型选择上传路径
            if self.file_type == "训练数据":
                target_path = "data/train_data/"
            elif self.file_type == "验证数据":
                target_path = "data/val_data/"
            elif self.file_type == "训练模型":
                target_path = "models/"
            elif self.file_type == "预测结果":
                target_path = "results/"
            else:
                target_path = "shared/"
            
            # 执行上传
            success, message = self.adapter.upload_file(
                self.file_path, 
                target_path,
                self.commit_message
            )
            
            if success:
                self.finished.emit(True, f"上传成功: {message}")
            else:
                self.finished.emit(False, f"上传失败: {message}")
                
        except Exception as e:
            self.finished.emit(False, f"上传出错: {str(e)}")


class DownloadThread(QThread):
    """下载线程"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, adapter, file_path, local_path):
        super().__init__()
        self.adapter = adapter
        self.file_path = file_path
        self.local_path = local_path
    
    def run(self):
        try:
            self.progress.emit(f"开始下载 {self.file_path}...")
            success, message = self.adapter.download_file(
                self.file_path,
                self.local_path
            )
            
            if success:
                self.finished.emit(True, f"下载成功: {message}")
            else:
                self.finished.emit(False, f"下载失败: {message}")
                
        except Exception as e:
            self.finished.emit(False, f"下载出错: {str(e)}")


class CollaborationTab(QWidget):
    """共享协同标签页"""
    
    def __init__(self, parent=None):
        super(CollaborationTab, self).__init__(parent)
        self.parent = parent
        self.github_adapter = None
        self.upload_thread = None
        self.download_thread = None
        self.init_ui()
        self.init_github_adapter()
    
    def init_ui(self):
        """初始化UI界面"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 1. GitHub连接配置区域
        config_group = QGroupBox("GitHub 连接配置")
        config_layout = QVBoxLayout()
        
        # 仓库信息
        repo_layout = QHBoxLayout()
        repo_layout.addWidget(QLabel("仓库所有者:"))
        self.repo_owner_edit = QLineEdit("Yuan-Chongxin")
        repo_layout.addWidget(self.repo_owner_edit)
        repo_layout.addWidget(QLabel("仓库名称:"))
        self.repo_name_edit = QLineEdit("GeoDLFWD")
        repo_layout.addWidget(self.repo_name_edit)
        config_layout.addLayout(repo_layout)
        
        # Token输入
        token_layout = QHBoxLayout()
        token_layout.addWidget(QLabel("GitHub Token:"))
        self.token_edit = QLineEdit()
        self.token_edit.setEchoMode(QLineEdit.Password)
        self.token_edit.setPlaceholderText("输入GitHub Personal Access Token")
        token_layout.addWidget(self.token_edit)
        connect_btn = QPushButton("连接GitHub")
        connect_btn.clicked.connect(self.connect_github)
        token_layout.addWidget(connect_btn)
        config_layout.addLayout(token_layout)
        
        # 连接状态
        self.connection_status = QLabel("未连接")
        self.connection_status.setStyleSheet("color: gray;")
        config_layout.addWidget(self.connection_status)
        
        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)
        
        # 2. 上传功能区域
        upload_group = QGroupBox("上传到GitHub")
        upload_layout = QVBoxLayout()
        
        # 上传类型选择
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("上传类型:"))
        self.upload_type_combo = QComboBox()
        self.upload_type_combo.addItems([
            "训练数据", "验证数据", "训练模型", "预测结果", "其他文件"
        ])
        type_layout.addWidget(self.upload_type_combo)
        upload_layout.addLayout(type_layout)
        
        # 文件选择
        file_layout = QHBoxLayout()
        self.upload_file_label = QLabel("未选择文件")
        file_layout.addWidget(self.upload_file_label)
        select_file_btn = QPushButton("选择文件/文件夹")
        select_file_btn.clicked.connect(self.select_upload_file)
        file_layout.addWidget(select_file_btn)
        upload_layout.addLayout(file_layout)
        
        # 提交信息
        commit_layout = QHBoxLayout()
        commit_layout.addWidget(QLabel("提交信息:"))
        self.commit_message_edit = QLineEdit()
        self.commit_message_edit.setPlaceholderText("输入提交说明（可选）")
        commit_layout.addWidget(self.commit_message_edit)
        upload_layout.addLayout(commit_layout)
        
        # 上传按钮
        upload_btn = QPushButton("上传到GitHub")
        upload_btn.clicked.connect(self.upload_to_github)
        upload_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        upload_layout.addWidget(upload_btn)
        
        upload_group.setLayout(upload_layout)
        main_layout.addWidget(upload_group)
        
        # 3. 下载功能区域
        download_group = QGroupBox("从GitHub下载")
        download_layout = QVBoxLayout()
        
        # 下载文件列表
        download_layout.addWidget(QLabel("可用资源:"))
        self.download_list = QListWidget()
        self.download_list.setMaximumHeight(150)
        download_layout.addWidget(self.download_list)
        
        # 刷新列表按钮
        refresh_btn = QPushButton("刷新资源列表")
        refresh_btn.clicked.connect(self.refresh_download_list)
        download_layout.addWidget(refresh_btn)
        
        # 下载按钮
        download_btn = QPushButton("下载选中资源")
        download_btn.clicked.connect(self.download_from_github)
        download_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 8px;")
        download_layout.addWidget(download_btn)
        
        download_group.setLayout(download_layout)
        main_layout.addWidget(download_group)
        
        # 4. 操作日志
        log_group = QGroupBox("操作日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        # 设置布局权重
        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 0)
        main_layout.setStretch(2, 0)
        main_layout.setStretch(3, 1)
    
    def init_github_adapter(self):
        """初始化GitHub适配器"""
        try:
            self.github_adapter = GitHubAdapterModule(
                repo_owner=self.repo_owner_edit.text(),
                repo_name=self.repo_name_edit.text()
            )
            self.log("GitHub适配器初始化成功")
        except Exception as e:
            self.log(f"GitHub适配器初始化失败: {str(e)}", error=True)
    
    def connect_github(self):
        """连接GitHub"""
        try:
            repo_owner = self.repo_owner_edit.text()
            repo_name = self.repo_name_edit.text()
            token = self.token_edit.text()
            
            if not token:
                QMessageBox.warning(self, "Warning", "Please enter GitHub Token")
                return
            
            # 重新初始化适配器
            self.github_adapter = GitHubAdapterModule(
                repo_owner=repo_owner,
                repo_name=repo_name
            )
            self.github_adapter.set_token(token)
            
            # 检查连接
            if self.github_adapter.check_git_available():
                self.connection_status.setText("已连接")
                self.connection_status.setStyleSheet("color: green;")
                self.log("GitHub连接成功")
                
                # 初始化仓库
                success, message = self.github_adapter.init_repository()
                self.log(message)
                
                # 设置远程仓库
                if success:
                    self.github_adapter.set_remote("origin", 
                        f"https://github.com/{repo_owner}/{repo_name}.git")
            else:
                self.connection_status.setText("Git未安装")
                self.connection_status.setStyleSheet("color: red;")
                self.log("Git未安装或未找到", error=True)
                QMessageBox.warning(
                    self, "Git Not Found",
                    "Git was not detected. Please install Git and ensure it is in your PATH.\n\n"
                    "Download: https://git-scm.com/download/win\n\n"
                    "After installing, restart this application."
                )
                
        except Exception as e:
            self.log(f"连接失败: {str(e)}", error=True)
            QMessageBox.critical(self, "Error", f"Connection failed: {str(e)}")
    
    def select_upload_file(self):
        """Select file to upload"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select File to Upload",
            "",
            "All Files (*.*)"
        )
        
        if file_path:
            self.upload_file_path = file_path
            self.upload_file_label.setText(os.path.basename(file_path))
            self.log(f"已选择文件: {os.path.basename(file_path)}")
    
    def upload_to_github(self):
        """Upload to GitHub"""
        if not self.github_adapter:
            QMessageBox.warning(self, "Warning", "Please connect to GitHub first")
            return
        
        if not hasattr(self, 'upload_file_path') or not self.upload_file_path:
            QMessageBox.warning(self, "Warning", "Please select a file to upload first")
            return
        
        file_type = self.upload_type_combo.currentText()
        commit_message = self.commit_message_edit.text() or f"上传{file_type}"
        
        # 创建上传线程
        self.upload_thread = UploadThread(
            self.github_adapter,
            self.upload_file_path,
            file_type,
            commit_message
        )
        self.upload_thread.progress.connect(self.log)
        self.upload_thread.finished.connect(self.on_upload_finished)
        self.upload_thread.start()
    
    def on_upload_finished(self, success, message):
        """Upload finished callback"""
        self.log(message, error=not success)
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.warning(self, "Failed", message)
    
    def refresh_download_list(self):
        """Refresh download list"""
        if not self.github_adapter:
            QMessageBox.warning(self, "Warning", "Please connect to GitHub first")
            return
        
        self.download_list.clear()
        self.log("正在获取资源列表...")
        
        # 这里应该调用GitHub API获取文件列表
        # 暂时使用示例数据
        items = [
            "data/train_data/model_001.txt",
            "data/train_data/model_002.txt",
            "models/unet_model_epoch_100.pth",
            "results/prediction_result_001.png"
        ]
        
        for item in items:
            self.download_list.addItem(QListWidgetItem(item))
        
        self.log(f"找到 {len(items)} 个可用资源")
    
    def download_from_github(self):
        """Download from GitHub"""
        if not self.github_adapter:
            QMessageBox.warning(self, "Warning", "Please connect to GitHub first")
            return
        
        selected_items = self.download_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select a resource to download first")
            return
        
        # Select save location
        save_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Save Location",
            ""
        )
        
        if not save_dir:
            return
        
        for item in selected_items:
            file_path = item.text()
            local_path = os.path.join(save_dir, os.path.basename(file_path))
            
            # 创建下载线程
            self.download_thread = DownloadThread(
                self.github_adapter,
                file_path,
                local_path
            )
            self.download_thread.progress.connect(self.log)
            self.download_thread.finished.connect(self.on_download_finished)
            self.download_thread.start()
    
    def on_download_finished(self, success, message):
        """Download finished callback"""
        self.log(message, error=not success)
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.warning(self, "Failed", message)
    
    def log(self, message, error=False):
        """记录日志"""
        color = "red" if error else "black"
        timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
        self.log_text.append(f'<span style="color: {color}">[{timestamp}] {message}</span>')
