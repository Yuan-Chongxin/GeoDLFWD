#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Upload Dialog
Supports uploading models and data, handles large files (>25MB) using Git LFS or GitHub Releases
"""

import os
import sys
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QTextEdit, QFileDialog, QMessageBox, 
                             QLineEdit, QComboBox, QProgressBar, QGroupBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from pathlib import Path

# Add parent directory to path to import github_adapter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from layers.collaboration_layer.github_adapter import GitHubAdapterModule


class UploadThread(QThread):
    """Upload thread"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, adapter, file_path, file_type, target_path, commit_message, use_lfs=False):
        super().__init__()
        self.adapter = adapter
        self.file_path = file_path
        self.file_type = file_type
        self.target_path = target_path
        self.commit_message = commit_message
        self.use_lfs = use_lfs
    
    def run(self):
        try:
            self.progress.emit(f"Starting upload {self.file_type}...")
            
            # Check file size
            file_size = os.path.getsize(self.file_path) / (1024 * 1024)  # MB
            self.progress.emit(f"File size: {file_size:.2f} MB")
            
            # If file is larger than 25MB and LFS is not enabled, warn user
            if file_size > 25 and not self.use_lfs:
                self.progress.emit("Warning: File exceeds 25MB, recommend using Git LFS")
            
            # If LFS is enabled, check if git-lfs is installed first
            if self.use_lfs:
                if not self.check_git_lfs():
                    self.finished.emit(False, "Git LFS is not installed. Please install Git LFS first: https://git-lfs.github.com/")
                    return
                
                # Initialize Git LFS
                self.progress.emit("Initializing Git LFS...")
                if not self.init_git_lfs():
                    self.finished.emit(False, "Git LFS initialization failed")
                    return
                
                # Track large files
                self.progress.emit(f"Tracking file type: {Path(self.file_path).suffix}")
                if not self.track_file_type(Path(self.file_path).suffix):
                    self.finished.emit(False, "Failed to track file type with Git LFS")
                    return
            
            # If using LFS, need to copy file to target location first, then use git add
            if self.use_lfs:
                import subprocess
                import shutil
                
                # Create target directory
                target_full_path = os.path.normpath(os.path.join(self.adapter.local_repo_path, self.target_path))
                src_path = os.path.normpath(os.path.abspath(self.file_path))
                os.makedirs(os.path.dirname(target_full_path), exist_ok=True)
                
                # Copy file only if source and target are different (avoid "are the same file" error)
                try:
                    if not os.path.samefile(src_path, target_full_path):
                        self.progress.emit("Copying file to repository...")
                        shutil.copy2(self.file_path, target_full_path)
                    else:
                        self.progress.emit("File already in repository, adding to Git LFS...")
                except OSError:
                    # samefile may fail if paths differ; do the copy
                    self.progress.emit("Copying file to repository...")
                    shutil.copy2(self.file_path, target_full_path)
                
                # Add file to git (will be handled by LFS)
                self.progress.emit("Adding file to Git LFS...")
                git_cmd = self._git_cmd()
                result = subprocess.run(git_cmd + ['add', self.target_path],
                                      capture_output=True,
                                      text=True,
                                      cwd=self.adapter.local_repo_path)
                if result.returncode != 0:
                    self.finished.emit(False, f"Failed to add file to Git LFS: {result.stderr}")
                    return

                # Commit changes
                self.progress.emit("Committing changes...")
                result = subprocess.run(git_cmd + ['commit', '-m', self.commit_message],
                                      capture_output=True,
                                      text=True,
                                      cwd=self.adapter.local_repo_path)
                if result.returncode != 0 and 'nothing to commit' not in (result.stdout or '').lower():
                    self.finished.emit(False, f"Commit failed: {result.stderr}")
                    return

                # Push regular files
                self.progress.emit("Pushing changes to GitHub...")
                result = subprocess.run(git_cmd + ['push', 'origin', 'main'],
                                      capture_output=True,
                                      text=True,
                                      cwd=self.adapter.local_repo_path)
                if result.returncode != 0:
                    self.finished.emit(False, f"Push failed: {result.stderr}")
                    return
                
                # Push LFS files
                self.progress.emit("Pushing Git LFS files...")
                if not self.push_lfs_files():
                    self.finished.emit(False, "Failed to push Git LFS files")
                    return
                
                self.finished.emit(True, f"Upload successful: {self.file_path} (using Git LFS)")
            else:
                # Standard upload
                success, message = self.adapter.upload_file(
                    self.file_path, 
                    self.target_path,
                    self.commit_message
                )
                
                if success:
                    self.finished.emit(True, f"Upload successful: {message}")
                else:
                    self.finished.emit(False, f"Upload failed: {message}")
                
        except Exception as e:
            self.finished.emit(False, f"Upload error: {str(e)}")
    
    def _git_cmd(self):
        """Get git command: use full path if adapter has it, else 'git'"""
        if self.adapter and self.adapter.git_path:
            return [self.adapter.git_path]
        return ['git']

    def check_git_lfs(self):
        """Check if Git LFS is installed"""
        try:
            import subprocess
            cmd = self._git_cmd() + ['lfs', 'version']
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False

    def init_git_lfs(self):
        """Initialize Git LFS"""
        try:
            import subprocess
            cmd = self._git_cmd() + ['lfs', 'install']
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=self.adapter.local_repo_path
            )
            return result.returncode == 0
        except Exception:
            return False

    def track_file_type(self, file_ext):
        """Track file type"""
        try:
            import subprocess
            if file_ext:
                pattern = f"*.{file_ext.lstrip('.')}"
            else:
                pattern = "*.pth"
            cmd = self._git_cmd() + ['lfs', 'track', pattern]
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=self.adapter.local_repo_path
            )
            if result.returncode == 0:
                add_cmd = self._git_cmd() + ['add', '.gitattributes']
                subprocess.run(
                    add_cmd, capture_output=True, text=True,
                    cwd=self.adapter.local_repo_path
                )
            return result.returncode == 0
        except Exception:
            return False

    def push_lfs_files(self):
        """Push LFS files"""
        try:
            import subprocess
            cmd = self._git_cmd() + ['lfs', 'push', 'origin', 'main']
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=self.adapter.local_repo_path
            )
            return result.returncode == 0
        except Exception:
            return False


class UploadDialog(QDialog):
    """Upload dialog"""
    
    def __init__(self, parent=None, upload_type="Model"):
        super(UploadDialog, self).__init__(parent)
        self.parent = parent
        self.upload_type = upload_type  # "Model" or "Data"
        self.file_path = None
        self.github_adapter = None
        self.upload_thread = None
        
        self.setWindowTitle(f"Upload {upload_type}")
        self.setMinimumSize(500, 400)
        self.init_ui()
        self.init_github_adapter()
    
    def init_ui(self):
        """Initialize UI interface"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # 1. GitHub Connection Configuration
        config_group = QGroupBox("GitHub Connection Configuration")
        config_layout = QVBoxLayout()
        
        # Repository information
        repo_layout = QHBoxLayout()
        repo_layout.addWidget(QLabel("Repository Owner:"))
        self.repo_owner_edit = QLineEdit("Yuan-Chongxin")
        repo_layout.addWidget(self.repo_owner_edit)
        repo_layout.addWidget(QLabel("Repository Name:"))
        self.repo_name_edit = QLineEdit("GeoDLFWD")
        repo_layout.addWidget(self.repo_name_edit)
        config_layout.addLayout(repo_layout)
        
        # Token input
        token_layout = QHBoxLayout()
        token_layout.addWidget(QLabel("GitHub Token:"))
        self.token_edit = QLineEdit()
        self.token_edit.setEchoMode(QLineEdit.Password)
        self.token_edit.setPlaceholderText("Enter GitHub Personal Access Token")
        token_layout.addWidget(self.token_edit)
        connect_btn = QPushButton("Connect")
        connect_btn.clicked.connect(self.connect_github)
        token_layout.addWidget(connect_btn)
        config_layout.addLayout(token_layout)
        
        # Connection status
        self.connection_status = QLabel("Not Connected")
        self.connection_status.setStyleSheet("color: gray;")
        config_layout.addWidget(self.connection_status)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # 2. File selection area
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()
        
        file_select_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)
        file_select_layout.addWidget(self.file_label)
        
        select_btn = QPushButton("Select File")
        select_btn.clicked.connect(self.select_file)
        file_select_layout.addWidget(select_btn)
        file_layout.addLayout(file_select_layout)
        
        # File size display
        self.file_size_label = QLabel("")
        self.file_size_label.setStyleSheet("color: #666;")
        file_layout.addWidget(self.file_size_label)
        
        # Git LFS option
        self.lfs_checkbox = QComboBox()
        self.lfs_checkbox.addItems(["Standard Upload", "Use Git LFS (Recommended for large files)"])
        self.lfs_checkbox.setCurrentIndex(1)  # Default to LFS
        file_layout.addWidget(QLabel("Upload Method:"))
        file_layout.addWidget(self.lfs_checkbox)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # 3. Commit message
        commit_group = QGroupBox("Commit Message")
        commit_layout = QVBoxLayout()
        self.commit_message_edit = QLineEdit()
        self.commit_message_edit.setPlaceholderText(f"Upload {self.upload_type} file")
        commit_layout.addWidget(self.commit_message_edit)
        commit_group.setLayout(commit_layout)
        layout.addWidget(commit_group)
        
        # 4. Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 5. Log area
        log_group = QGroupBox("Operation Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        # 6. Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        upload_btn = QPushButton(f"Upload {self.upload_type}")
        upload_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px 20px;")
        upload_btn.clicked.connect(self.upload_file)
        button_layout.addWidget(upload_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
    
    def init_github_adapter(self):
        """Initialize GitHub adapter"""
        try:
            self.github_adapter = GitHubAdapterModule(
                repo_owner=self.repo_owner_edit.text(),
                repo_name=self.repo_name_edit.text()
            )
            self.log("GitHub adapter initialized successfully")
        except Exception as e:
            self.log(f"Failed to initialize GitHub adapter: {str(e)}", error=True)
    
    def connect_github(self):
        """Connect to GitHub"""
        try:
            repo_owner = self.repo_owner_edit.text()
            repo_name = self.repo_name_edit.text()
            token = self.token_edit.text()
            
            if not token:
                QMessageBox.warning(self, "Warning", "Please enter GitHub Token")
                return
            
            # Re-initialize adapter
            self.github_adapter = GitHubAdapterModule(
                repo_owner=repo_owner,
                repo_name=repo_name
            )
            self.github_adapter.set_token(token)
            
            # Check connection
            if self.github_adapter.check_git_available():
                self.connection_status.setText("Connected")
                self.connection_status.setStyleSheet("color: green;")
                self.log("GitHub connection successful")
                
                # Initialize repository
                success, message = self.github_adapter.init_repository()
                self.log(message)
                
                # Set remote repository
                if success:
                    self.github_adapter.set_remote("origin", 
                        f"https://github.com/{repo_owner}/{repo_name}.git")
            else:
                self.connection_status.setText("Git Not Installed")
                self.connection_status.setStyleSheet("color: red;")
                self.log("Git not installed or not found", error=True)
                QMessageBox.warning(
                    self, "Git Not Found",
                    "Git was not detected. Please install Git and ensure it is in your PATH.\n\n"
                    "Download: https://git-scm.com/download/win\n\n"
                    "After installing, restart this application."
                )
                
        except Exception as e:
            self.log(f"Connection failed: {str(e)}", error=True)
            QMessageBox.critical(self, "Error", f"Connection failed: {str(e)}")
    
    def select_file(self):
        """Select file to upload"""
        if self.upload_type == "Model":
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "Select Model File",
                "",
                "Model Files (*.pth *.pt *.h5 *.ckpt *.pkl);;All Files (*.*)"
            )
        else:  # Data
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "Select Data File",
                "",
                "Data Files (*.txt *.csv *.mat *.npy);;All Files (*.*)"
            )
        
        if file_path:
            self.file_path = file_path
            filename = os.path.basename(file_path)
            self.file_label.setText(f"Selected: {filename}")
            
            # Display file size
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            self.file_size_label.setText(f"File Size: {file_size:.2f} MB")
            
            # If file is larger than 25MB, suggest using LFS
            if file_size > 25:
                self.lfs_checkbox.setCurrentIndex(1)  # Auto-select LFS
                self.log(f"File exceeds 25MB ({file_size:.2f} MB), automatically selected Git LFS upload method")
            else:
                self.log(f"File selected: {filename} ({file_size:.2f} MB)")
    
    def upload_file(self):
        """Upload file"""
        if not self.github_adapter:
            QMessageBox.warning(self, "Warning", "Please connect to GitHub first")
            return
        
        if not self.file_path or not os.path.exists(self.file_path):
            QMessageBox.warning(self, "Warning", "Please select a file to upload first")
            return
        
        # Check connection status
        if self.connection_status.text() != "Connected":
            QMessageBox.warning(self, "Warning", "Please connect to GitHub first")
            return
        
        # Determine target path
        if self.upload_type == "Model":
            target_path = f"models/{os.path.basename(self.file_path)}"
        else:  # Data
            target_path = f"data/{os.path.basename(self.file_path)}"
        
        commit_message = self.commit_message_edit.text() or f"Upload {self.upload_type}: {os.path.basename(self.file_path)}"
        use_lfs = self.lfs_checkbox.currentIndex() == 1
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Create upload thread
        self.upload_thread = UploadThread(
            self.github_adapter,
            self.file_path,
            self.upload_type,
            target_path,
            commit_message,
            use_lfs
        )
        self.upload_thread.progress.connect(self.log)
        self.upload_thread.finished.connect(self.on_upload_finished)
        self.upload_thread.start()
    
    def on_upload_finished(self, success, message):
        """Upload finished callback"""
        self.progress_bar.setVisible(False)
        self.log(message, error=not success)
        if success:
            QMessageBox.information(self, "Success", message)
            self.accept()  # Close dialog
        else:
            QMessageBox.warning(self, "Failed", message)
    
    def log(self, message, error=False):
        """Log message"""
        color = "red" if error else "black"
        from PyQt5.QtCore import QDateTime
        timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
        self.log_text.append(f'<span style="color: {color}">[{timestamp}] {message}</span>')
