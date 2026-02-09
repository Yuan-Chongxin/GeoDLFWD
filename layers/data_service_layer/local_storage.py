#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本地数据存储模块
负责保存研究人员的自有训练数据、本地训练模型及训练日志
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class LocalStorageModule:
    """
    本地数据存储模块
    负责本地数据的存储、管理与检索
    """
    
    def __init__(self, base_dir: str = "./data"):
        """
        初始化本地存储模块
        
        Args:
            base_dir: 基础存储目录
        """
        self.base_dir = Path(base_dir)
        self.training_data_dir = self.base_dir / "training_data"
        self.models_dir = self.base_dir / "models"
        self.logs_dir = self.base_dir / "logs"
        self.results_dir = self.base_dir / "results"
        
        # 创建目录结构
        self._create_directories()
    
    def _create_directories(self):
        """创建必要的目录结构"""
        directories = [
            self.training_data_dir,
            self.models_dir,
            self.logs_dir,
            self.results_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def save_training_data(self,
                          data_path: str,
                          metadata: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        保存训练数据
        
        Args:
            data_path: 训练数据文件/目录路径
            metadata: 元数据信息（如数据描述、创建时间等）
            
        Returns:
            (success, message)
        """
        try:
            source_path = Path(data_path)
            if not source_path.exists():
                return False, f"源路径不存在: {data_path}"
            
            # 生成目标路径
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if source_path.is_file():
                target_path = self.training_data_dir / f"{timestamp}_{source_path.name}"
                shutil.copy2(source_path, target_path)
            else:
                target_path = self.training_data_dir / f"{timestamp}_{source_path.name}"
                shutil.copytree(source_path, target_path)
            
            # 保存元数据
            if metadata is None:
                metadata = {}
            
            metadata['source_path'] = str(source_path)
            metadata['saved_path'] = str(target_path)
            metadata['saved_time'] = datetime.now().isoformat()
            
            metadata_file = target_path.parent / f"{target_path.name}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            return True, f"训练数据已保存: {target_path}"
            
        except Exception as e:
            return False, f"保存训练数据失败: {str(e)}"
    
    def save_model(self,
                  model_path: str,
                  model_name: Optional[str] = None,
                  metadata: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        保存训练模型
        
        Args:
            model_path: 模型文件路径
            model_name: 模型名称（可选）
            metadata: 元数据信息（如训练参数、性能指标等）
            
        Returns:
            (success, message)
        """
        try:
            source_path = Path(model_path)
            if not source_path.exists():
                return False, f"模型文件不存在: {model_path}"
            
            # 生成目标路径
            if model_name:
                target_path = self.models_dir / model_name
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                target_path = self.models_dir / f"model_{timestamp}{source_path.suffix}"
            
            shutil.copy2(source_path, target_path)
            
            # 保存元数据
            if metadata is None:
                metadata = {}
            
            metadata['source_path'] = str(source_path)
            metadata['saved_path'] = str(target_path)
            metadata['saved_time'] = datetime.now().isoformat()
            
            metadata_file = target_path.parent / f"{target_path.stem}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            return True, f"模型已保存: {target_path}"
            
        except Exception as e:
            return False, f"保存模型失败: {str(e)}"
    
    def save_training_log(self,
                         log_data: Dict,
                         log_name: Optional[str] = None) -> Tuple[bool, str]:
        """
        保存训练日志
        
        Args:
            log_data: 日志数据字典
            log_name: 日志文件名（可选）
            
        Returns:
            (success, message)
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if log_name:
                log_file = self.logs_dir / f"{log_name}_{timestamp}.json"
            else:
                log_file = self.logs_dir / f"training_log_{timestamp}.json"
            
            log_data['timestamp'] = datetime.now().isoformat()
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            return True, f"训练日志已保存: {log_file}"
            
        except Exception as e:
            return False, f"保存训练日志失败: {str(e)}"
    
    def list_training_data(self) -> List[Dict]:
        """
        列出所有训练数据
        
        Returns:
            训练数据列表，每个元素包含路径和元数据
        """
        data_list = []
        
        for item in self.training_data_dir.iterdir():
            if item.is_file() and not item.name.endswith('_metadata.json'):
                metadata_file = item.parent / f"{item.stem}_metadata.json"
                metadata = {}
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                    except:
                        pass
                
                data_list.append({
                    'path': str(item),
                    'name': item.name,
                    'size': item.stat().st_size,
                    'modified': datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
                    'metadata': metadata
                })
        
        return sorted(data_list, key=lambda x: x['modified'], reverse=True)
    
    def list_models(self) -> List[Dict]:
        """
        列出所有保存的模型
        
        Returns:
            模型列表，每个元素包含路径和元数据
        """
        model_list = []
        
        for item in self.models_dir.iterdir():
            if item.is_file() and not item.name.endswith('_metadata.json'):
                metadata_file = item.parent / f"{item.stem}_metadata.json"
                metadata = {}
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                    except:
                        pass
                
                model_list.append({
                    'path': str(item),
                    'name': item.name,
                    'size': item.stat().st_size,
                    'modified': datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
                    'metadata': metadata
                })
        
        return sorted(model_list, key=lambda x: x['modified'], reverse=True)
    
    def list_logs(self) -> List[Dict]:
        """
        列出所有训练日志
        
        Returns:
            日志列表，每个元素包含路径和基本信息
        """
        log_list = []
        
        for item in self.logs_dir.iterdir():
            if item.is_file() and item.suffix == '.json':
                try:
                    with open(item, 'r', encoding='utf-8') as f:
                        log_data = json.load(f)
                    
                    log_list.append({
                        'path': str(item),
                        'name': item.name,
                        'size': item.stat().st_size,
                        'timestamp': log_data.get('timestamp', ''),
                        'summary': log_data.get('summary', {})
                    })
                except:
                    pass
        
        return sorted(log_list, key=lambda x: x['timestamp'], reverse=True)
    
    def get_training_data(self, data_name: str) -> Optional[str]:
        """
        获取训练数据路径
        
        Args:
            data_name: 数据名称或路径
            
        Returns:
            数据完整路径，如果不存在则返回None
        """
        data_path = self.training_data_dir / data_name
        if data_path.exists():
            return str(data_path)
        return None
    
    def get_model(self, model_name: str) -> Optional[str]:
        """
        获取模型路径
        
        Args:
            model_name: 模型名称或路径
            
        Returns:
            模型完整路径，如果不存在则返回None
        """
        model_path = self.models_dir / model_name
        if model_path.exists():
            return str(model_path)
        return None
    
    def delete_training_data(self, data_name: str) -> Tuple[bool, str]:
        """
        删除训练数据
        
        Args:
            data_name: 数据名称
            
        Returns:
            (success, message)
        """
        try:
            data_path = self.training_data_dir / data_name
            if data_path.exists():
                if data_path.is_file():
                    data_path.unlink()
                else:
                    shutil.rmtree(data_path)
                
                # 删除元数据文件
                metadata_file = data_path.parent / f"{data_path.stem}_metadata.json"
                if metadata_file.exists():
                    metadata_file.unlink()
                
                return True, f"训练数据已删除: {data_name}"
            else:
                return False, f"训练数据不存在: {data_name}"
                
        except Exception as e:
            return False, f"删除训练数据失败: {str(e)}"
    
    def delete_model(self, model_name: str) -> Tuple[bool, str]:
        """
        删除模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            (success, message)
        """
        try:
            model_path = self.models_dir / model_name
            if model_path.exists():
                model_path.unlink()
                
                # 删除元数据文件
                metadata_file = model_path.parent / f"{model_path.stem}_metadata.json"
                if metadata_file.exists():
                    metadata_file.unlink()
                
                return True, f"模型已删除: {model_name}"
            else:
                return False, f"模型不存在: {model_name}"
                
        except Exception as e:
            return False, f"删除模型失败: {str(e)}"
