#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
正演数据集类
输入：地质模型参数
输出：视电阻率和相位（TE/TM/TE&TM模式）
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional

class ForwardDataset(Dataset):
    """
    正演训练数据集
    输入：地质模型参数文件
    输出：视电阻率和相位文件（TE/TM模式）
    """
    
    def __init__(self,
                 model_param_files: List[str],
                 te_resistivity_files: List[str],
                 te_phase_files: List[str],
                 tm_resistivity_files: List[str],
                 tm_phase_files: List[str],
                 mode: str = 'TE&TM',
                 normalize: bool = True):
        """
        初始化数据集
        
        Args:
            model_param_files: 模型参数文件路径列表
            te_resistivity_files: TE模式视电阻率文件路径列表
            te_phase_files: TE模式相位文件路径列表
            tm_resistivity_files: TM模式视电阻率文件路径列表
            tm_phase_files: TM模式相位文件路径列表
            mode: 训练模式 ('TE', 'TM', 'TE&TM')
            normalize: 是否归一化数据
        """
        assert len(model_param_files) == len(te_resistivity_files) == len(te_phase_files) == \
               len(tm_resistivity_files) == len(tm_phase_files), \
               "所有文件列表长度必须一致"
        
        self.model_param_files = model_param_files
        self.te_resistivity_files = te_resistivity_files
        self.te_phase_files = te_phase_files
        self.tm_resistivity_files = tm_resistivity_files
        self.tm_phase_files = tm_phase_files
        self.mode = mode
        self.normalize = normalize
        
        # 数据统计信息（用于归一化）
        self.model_mean = None
        self.model_std = None
        self.resistivity_mean = None
        self.resistivity_std = None
        self.phase_mean = None
        self.phase_std = None
        
        # 计算统计信息
        if normalize:
            self._compute_statistics()
    
    def _compute_statistics(self):
        """计算数据统计信息用于归一化"""
        print("正在计算数据统计信息...")
        
        # 采样计算（避免加载所有数据）
        sample_size = min(100, len(self.model_param_files))
        sample_indices = np.linspace(0, len(self.model_param_files) - 1, sample_size, dtype=int)
        
        model_values = []
        resistivity_values = []
        phase_values = []
        
        for idx in sample_indices:
            # 加载模型参数
            model_data = self._load_file(self.model_param_files[idx])
            model_values.append(model_data.flatten())
            
            # 加载视电阻率
            if self.mode in ['TE', 'TE&TM']:
                te_res = self._load_file(self.te_resistivity_files[idx])
                resistivity_values.append(te_res.flatten())
            
            if self.mode in ['TM', 'TE&TM']:
                tm_res = self._load_file(self.tm_resistivity_files[idx])
                resistivity_values.append(tm_res.flatten())
            
            # 加载相位
            if self.mode in ['TE', 'TE&TM']:
                te_ph = self._load_file(self.te_phase_files[idx])
                phase_values.append(te_ph.flatten())
            
            if self.mode in ['TM', 'TE&TM']:
                tm_ph = self._load_file(self.tm_phase_files[idx])
                phase_values.append(tm_ph.flatten())
        
        # 计算均值和标准差
        if model_values:
            all_models = np.concatenate(model_values)
            self.model_mean = np.mean(all_models)
            self.model_std = np.std(all_models) + 1e-8
        
        if resistivity_values:
            all_resistivity = np.concatenate(resistivity_values)
            self.resistivity_mean = np.mean(all_resistivity)
            self.resistivity_std = np.std(all_resistivity) + 1e-8
        
        if phase_values:
            all_phase = np.concatenate(phase_values)
            self.phase_mean = np.mean(all_phase)
            self.phase_std = np.std(all_phase) + 1e-8
        
        print(f"数据统计信息:")
        print(f"  模型参数: mean={self.model_mean:.6f}, std={self.model_std:.6f}")
        print(f"  视电阻率: mean={self.resistivity_mean:.6f}, std={self.resistivity_std:.6f}")
        print(f"  相位: mean={self.phase_mean:.6f}, std={self.phase_std:.6f}")
    
    def _load_file(self, filepath: str) -> np.ndarray:
        """加载数据文件"""
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.txt':
            return np.loadtxt(filepath)
        elif ext == '.npy':
            return np.load(filepath)
        elif ext == '.dat':
            # 尝试作为文本文件加载
            try:
                return np.loadtxt(filepath)
            except:
                return np.fromfile(filepath, dtype=np.float32)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
    
    def __len__(self) -> int:
        return len(self.model_param_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一个数据样本
        
        Returns:
            (model_params, target_data)
            - model_params: 模型参数 (channels, height, width)
            - target_data: 目标数据 (channels, height, width)
                channels顺序: [TE_resistivity, TE_phase, TM_resistivity, TM_phase]
        """
        # 加载模型参数
        model_data = self._load_file(self.model_param_files[idx])
        
        # 归一化模型参数
        if self.normalize and self.model_mean is not None:
            model_data = (model_data - self.model_mean) / self.model_std
        
        model_tensor = torch.from_numpy(model_data).float()
        if model_tensor.dim() == 2:
            model_tensor = model_tensor.unsqueeze(0)  # 添加通道维度
        
        # 构建目标数据
        target_channels = []
        
        if self.mode in ['TE', 'TE&TM']:
            # TE模式视电阻率
            te_res = self._load_file(self.te_resistivity_files[idx])
            if self.normalize and self.resistivity_mean is not None:
                te_res = (te_res - self.resistivity_mean) / self.resistivity_std
            target_channels.append(torch.from_numpy(te_res).float())
            
            # TE模式相位
            te_ph = self._load_file(self.te_phase_files[idx])
            if self.normalize and self.phase_mean is not None:
                te_ph = (te_ph - self.phase_mean) / self.phase_std
            target_channels.append(torch.from_numpy(te_ph).float())
        
        if self.mode in ['TM', 'TE&TM']:
            # TM模式视电阻率
            tm_res = self._load_file(self.tm_resistivity_files[idx])
            if self.normalize and self.resistivity_mean is not None:
                tm_res = (tm_res - self.resistivity_mean) / self.resistivity_std
            target_channels.append(torch.from_numpy(tm_res).float())
            
            # TM模式相位
            tm_ph = self._load_file(self.tm_phase_files[idx])
            if self.normalize and self.phase_mean is not None:
                tm_ph = (tm_ph - self.phase_mean) / self.phase_std
            target_channels.append(torch.from_numpy(tm_ph).float())
        
        # 只在TE或TM模式时，需要填充缺失的通道
        if self.mode == 'TE':
            # 填充TM通道为0
            target_channels.extend([torch.zeros_like(target_channels[0]), 
                                   torch.zeros_like(target_channels[1])])
        elif self.mode == 'TM':
            # 在开头填充TE通道为0
            target_channels = [torch.zeros_like(target_channels[0]), 
                              torch.zeros_like(target_channels[1])] + target_channels
        
        # 堆叠为多通道张量
        target_tensor = torch.stack(target_channels, dim=0)
        
        return model_tensor, target_tensor
