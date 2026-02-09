#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据驱动DL训练框架
核心功能层 - 负责正演模型的训练与结果输出
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Callable, Tuple

class ForwardTrainingFramework:
    """
    正演训练框架
    输入：地质模型参数
    输出：视电阻率和相位（TE/TM/TE&TM模式）
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: Optional[torch.device] = None,
                 verbose: bool = True):
        """
        初始化训练框架
        
        Args:
            model: 深度学习模型（UNet, DinkNet等）
            device: 计算设备（GPU/CPU）
            verbose: 是否打印详细信息
        """
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        
        # 训练参数
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
        # 训练状态
        self.is_training = False
        self.current_epoch = 0
        self.total_epochs = 0
        
        # 回调函数
        self.progress_callback = None
        self.loss_callback = None
        self.epoch_callback = None
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        
        if self.verbose:
            print(f"[ForwardTrainingFramework] 初始化完成，使用设备: {self.device}")
    
    def set_optimizer(self, 
                     optimizer_type: str = 'Adam',
                     learning_rate: float = 0.001,
                     **kwargs):
        """
        设置优化器
        
        Args:
            optimizer_type: 优化器类型 ('Adam', 'SGD', 'RMSprop')
            learning_rate: 学习率
            **kwargs: 其他优化器参数
        """
        if optimizer_type == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, **kwargs)
        elif optimizer_type == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, **kwargs)
        elif optimizer_type == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate, **kwargs)
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        
        if self.verbose:
            print(f"[ForwardTrainingFramework] 优化器设置: {optimizer_type}, 学习率: {learning_rate}")
    
    def set_callbacks(self,
                     progress_callback: Optional[Callable] = None,
                     loss_callback: Optional[Callable] = None,
                     epoch_callback: Optional[Callable] = None):
        """
        设置回调函数
        
        Args:
            progress_callback: 进度回调函数 (epoch, total_epochs, progress_percent)
            loss_callback: 损失回调函数 (train_loss, val_loss)
            epoch_callback: 每个epoch结束回调函数 (epoch, train_loss, val_loss)
        """
        self.progress_callback = progress_callback
        self.loss_callback = loss_callback
        self.epoch_callback = epoch_callback
    
    def train_epoch(self, 
                   train_loader: DataLoader,
                   epoch: int) -> float:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch编号
            
        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (model_params, target_data) in enumerate(train_loader):
            # 将数据移到设备
            model_params = model_params.to(self.device)
            target_data = target_data.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(model_params)
            
            # 计算损失
            # target_data格式: (batch, channels, height, width)
            # channels: [TE_resistivity, TE_phase, TM_resistivity, TM_phase] 或类似
            loss = self.criterion(outputs, target_data)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if self.verbose and (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            平均验证损失
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for model_params, target_data in val_loader:
                model_params = model_params.to(self.device)
                target_data = target_data.to(self.device)
                
                outputs = self.model(model_params)
                loss = self.criterion(outputs, target_data)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def train(self,
             train_loader: DataLoader,
             val_loader: Optional[DataLoader] = None,
             epochs: int = 100,
             save_path: Optional[str] = None,
             save_interval: int = 10) -> Dict:
        """
        执行完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
            epochs: 训练轮数
            save_path: 模型保存路径
            save_interval: 保存间隔（每N个epoch保存一次）
            
        Returns:
            训练历史字典
        """
        if self.optimizer is None:
            raise ValueError("优化器未设置，请先调用 set_optimizer()")
        
        self.is_training = True
        self.total_epochs = epochs
        self.train_losses = []
        self.val_losses = []
        
        best_val_loss = float('inf')
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"开始正演训练")
            print(f"{'='*60}")
            print(f"总epoch数: {epochs}")
            print(f"设备: {self.device}")
            print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            
            # 训练
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = 0.0
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
            
            # 更新进度
            if self.progress_callback:
                progress = int((epoch / epochs) * 100)
                self.progress_callback(epoch, epochs, progress)
            
            # 更新损失曲线
            if self.loss_callback:
                self.loss_callback(train_loss, val_loss if val_loader else None)
            
            # Epoch回调
            if self.epoch_callback:
                self.epoch_callback(epoch, train_loss, val_loss if val_loader else None)
            
            # 打印信息
            if self.verbose:
                if val_loader:
                    print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                else:
                    print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f}")
            
            # 保存最佳模型
            if val_loader and val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_path:
                    best_model_path = save_path.replace('.pth', '_best.pth')
                    self.save_model(best_model_path)
                    if self.verbose:
                        print(f"  -> 保存最佳模型: {best_model_path}")
            
            # 定期保存
            if save_path and epoch % save_interval == 0:
                epoch_model_path = save_path.replace('.pth', f'_epoch_{epoch}.pth')
                self.save_model(epoch_model_path)
        
        total_time = time.time() - start_time
        
        # 保存最终模型
        if save_path:
            self.save_model(save_path)
        
        self.is_training = False
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"训练完成！")
            print(f"总耗时: {total_time // 60:.0f}m {total_time % 60:.0f}s")
            print(f"最终训练损失: {self.train_losses[-1]:.6f}")
            if self.val_losses:
                print(f"最终验证损失: {self.val_losses[-1]:.6f}")
            print(f"{'='*60}\n")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'total_time': total_time,
            'best_val_loss': best_val_loss if val_loader else None
        }
    
    def predict(self, 
               model_params: torch.Tensor,
               mode: str = 'TE&TM') -> Dict[str, np.ndarray]:
        """
        使用训练好的模型进行正演预测
        
        Args:
            model_params: 地质模型参数 (batch, channels, height, width)
            mode: 预测模式 ('TE', 'TM', 'TE&TM')
            
        Returns:
            预测结果字典，包含视电阻率和相位
            {
                'TE': {'resistivity': ..., 'phase': ...},
                'TM': {'resistivity': ..., 'phase': ...}
            }
        """
        self.model.eval()
        
        with torch.no_grad():
            model_params = model_params.to(self.device)
            outputs = self.model(model_params)
            
            # 将输出转换为numpy数组
            outputs_np = outputs.cpu().numpy()
            
            # 解析输出
            # 假设输出格式: (batch, channels, height, width)
            # channels顺序: [TE_resistivity, TE_phase, TM_resistivity, TM_phase]
            results = {}
            
            if mode in ['TE', 'TE&TM']:
                results['TE'] = {
                    'resistivity': outputs_np[:, 0, :, :],
                    'phase': outputs_np[:, 1, :, :]
                }
            
            if mode in ['TM', 'TE&TM']:
                results['TM'] = {
                    'resistivity': outputs_np[:, 2, :, :],
                    'phase': outputs_np[:, 3, :, :]
                }
            
            return results
    
    def save_model(self, save_path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'epoch': self.current_epoch
        }, save_path)
        
        if self.verbose:
            print(f"模型已保存: {save_path}")
    
    def load_model(self, load_path: str):
        """加载模型"""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.current_epoch = checkpoint.get('epoch', 0)
        
        if self.verbose:
            print(f"模型已加载: {load_path}")
