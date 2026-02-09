#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
网络结构适配模块
为不同复杂度的正演场景提供适配的网络模型选择与参数优化建议
"""

from typing import Dict, List, Optional, Tuple
from enum import Enum

class ModelComplexity(Enum):
    """模型复杂度枚举"""
    SIMPLE = "simple"  # 简单场景
    MEDIUM = "medium"  # 中等复杂度
    COMPLEX = "complex"  # 复杂场景

class NetworkAdapterModule:
    """
    网络结构适配模块
    负责为不同场景推荐合适的网络结构和参数
    """
    
    def __init__(self):
        """初始化网络结构适配模块"""
        # 预定义的网络模型配置
        self.model_configs = {
            'UNet': {
                'complexity': ModelComplexity.MEDIUM,
                'suitable_scenarios': ['标准正演', '中等复杂度地质模型'],
                'parameters': {
                    'in_channels': 1,
                    'out_channels': [2, 4],  # TE/TM或Both
                    'base_features': 64,
                    'depth': 4
                },
                'advantages': ['结构简单', '训练稳定', '适合中等规模数据'],
                'disadvantages': ['对复杂特征提取能力有限']
            },
            'DinkNet50': {
                'complexity': ModelComplexity.COMPLEX,
                'suitable_scenarios': ['复杂正演', '高精度要求', '大规模数据'],
                'parameters': {
                    'in_channels': 1,
                    'out_channels': [2, 4],
                    'backbone': 'resnet50',
                    'pretrained': True
                },
                'advantages': ['特征提取能力强', '适合复杂场景', '精度高'],
                'disadvantages': ['训练时间长', '需要更多计算资源']
            },
            'DinkNet34': {
                'complexity': ModelComplexity.MEDIUM,
                'suitable_scenarios': ['标准正演', '平衡精度与速度'],
                'parameters': {
                    'in_channels': 1,
                    'out_channels': [2, 4],
                    'backbone': 'resnet34',
                    'pretrained': True
                },
                'advantages': ['精度与速度平衡', '适合大多数场景'],
                'disadvantages': ['对极端复杂场景可能不足']
            },
            'SimpleCNN': {
                'complexity': ModelComplexity.SIMPLE,
                'suitable_scenarios': ['简单正演', '快速原型', '小规模数据'],
                'parameters': {
                    'in_channels': 1,
                    'out_channels': [2, 4],
                    'layers': 3,
                    'features': 32
                },
                'advantages': ['训练快速', '资源占用少', '易于调试'],
                'disadvantages': ['精度相对较低', '不适合复杂场景']
            }
        }
    
    def recommend_model(self,
                       scenario_complexity: ModelComplexity,
                       data_size: str = "medium",
                       precision_requirement: str = "medium",
                       speed_requirement: str = "medium") -> List[Tuple[str, Dict]]:
        """
        根据场景推荐合适的网络模型
        
        Args:
            scenario_complexity: 场景复杂度
            data_size: 数据规模（small, medium, large）
            precision_requirement: 精度要求（low, medium, high）
            speed_requirement: 速度要求（low, medium, high）
            
        Returns:
            推荐的模型列表，按推荐优先级排序，每个元素为(模型名, 配置信息)
        """
        recommendations = []
        
        for model_name, config in self.model_configs.items():
            score = 0
            
            # 复杂度匹配
            if config['complexity'] == scenario_complexity:
                score += 3
            elif (scenario_complexity == ModelComplexity.SIMPLE and 
                  config['complexity'] == ModelComplexity.MEDIUM):
                score += 1
            elif (scenario_complexity == ModelComplexity.COMPLEX and 
                  config['complexity'] == ModelComplexity.MEDIUM):
                score += 1
            
            # 精度要求匹配
            if precision_requirement == "high" and config['complexity'] == ModelComplexity.COMPLEX:
                score += 2
            elif precision_requirement == "medium" and config['complexity'] == ModelComplexity.MEDIUM:
                score += 2
            elif precision_requirement == "low" and config['complexity'] == ModelComplexity.SIMPLE:
                score += 2
            
            # 速度要求匹配
            if speed_requirement == "high" and config['complexity'] == ModelComplexity.SIMPLE:
                score += 2
            elif speed_requirement == "medium" and config['complexity'] == ModelComplexity.MEDIUM:
                score += 2
            elif speed_requirement == "low" and config['complexity'] == ModelComplexity.COMPLEX:
                score += 2
            
            # 数据规模匹配
            if data_size == "large" and config['complexity'] == ModelComplexity.COMPLEX:
                score += 1
            elif data_size == "small" and config['complexity'] == ModelComplexity.SIMPLE:
                score += 1
            
            if score > 0:
                recommendations.append((model_name, config, score))
        
        # 按分数排序
        recommendations.sort(key=lambda x: x[2], reverse=True)
        
        # 返回前3个推荐
        return [(name, config) for name, config, _ in recommendations[:3]]
    
    def get_model_config(self, model_name: str) -> Optional[Dict]:
        """
        获取模型配置信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型配置字典，如果不存在则返回None
        """
        return self.model_configs.get(model_name)
    
    def optimize_parameters(self,
                          model_name: str,
                          input_channels: int = 1,
                          output_channels: int = 2,
                          data_dim: Tuple[int, int] = (32, 32)) -> Dict:
        """
        优化模型参数
        
        Args:
            model_name: 模型名称
            input_channels: 输入通道数
            output_channels: 输出通道数
            data_dim: 数据维度 (height, width)
            
        Returns:
            优化后的参数字典
        """
        base_config = self.get_model_config(model_name)
        if not base_config:
            return {}
        
        optimized = base_config['parameters'].copy()
        
        # 根据输入输出通道数调整
        optimized['in_channels'] = input_channels
        optimized['out_channels'] = output_channels
        
        # 根据数据维度调整
        if 'base_features' in optimized:
            # 根据数据大小调整特征数
            total_pixels = data_dim[0] * data_dim[1]
            if total_pixels > 1024:  # 大于32x32
                optimized['base_features'] = max(64, optimized.get('base_features', 64))
            else:
                optimized['base_features'] = min(32, optimized.get('base_features', 64))
        
        # 根据数据维度调整深度
        if 'depth' in optimized:
            if max(data_dim) > 64:
                optimized['depth'] = min(5, optimized.get('depth', 4) + 1)
            elif max(data_dim) < 32:
                optimized['depth'] = max(3, optimized.get('depth', 4) - 1)
        
        return optimized
    
    def compare_models(self, model_names: List[str]) -> Dict:
        """
        比较多个模型的特性
        
        Args:
            model_names: 要比较的模型名称列表
            
        Returns:
            比较结果字典
        """
        comparison = {
            'models': [],
            'comparison_table': {}
        }
        
        for model_name in model_names:
            config = self.get_model_config(model_name)
            if config:
                comparison['models'].append({
                    'name': model_name,
                    'complexity': config['complexity'].value,
                    'suitable_scenarios': config['suitable_scenarios'],
                    'advantages': config['advantages'],
                    'disadvantages': config['disadvantages']
                })
        
        # 创建对比表
        if len(comparison['models']) > 1:
            comparison['comparison_table'] = {
                'complexity': {m['name']: m['complexity'] for m in comparison['models']},
                'scenarios': {m['name']: m['suitable_scenarios'] for m in comparison['models']}
            }
        
        return comparison
    
    def get_training_suggestions(self,
                                model_name: str,
                                data_size: int) -> Dict:
        """
        获取训练建议
        
        Args:
            model_name: 模型名称
            data_size: 训练数据大小
            
        Returns:
            训练建议字典（学习率、批次大小等）
        """
        config = self.get_model_config(model_name)
        if not config:
            return {}
        
        suggestions = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'optimizer': 'Adam',
            'scheduler': 'StepLR'
        }
        
        # 根据模型复杂度和数据大小调整
        if config['complexity'] == ModelComplexity.COMPLEX:
            suggestions['learning_rate'] = 0.0005
            suggestions['batch_size'] = 16 if data_size < 1000 else 32
            suggestions['epochs'] = 150
        elif config['complexity'] == ModelComplexity.SIMPLE:
            suggestions['learning_rate'] = 0.002
            suggestions['batch_size'] = 64 if data_size > 500 else 32
            suggestions['epochs'] = 50
        
        # 根据数据大小调整批次大小
        if data_size < 500:
            suggestions['batch_size'] = min(suggestions['batch_size'], 16)
        elif data_size > 5000:
            suggestions['batch_size'] = max(suggestions['batch_size'], 64)
        
        return suggestions
    
    def list_available_models(self) -> List[str]:
        """
        列出所有可用的模型
        
        Returns:
            模型名称列表
        """
        return list(self.model_configs.keys())
