#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
正演结果解析模块
对训练输出的视电阻率、相位数据进行标准化处理，确保结果的准确性与可读性
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

class ForwardResultParserModule:
    """
    正演结果解析模块
    负责标准化处理正演结果数据
    """
    
    def __init__(self):
        """初始化正演结果解析模块"""
        self.default_units = {
            'resistivity': 'Ohm.m',  # 电阻率单位
            'phase': 'degrees',  # 相位单位
            'frequency': 'Hz'  # 频率单位
        }
    
    def parse_prediction_result(self,
                               result_data: np.ndarray,
                               mode: str = 'TE',
                               data_format: str = 'array') -> Dict:
        """
        解析预测结果
        
        Args:
            result_data: 预测结果数据（numpy数组）
            mode: 模式（TE, TM, Both）
            data_format: 数据格式（array, file）
            
        Returns:
            解析后的结果字典，包含视电阻率和相位数据
        """
        if data_format == 'file':
            # 从文件加载数据
            result_data = np.loadtxt(result_data) if isinstance(result_data, (str, Path)) else result_data
        
        parsed_result = {
            'mode': mode,
            'data_shape': result_data.shape,
            'timestamp': None
        }
        
        # 根据模式解析数据
        if mode == 'TE' or mode == 'TM':
            # 单模式：2通道输出（视电阻率 + 相位）
            if result_data.shape[0] == 2:
                parsed_result['apparent_resistivity'] = result_data[0]
                parsed_result['phase'] = result_data[1]
            else:
                # 如果只有1个通道，假设是视电阻率
                parsed_result['apparent_resistivity'] = result_data[0] if len(result_data.shape) > 1 else result_data
                parsed_result['phase'] = None
                
        elif mode == 'Both':
            # 双模式：4通道输出（TE视电阻率 + TE相位 + TM视电阻率 + TM相位）
            if result_data.shape[0] >= 4:
                parsed_result['TE_apparent_resistivity'] = result_data[0]
                parsed_result['TE_phase'] = result_data[1]
                parsed_result['TM_apparent_resistivity'] = result_data[2]
                parsed_result['TM_phase'] = result_data[3]
            else:
                # 如果通道数不足，尝试其他解析方式
                parsed_result['raw_data'] = result_data
        
        return parsed_result
    
    def standardize_resistivity(self,
                               resistivity_data: np.ndarray,
                               unit: str = 'Ohm.m',
                               log_scale: bool = True) -> np.ndarray:
        """
        标准化视电阻率数据
        
        Args:
            resistivity_data: 视电阻率数据
            unit: 单位
            log_scale: 是否使用对数尺度
            
        Returns:
            标准化后的数据
        """
        # 确保数据为numpy数组
        if not isinstance(resistivity_data, np.ndarray):
            resistivity_data = np.array(resistivity_data)
        
        # 处理无效值
        resistivity_data = np.where(np.isnan(resistivity_data), 0, resistivity_data)
        resistivity_data = np.where(np.isinf(resistivity_data), 0, resistivity_data)
        resistivity_data = np.where(resistivity_data <= 0, 1e-6, resistivity_data)  # 避免负值或零值
        
        # 对数尺度转换
        if log_scale:
            resistivity_data = np.log10(resistivity_data)
        
        return resistivity_data
    
    def standardize_phase(self,
                        phase_data: np.ndarray,
                        unit: str = 'degrees',
                        normalize: bool = True) -> np.ndarray:
        """
        标准化相位数据
        
        Args:
            phase_data: 相位数据
            unit: 单位（degrees或radians）
            normalize: 是否归一化到[-180, 180]度范围
            
        Returns:
            标准化后的数据
        """
        # 确保数据为numpy数组
        if not isinstance(phase_data, np.ndarray):
            phase_data = np.array(phase_data)
        
        # 处理无效值
        phase_data = np.where(np.isnan(phase_data), 0, phase_data)
        phase_data = np.where(np.isinf(phase_data), 0, phase_data)
        
        # 单位转换
        if unit == 'radians':
            phase_data = np.degrees(phase_data)
        
        # 归一化到[-180, 180]度范围
        if normalize:
            phase_data = ((phase_data + 180) % 360) - 180
        
        return phase_data
    
    def validate_result(self,
                       result_data: Dict) -> Tuple[bool, List[str]]:
        """
        验证结果数据的有效性
        
        Args:
            result_data: 结果数据字典
            
        Returns:
            (is_valid, error_messages)
        """
        is_valid = True
        errors = []
        
        # 检查必要字段
        required_fields = ['mode']
        for field in required_fields:
            if field not in result_data:
                is_valid = False
                errors.append(f"缺少必要字段: {field}")
        
        # 检查数据有效性
        mode = result_data.get('mode', '')
        
        if mode in ['TE', 'TM']:
            if 'apparent_resistivity' not in result_data:
                is_valid = False
                errors.append(f"{mode}模式缺少视电阻率数据")
            if 'phase' not in result_data:
                is_valid = False
                errors.append(f"{mode}模式缺少相位数据")
                
        elif mode == 'Both':
            required_both_fields = [
                'TE_apparent_resistivity', 'TE_phase',
                'TM_apparent_resistivity', 'TM_phase'
            ]
            for field in required_both_fields:
                if field not in result_data:
                    is_valid = False
                    errors.append(f"Both模式缺少字段: {field}")
        
        # 检查数据范围
        for key, value in result_data.items():
            if isinstance(value, np.ndarray):
                if np.any(np.isnan(value)):
                    is_valid = False
                    errors.append(f"{key}包含NaN值")
                if np.any(np.isinf(value)):
                    is_valid = False
                    errors.append(f"{key}包含Inf值")
        
        return is_valid, errors
    
    def export_result(self,
                     result_data: Dict,
                     output_path: str,
                     format: str = 'numpy') -> Tuple[bool, str]:
        """
        导出结果数据
        
        Args:
            result_data: 结果数据字典
            output_path: 输出路径
            format: 输出格式（numpy, text, json）
            
        Returns:
            (success, message)
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'numpy':
                # 保存为numpy格式
                np.savez(output_path, **result_data)
                
            elif format == 'text':
                # 保存为文本格式
                with open(output_path, 'w') as f:
                    for key, value in result_data.items():
                        if isinstance(value, np.ndarray):
                            f.write(f"{key}:\n")
                            np.savetxt(f, value)
                            f.write("\n")
                        else:
                            f.write(f"{key}: {value}\n")
                            
            elif format == 'json':
                # 保存为JSON格式（需要将numpy数组转换为列表）
                import json
                json_data = {}
                for key, value in result_data.items():
                    if isinstance(value, np.ndarray):
                        json_data[key] = value.tolist()
                    else:
                        json_data[key] = value
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            return True, f"结果已导出: {output_path}"
            
        except Exception as e:
            return False, f"导出失败: {str(e)}"
    
    def get_statistics(self, result_data: Dict) -> Dict:
        """
        获取结果数据的统计信息
        
        Args:
            result_data: 结果数据字典
            
        Returns:
            统计信息字典
        """
        statistics = {}
        
        for key, value in result_data.items():
            if isinstance(value, np.ndarray):
                statistics[key] = {
                    'shape': value.shape,
                    'min': float(np.nanmin(value)),
                    'max': float(np.nanmax(value)),
                    'mean': float(np.nanmean(value)),
                    'std': float(np.nanstd(value)),
                    'median': float(np.nanmedian(value))
                }
        
        return statistics
    
    def merge_results(self,
                     te_result: Optional[Dict] = None,
                     tm_result: Optional[Dict] = None) -> Dict:
        """
        合并TE和TM模式的结果
        
        Args:
            te_result: TE模式结果
            tm_result: TM模式结果
            
        Returns:
            合并后的结果字典
        """
        merged = {
            'mode': 'Both',
            'data_shape': None
        }
        
        if te_result:
            merged['TE_apparent_resistivity'] = te_result.get('apparent_resistivity')
            merged['TE_phase'] = te_result.get('phase')
        
        if tm_result:
            merged['TM_apparent_resistivity'] = tm_result.get('apparent_resistivity')
            merged['TM_phase'] = tm_result.get('phase')
        
        return merged
