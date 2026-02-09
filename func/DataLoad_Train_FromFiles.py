# -*- coding: utf-8 -*-
"""
从文件列表加载训练数据
基于用户选择的文件路径加载数据，而不是从目录中读取

创建于2024年
"""

import numpy as np
from skimage.measure import block_reduce
import os

# Import from DataLoad_Train module
try:
    from func.DataLoad_Train import validate_data, decimate
except ImportError:
    # Fallback if import fails
    from DataLoad_Train import validate_data, decimate

def DataLoad_Train_FromFiles(model_files, resistivity_files=None, phase_files=None,
                             te_resistivity_files=None, te_phase_files=None,
                             tm_resistivity_files=None, tm_phase_files=None,
                             data_dim=[32, 32], model_dim=[32, 32],
                             data_dsp_blk=(1, 1), label_dsp_blk=(1, 1),
                             MT_Mode='TM'):
    """
    从文件列表加载训练数据
    
    参数:
    - model_files: 电阻率模型文件路径列表
    - resistivity_files: 视电阻率文件路径列表（TE或TM模式）
    - phase_files: 相位文件路径列表（TE或TM模式）
    - te_resistivity_files: TE视电阻率文件路径列表（Both模式）
    - te_phase_files: TE相位文件路径列表（Both模式）
    - tm_resistivity_files: TM视电阻率文件路径列表（Both模式）
    - tm_phase_files: TM相位文件路径列表（Both模式）
    - data_dim: 数据维度
    - model_dim: 模型维度
    - data_dsp_blk: 数据降采样块大小
    - label_dsp_blk: 标签降采样块大小
    - MT_Mode: MT模式（'TE', 'TM', 'Both'）
    
    返回:
    - train_set: 训练数据集（电阻率模型）
    - label_set: 标签数据集（视电阻率和相位）
    - data_dsp_dim: 数据降采样维度
    - label_dsp_dim: 标签降采样维度
    - valid_count: 有效的训练数据个数
    """
    print(f"[DataLoad_Train_FromFiles] 开始从文件列表加载训练数据...")
    print(f"MT Mode: {MT_Mode}")
    print(f"Model files count: {len(model_files) if model_files else 0}")
    
    # Print file paths for verification (showing index matching)
    if model_files:
        print(f"[DataLoad_Train_FromFiles] Model files (first 5 with indices):")
        for i, f in enumerate(model_files[:5]):
            print(f"  [{i}] {os.path.basename(f)}")
    if MT_Mode == 'TE' or MT_Mode == 'TM':
        if resistivity_files:
            print(f"[DataLoad_Train_FromFiles] Resistivity files (first 5 with indices):")
            for i, f in enumerate(resistivity_files[:5]):
                print(f"  [{i}] {os.path.basename(f)}")
        if phase_files:
            print(f"[DataLoad_Train_FromFiles] Phase files (first 5 with indices):")
            for i, f in enumerate(phase_files[:5]):
                print(f"  [{i}] {os.path.basename(f)}")
    else:  # Both mode
        if te_resistivity_files:
            print(f"[DataLoad_Train_FromFiles] TE Resistivity files (first 5 with indices):")
            for i, f in enumerate(te_resistivity_files[:5]):
                print(f"  [{i}] {os.path.basename(f)}")
        if te_phase_files:
            print(f"[DataLoad_Train_FromFiles] TE Phase files (first 5 with indices):")
            for i, f in enumerate(te_phase_files[:5]):
                print(f"  [{i}] {os.path.basename(f)}")
        if tm_resistivity_files:
            print(f"[DataLoad_Train_FromFiles] TM Resistivity files (first 5 with indices):")
            for i, f in enumerate(tm_resistivity_files[:5]):
                print(f"  [{i}] {os.path.basename(f)}")
        if tm_phase_files:
            print(f"[DataLoad_Train_FromFiles] TM Phase files (first 5 with indices):")
            for i, f in enumerate(tm_phase_files[:5]):
                print(f"  [{i}] {os.path.basename(f)}")
    
    invalid_files_list = []
    valid_count = 0
    
    # Input: resistivity model (1 ch). Output: TE&TM = TE apparent resistivity, TE phase, TM apparent resistivity, TM phase (4 ch)
    if MT_Mode == 'Both' or MT_Mode == 'TE&TM':
        out_channels = 4  # TE apparent resistivity, TE phase, TM apparent resistivity, TM phase
        data_channels = 4
    else:
        out_channels = 2  # TE or TM: apparent resistivity + phase
        data_channels = 2
    
    try:
        # 确保文件数量匹配
        if MT_Mode == 'TE' or MT_Mode == 'TM':
            if not resistivity_files or not phase_files:
                raise ValueError(f"{MT_Mode} mode requires resistivity_files and phase_files")
            if len(model_files) != len(resistivity_files) or len(model_files) != len(phase_files):
                raise ValueError(f"File count mismatch: model({len(model_files)}), resistivity({len(resistivity_files)}), phase({len(phase_files)})")
        else:  # Both mode
            if not te_resistivity_files or not te_phase_files or not tm_resistivity_files or not tm_phase_files:
                raise ValueError("Both mode requires all TE and TM files")
            if not (len(model_files) == len(te_resistivity_files) == len(te_phase_files) == 
                    len(tm_resistivity_files) == len(tm_phase_files)):
                raise ValueError(f"File count mismatch in Both mode")
        
        # 处理每个文件对
        for i in range(len(model_files)):
            model_file = model_files[i]
            
            try:
                # 加载电阻率模型
                if not os.path.exists(model_file):
                    print(f"Warning: Model file not found: {model_file}")
                    continue
                raw_model = np.loadtxt(model_file, encoding='utf-8')
                
                if MT_Mode == 'TE' or MT_Mode == 'TM':
                    # TE或TM模式
                    resistivity_file = resistivity_files[i]
                    phase_file = phase_files[i]
                    
                    if not os.path.exists(resistivity_file) or not os.path.exists(phase_file):
                        print(f"Warning: Data files not found for index {i}")
                        print(f"  Model file: {os.path.basename(model_file)}")
                        print(f"  Resistivity file: {os.path.basename(resistivity_file)}")
                        print(f"  Phase file: {os.path.basename(phase_file)}")
                        continue
                    
                    # Verify file index matching (for debugging - print first 5)
                    if i < 5:
                        print(f"[DataLoad_Train_FromFiles] Processing index {i} (matching by file number):")
                        print(f"  Model[{i}]: {os.path.basename(model_file)}")
                        print(f"  Resistivity[{i}]: {os.path.basename(resistivity_file)}")
                        print(f"  Phase[{i}]: {os.path.basename(phase_file)}")
                    
                    raw_resistivity = np.loadtxt(resistivity_file, encoding='utf-8')
                    raw_phase = np.loadtxt(phase_file, encoding='utf-8')
                    
                    # 验证文件读取：打印前5个样本的原始数据范围
                    if i < 5:
                        print(f"[DataLoad] File reading verification (index {i}):")
                        print(f"  Resistivity file: {os.path.basename(resistivity_file)}")
                        print(f"    Raw data shape: {raw_resistivity.shape}")
                        print(f"    Raw data range: [{raw_resistivity.min():.6f}, {raw_resistivity.max():.6f}]")
                        print(f"    Expected: positive values, typically 0.1-1000 Ω·m")
                        print(f"  Phase file: {os.path.basename(phase_file)}")
                        print(f"    Raw data shape: {raw_phase.shape}")
                        print(f"    Raw data range: [{raw_phase.min():.6f}, {raw_phase.max():.6f}]")
                        print(f"    Expected: 0-90 degrees")
                    
                    # 数据校验
                    is_valid, invalid_files = validate_data(raw_resistivity, raw_phase, raw_model, i, MT_Mode)
                    if not is_valid:
                        invalid_files_list.extend(invalid_files)
                        #print(f"跳过无效数据(Index: {i}):")
                        for reason in invalid_files:
                            print(f"  - {reason}")
                        continue
                    
                    # 处理数据
                    # Step 1: Reshape raw_resistivity from 1D to 2D, then transpose
                    # raw_resistivity shape: (data_dim[0]*data_dim[1],) -> reshape -> (data_dim[0], data_dim[1]) -> .T -> (data_dim[1], data_dim[0])
                    train_data1 = np.reshape(raw_resistivity, (data_dim[0], data_dim[1])).T
                    train_data1 = np.log10(train_data1)
                    
                    train_data2 = np.reshape(raw_phase, (data_dim[0], data_dim[1])).T
                    
                    # Step 2: Combine resistivity and phase: (2, data_dim[1], data_dim[0])
                    # IMPORTANT: Channel order must match prediction
                    # data1_set[0] = resistivity (log10), data1_set[1] = phase (no log10)
                    data1_set = np.array([train_data1, train_data2])
                    # Step 3: Transpose to (data_dim[1], data_dim[0], 2)
                    # After transpose: data1_set[:, :, 0] = resistivity (log10), data1_set[:, :, 1] = phase
                    data1_set = np.transpose(data1_set, (1, 2, 0))
                    
                    # Verify channel order for first sample
                    if i == 0:
                        print(f"[DataLoad] Channel order verification (first sample):")
                        print(f"  Source files:")
                        print(f"    Resistivity file: {os.path.basename(resistivity_file)}")
                        print(f"    Phase file: {os.path.basename(phase_file)}")
                        print(f"  After processing:")
                        print(f"    data1_set[:, :, 0] (resistivity, log10): min={data1_set[:, :, 0].min():.6f}, max={data1_set[:, :, 0].max():.6f}")
                        print(f"      (Original resistivity range: [{raw_resistivity.min():.6f}, {raw_resistivity.max():.6f}] Ω·m)")
                        print(f"    data1_set[:, :, 1] (phase, no log10): min={data1_set[:, :, 1].min():.6f}, max={data1_set[:, :, 1].max():.6f}")
                        print(f"      (Original phase range: [{raw_phase.min():.6f}, {raw_phase.max():.6f}] degrees)")
                    
                    # 处理每个通道（视电阻率和相位，2通道）- 输出用label_dsp_blk
                    for k in range(data_channels):
                        data11_set = np.float32(data1_set[:, :, k])
                        data11_set = block_reduce(data11_set, block_size=label_dsp_blk, func=decimate)
                        label_dsp_dim = data11_set.shape
                        data11_set = data11_set.reshape(1, label_dsp_dim[0] * label_dsp_dim[1])
                        # Debug: Print shape for first sample
                        if i == 0 and k == 0:
                            print(f"[DataLoad] Channel {k} (resistivity) after downsample: shape={label_dsp_dim}, range=[{data11_set.min():.6f}, {data11_set.max():.6f}]")
                        if i == 0 and k == 1:
                            print(f"[DataLoad] Channel {k} (phase) after downsample: shape={label_dsp_dim}, range=[{data11_set.min():.6f}, {data11_set.max():.6f}]")
                        if k == 0:
                            labelr_set = data11_set
                        else:
                            labelr_set = np.append(labelr_set, data11_set, axis=0)
                    
                    # 处理输入（电阻率模型，1通道）- 来自model_dim，用data_dsp_blk
                    train_label1 = raw_model
                    train_label1 = np.log10(train_label1)
                    train_label1 = block_reduce(train_label1, block_size=data_dsp_blk, func=np.max)
                    data_dsp_dim = train_label1.shape
                    train_label1 = train_label1.reshape(1, data_dsp_dim[0] * data_dsp_dim[1])
                    train_label1 = np.float32(train_label1)
                    
                else:  # Both mode
                    # Both模式：需要加载TE和TM的所有文件
                    te_res_file = te_resistivity_files[i]
                    te_ph_file = te_phase_files[i]
                    tm_res_file = tm_resistivity_files[i]
                    tm_ph_file = tm_phase_files[i]
                    
                    if not all(os.path.exists(f) for f in [te_res_file, te_ph_file, tm_res_file, tm_ph_file]):
                        print(f"Warning: Some data files not found for index {i}")
                        continue
                    
                    te_resistivity = np.loadtxt(te_res_file, encoding='utf-8')
                    te_phase = np.loadtxt(te_ph_file, encoding='utf-8')
                    tm_resistivity = np.loadtxt(tm_res_file, encoding='utf-8')
                    tm_phase = np.loadtxt(tm_ph_file, encoding='utf-8')
                    
                    # 数据校验（使用TE数据进行校验）
                    is_valid, invalid_files = validate_data(te_resistivity, te_phase, raw_model, i, 'TE')
                    if not is_valid:
                        invalid_files_list.extend(invalid_files)
                        #print(f"跳过无效数据(Index: {i})")
                        continue
                    
                    # 处理4个通道的数据
                    te_res_data = np.reshape(te_resistivity, (data_dim[0], data_dim[1])).T
                    te_res_data = np.log10(te_res_data)
                    te_ph_data = np.reshape(te_phase, (data_dim[0], data_dim[1])).T
                    tm_res_data = np.reshape(tm_resistivity, (data_dim[0], data_dim[1])).T
                    tm_res_data = np.log10(tm_res_data)
                    tm_ph_data = np.reshape(tm_phase, (data_dim[0], data_dim[1])).T
                    
                    data1_set = np.array([te_res_data, te_ph_data, tm_res_data, tm_ph_data])
                    data1_set = np.transpose(data1_set, (1, 2, 0))
                    
                    # 处理每个通道（4通道）- 输出用label_dsp_blk
                    for k in range(data_channels):
                        data11_set = np.float32(data1_set[:, :, k])
                        data11_set = block_reduce(data11_set, block_size=label_dsp_blk, func=decimate)
                        label_dsp_dim = data11_set.shape
                        data11_set = data11_set.reshape(1, label_dsp_dim[0] * label_dsp_dim[1])
                        if k == 0:
                            labelr_set = data11_set
                        else:
                            labelr_set = np.append(labelr_set, data11_set, axis=0)
                    
                    # 处理输入（电阻率模型）- 来自model_dim，用data_dsp_blk
                    train_label1 = raw_model
                    train_label1 = np.log10(train_label1)
                    train_label1 = block_reduce(train_label1, block_size=data_dsp_blk, func=np.max)
                    data_dsp_dim = train_label1.shape
                    train_label1 = train_label1.reshape(1, data_dsp_dim[0] * data_dsp_dim[1])
                    train_label1 = np.float32(train_label1)
                
                # 添加到数据集
                if valid_count == 0:
                    train_set = train_label1
                    label_set = labelr_set
                else:
                    train_set = np.append(train_set, train_label1, axis=0)
                    label_set = np.append(label_set, labelr_set, axis=0)
                
                valid_count += 1
                
            except Exception as e:
                print(f"Error processing file pair {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 调整数据形状：train_set=输入(data_dsp_dim)，label_set=输出(label_dsp_dim)
        if valid_count > 0:
            train_set = train_set.reshape((valid_count, 1, data_dsp_dim[0] * data_dsp_dim[1]))
            label_set = label_set.reshape((valid_count, out_channels, label_dsp_dim[0] * label_dsp_dim[1]))
            
            # Verify channel order for first sample
            if valid_count > 0:
                print(f"[DataLoad] Final label_set shape: {label_set.shape}")
                print(f"[DataLoad] Channel order verification (first sample):")
                print(f"  label_set[0, 0, :] (resistivity, log10): min={label_set[0, 0, :].min():.6f}, max={label_set[0, 0, :].max():.6f}")
                if out_channels >= 2:
                    print(f"  label_set[0, 1, :] (phase, no log10): min={label_set[0, 1, :].min():.6f}, max={label_set[0, 1, :].max():.6f}")
        
        # 处理没有有效数据的情况
        if valid_count == 0:
            print("[DataLoad_Train_FromFiles] 警告: 没有有效数据，返回空数组和默认维度")
            # 初始化默认维度
            data_dsp_dim = (1, 1)
            label_dsp_dim = (1, 1)
            train_set = np.array([])
            label_set = np.array([])
        
        print(f"[DataLoad_Train_FromFiles] 成功加载 {valid_count} 个有效训练样本")
        print(f"训练集形状 (输入-电阻率模型): {train_set.shape if valid_count > 0 else 'N/A'}")
        print(f"标签集形状 (输出-视电阻率和相位): {label_set.shape if valid_count > 0 else 'N/A'}")
        if invalid_files_list:
            print(f"[DataLoad_Train_FromFiles] 跳过了 {len(invalid_files_list)} 个无效文件")
        
        # MT_train约定: data_dsp_dim=输出维度, label_dsp_dim=输入维度
        return train_set, label_set, label_dsp_dim, data_dsp_dim, valid_count
        
    except Exception as e:
        print(f"Error in DataLoad_Train_FromFiles: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, 0
