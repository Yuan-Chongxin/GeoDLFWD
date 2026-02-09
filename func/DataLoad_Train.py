# -*- coding: utf-8 -*-
"""
读取训练数据

创建于2025年7月

作者：ycx

"""

import numpy as np
from skimage.measure import block_reduce
import skimage
import scipy.io
import pandas as pd
from IPython.core.debugger import set_trace
from scipy.interpolate import lagrange
import os
import cv2

def validate_data(resistivity_data, phase_data, model_data, file_id, mode):
    """
    校验视电阻率、相位和电阻率模型数据是否有效
    
    参数:
    - resistivity_data: 视电阻率数据
    - phase_data: 相位数据
    - model_data: 电阻率模型数据
    - file_id: 文件编号
    - mode: 模式（'TE'或'TM'）
    
    返回:
    - 布尔值：数据是否有效
    - 列表：需要删除的文件路径
    """
    invalid_files = []
    valid = True
    
    # 检查视电阻率数据是否有负数或异常值
    if np.any(np.logical_or(resistivity_data <= 0, resistivity_data >= 10000000)):
        valid = False
        min_res = np.min(resistivity_data)
        max_res = np.max(resistivity_data)
        invalid_files.append(f"{mode}视电阻率文件(ID: {file_id})包含负数或值异常: min={min_res:.2f}, max={max_res:.2f}")
        
    # 检查相位数据是否在0-90之间
    if np.any(phase_data < 0) or np.any(phase_data > 90):
        valid = False
        min_phase = np.min(phase_data)
        max_phase = np.max(phase_data)
        invalid_files.append(f"{mode}相位文件(ID: {file_id})值不在0-90区间: min={min_phase:.2f}, max={max_phase:.2f}")
        
    # 检查电阻率模型数据是否有负数或异常值
    if np.any(np.logical_or(model_data <= 0, model_data >= 10000000)):
        valid = False
        min_model = np.min(model_data)
        max_model = np.max(model_data)
        invalid_files.append(f"电阻率模型文件(ID: {file_id})包含负数或值异常: min={min_model:.2f}, max={max_model:.2f}")
    
    return valid, invalid_files

def DataLoad_Train(train_size, train_data_dir, data_dim, out_channels, model_dim, data_dsp_blk, label_dsp_blk, start,
                   datafilename, dataname, truthfilename, truthname, 
                   TE_Resistivity_Dir, TE_Phase_Dir, TM_Resistivity_Dir, TM_Phase_Dir,
                  Resistivity_Model_Dir, MT_Mode):
    """
    加载训练数据，支持TE、TM或Both模式，并对视电阻率、相位和电阻率模型进行校验
    
    注意：模型的输入是电阻率模型（1通道），输出是视电阻率和相位（2或4通道）
    
    参数:
    - train_size: 训练数据大小
    - out_channels: 输出通道数（2对应TE或TM模式，4对应Both模式）- 视电阻率和相位的通道数
    - MT_Mode: MT模式（'TE'、'TM'或'Both'）
    - 其他参数为数据路径和处理参数
    
    返回:
    - train_set: 训练数据集（电阻率模型，1通道）- 模型的输入
    - label_set: 标签数据集（视电阻率和相位，2或4通道）- 模型的输出
    - data_dsp_dim: 数据降采样维度
    - label_dsp_dim: 标签降采样维度
    - valid_count: 有效的训练数据个数
    """
    import time
    start_time = time.time()
    print(f"[DataLoad_Train] 开始加载训练数据，预计处理 {train_size} 个样本...")
    
    # Model channel configuration:
    # in_channels = 1: Model input is resistivity model (1 channel)
    # out_channels: Model output is apparent resistivity + phase (2 or 4 channels)
    #   - TE or TM mode: out_channels = 2 (resistivity + phase)
    #   - Both mode: out_channels = 4 (TE resistivity + TE phase + TM resistivity + TM phase)
    in_channels = 1  # Model input channels: resistivity model (fixed at 1)
    # out_channels is already a function parameter representing model output channels
    
    invalid_files_list = []
    valid_count = 0
    
    # 添加错误处理，确保函数在KeyboardInterrupt时也能提供有意义的信息
    try:
        # Use out_channels parameter: output has 2 channels AND mode is TE
        if out_channels == 2 and MT_Mode == 'TE':
            for i in range(start, start + train_size):
                # 每处理10个数据打印一次进度，方便监控
                if (i - start) % 10 == 0 and (i - start) > 0:
                    print(f"[DataLoad_Train] 已处理 {i - start} 个样本，有效数据: {valid_count}")
                
                # 加载原始数据用于校验
                # 确保目录路径以斜杠结尾
                te_res_dir = TE_Resistivity_Dir + '/' if not TE_Resistivity_Dir.endswith('/') and not TE_Resistivity_Dir.endswith('\\') else TE_Resistivity_Dir
                te_ph_dir = TE_Phase_Dir + '/' if not TE_Phase_Dir.endswith('/') and not TE_Phase_Dir.endswith('\\') else TE_Phase_Dir
                res_mod_dir = Resistivity_Model_Dir + '/' if not Resistivity_Model_Dir.endswith('/') and not Resistivity_Model_Dir.endswith('\\') else Resistivity_Model_Dir
                
                filename_seis1 = te_res_dir + str(i) + '.txt'
                #print(filename_seis1)
                raw_resistivity = np.loadtxt(filename_seis1, encoding='utf-8')
                
                filename_seis2 = te_ph_dir + str(i) + '.txt'
                #print(filename_seis2)
                raw_phase = np.loadtxt(filename_seis2, encoding='utf-8')
                
                filename1_label1 = res_mod_dir + 'zz' + str(i) + '.txt'
                #print(filename1_label1)
                raw_model = np.loadtxt(filename1_label1, encoding='utf-8')
                
                # 验证文件读取（前5个样本）
                if i < 5:
                    print(f"[DataLoad_Train] File reading verification (index {i}):")
                    print(f"  Resistivity file: {filename_seis1}")
                    print(f"    Raw data shape: {raw_resistivity.shape}, range: [{raw_resistivity.min():.6f}, {raw_resistivity.max():.6f}]")
                    print(f"  Phase file: {filename_seis2}")
                    print(f"    Raw data shape: {raw_phase.shape}, range: [{raw_phase.min():.6f}, {raw_phase.max():.6f}]")
                    print(f"  Model file: {filename1_label1}")
                    print(f"    Raw data shape: {raw_model.shape}, range: [{raw_model.min():.6f}, {raw_model.max():.6f}]")
                
                # 数据校验
                is_valid, invalid_files = validate_data(raw_resistivity, raw_phase, raw_model, i, 'TE')
                if is_valid:
                    # 数据有效，继续处理
                    # Step 1: Reshape raw_resistivity from 1D to 2D, then transpose
                    # raw_resistivity shape: (data_dim[0]*data_dim[1],) -> reshape -> (data_dim[0], data_dim[1]) -> .T -> (data_dim[1], data_dim[0])
                    train_data1 = np.reshape(raw_resistivity, (data_dim[0], data_dim[1])).T
                    train_data1 = np.log10(train_data1)
                    
                    # Step 2: Reshape raw_phase from 1D to 2D, then transpose
                    train_data2 = np.reshape(raw_phase, (data_dim[0], data_dim[1])).T
                    
                    # Step 3: Combine resistivity and phase: (2, data_dim[1], data_dim[0])
                    # IMPORTANT: Channel order must match prediction
                    # data1_set[0] = resistivity (log10), data1_set[1] = phase (no log10)
                    data1_set = np.array([train_data1, train_data2])
                    # Step 4: Transpose to (data_dim[1], data_dim[0], 2)
                    # After transpose: data1_set[:, :, 0] = resistivity (log10), data1_set[:, :, 1] = phase
                    data1_set = np.transpose(data1_set, (1, 2, 0))
                    
                    # Verify channel order for first sample
                    if i == 0:
                        print(f"[DataLoad_Train] Channel order verification (first sample):")
                        print(f"  Source files:")
                        print(f"    Resistivity file: {filename_seis1}")
                        print(f"    Phase file: {filename_seis2}")
                        print(f"  After processing:")
                        print(f"    data1_set[:, :, 0] (resistivity, log10): min={data1_set[:, :, 0].min():.6f}, max={data1_set[:, :, 0].max():.6f}")
                        print(f"      (Original resistivity range: [{raw_resistivity.min():.6f}, {raw_resistivity.max():.6f}] Ω·m)")
                        print(f"    data1_set[:, :, 1] (phase, no log10): min={data1_set[:, :, 1].min():.6f}, max={data1_set[:, :, 1].max():.6f}")
                        print(f"      (Original phase range: [{raw_phase.min():.6f}, {raw_phase.max():.6f}] degrees)")
                    
                    # 处理每个通道（视电阻率和相位，2通道）
                    for k in range(0, out_channels):
                        # Step 5: Extract channel k: (data_dim[1], data_dim[0])
                        data11_set = np.float32(data1_set[:, :, k])
                        # Step 6: Downsample: (data_dim[1], data_dim[0]) -> (H_dsp, W_dsp)
                        # Note: After .T and transpose(1,2,0), data1_set[:,:,k] has shape (data_dim[1], data_dim[0])
                        # After downsample: (H_dsp, W_dsp) where H_dsp = data_dim[1]//dsp, W_dsp = data_dim[0]//dsp
                        data11_set = block_reduce(data11_set, block_size=label_dsp_blk, func=decimate)
                        data_dsp_dim = data11_set.shape
                        data11_set = data11_set.reshape(1, data_dsp_dim[0] * data_dsp_dim[1])
                        # Debug: Print shape for first sample
                        if i == 0 and k == 0:
                            print(f"[DataLoad_Train] Channel {k} (resistivity) after downsample: shape={data_dsp_dim}, range=[{data11_set.min():.6f}, {data11_set.max():.6f}]")
                        if i == 0 and k == 1:
                            print(f"[DataLoad_Train] Channel {k} (phase) after downsample: shape={data_dsp_dim}, range=[{data11_set.min():.6f}, {data11_set.max():.6f}]")
                        if k == 0:
                            data_set = data11_set
                        else:
                            data_set = np.append(data_set, data11_set, axis=0)
                    
                    # train_label1 是模型的输入（电阻率模型）- 来自model_dim，用data_dsp_blk
                    train_label1 = raw_model
                    train_label1 = np.log10(train_label1)
                    train_label1 = block_reduce(train_label1, block_size=data_dsp_blk, func=np.max)
                    label_dsp_dim = train_label1.shape
                    train_label1 = train_label1.reshape(1, label_dsp_dim[0] * label_dsp_dim[1])
                    train_label1 = np.float32(train_label1)
                    
                    # train_set = 模型的输入（电阻率模型，1通道）
                    # label_set = 模型的输出（视电阻率和相位，2或4通道）
                    if valid_count == 0:
                        # 第一次初始化数组
                        train_set = train_label1  # 输入：电阻率模型
                        label_set = data_set      # 输出：视电阻率和相位
                    else:
                        # 使用np.append添加新数据
                        train_set = np.append(train_set, train_label1, axis=0)
                        label_set = np.append(label_set, data_set, axis=0)
                    valid_count += 1
                else:
                    # 数据无效，记录需要删除的文件
                    invalid_files_list.extend(invalid_files)
                    print(f"跳过无效数据(ID: {i}):")
                    for reason in invalid_files:
                        print(f"  - {reason}")
                    
        # Use out_channels parameter: output has 2 channels AND mode is TM
        elif out_channels == 2 and MT_Mode == 'TM':
            for i in range(start, start + train_size):
                # 每处理10个数据打印一次进度，方便监控
                if (i - start) % 10 == 0 and (i - start) > 0:
                    print(f"[DataLoad_Train] 已处理 {i - start} 个样本，有效数据: {valid_count}")
                    
                # 加载原始数据用于校验
                # 确保目录路径以斜杠结尾
                tm_res_dir = TM_Resistivity_Dir + '/' if not TM_Resistivity_Dir.endswith('/') and not TM_Resistivity_Dir.endswith('\\') else TM_Resistivity_Dir
                tm_ph_dir = TM_Phase_Dir + '/' if not TM_Phase_Dir.endswith('/') and not TM_Phase_Dir.endswith('\\') else TM_Phase_Dir
                res_mod_dir = Resistivity_Model_Dir + '/' if not Resistivity_Model_Dir.endswith('/') and not Resistivity_Model_Dir.endswith('\\') else Resistivity_Model_Dir
                
                filename_seis1 = tm_res_dir + str(i) + '.txt'
                #print(filename_seis1)
                raw_resistivity = np.loadtxt(filename_seis1, encoding='utf-8')
                
                filename_seis2 = tm_ph_dir + str(i) + '.txt'
                #print(filename_seis2)
                raw_phase = np.loadtxt(filename_seis2, encoding='utf-8')
                
                filename1_label1 = res_mod_dir + 'zz' +str(i) + '.txt'
                #print(filename1_label1)
                raw_model = np.loadtxt(filename1_label1, encoding='utf-8')
                
                # 验证文件读取（前5个样本）
                if i < 5:
                    print(f"[DataLoad_Train] File reading verification (index {i}):")
                    print(f"  Resistivity file: {filename_seis1}")
                    print(f"    Raw data shape: {raw_resistivity.shape}, range: [{raw_resistivity.min():.6f}, {raw_resistivity.max():.6f}]")
                    print(f"  Phase file: {filename_seis2}")
                    print(f"    Raw data shape: {raw_phase.shape}, range: [{raw_phase.min():.6f}, {raw_phase.max():.6f}]")
                    print(f"  Model file: {filename1_label1}")
                    print(f"    Raw data shape: {raw_model.shape}, range: [{raw_model.min():.6f}, {raw_model.max():.6f}]")
                
                # 数据校验
                is_valid, invalid_files = validate_data(raw_resistivity, raw_phase, raw_model, i, 'TM')
                if is_valid:
                    # 数据有效，继续处理
                    # Step 1: Reshape raw_resistivity from 1D to 2D, then transpose
                    # raw_resistivity shape: (data_dim[0]*data_dim[1],) -> reshape -> (data_dim[0], data_dim[1]) -> .T -> (data_dim[1], data_dim[0])
                    train_data1 = np.reshape(raw_resistivity, (data_dim[0], data_dim[1])).T
                    train_data1 = np.log10(train_data1)
                    
                    # Step 2: Reshape raw_phase from 1D to 2D, then transpose
                    train_data2 = np.reshape(raw_phase, (data_dim[0], data_dim[1])).T
                    
                    # Step 3: Combine resistivity and phase: (2, data_dim[1], data_dim[0])
                    # IMPORTANT: Channel order must match prediction
                    # data1_set[0] = resistivity (log10), data1_set[1] = phase (no log10)
                    data1_set = np.array([train_data1, train_data2])
                    # Step 4: Transpose to (data_dim[1], data_dim[0], 2)
                    # After transpose: data1_set[:, :, 0] = resistivity (log10), data1_set[:, :, 1] = phase
                    data1_set = np.transpose(data1_set, (1, 2, 0))
                    
                    # Verify channel order for first sample
                    if i == 0:
                        print(f"[DataLoad_Train] Channel order verification (first sample):")
                        print(f"  Source files:")
                        print(f"    Resistivity file: {filename_seis1}")
                        print(f"    Phase file: {filename_seis2}")
                        print(f"  After processing:")
                        print(f"    data1_set[:, :, 0] (resistivity, log10): min={data1_set[:, :, 0].min():.6f}, max={data1_set[:, :, 0].max():.6f}")
                        print(f"      (Original resistivity range: [{raw_resistivity.min():.6f}, {raw_resistivity.max():.6f}] Ω·m)")
                        print(f"    data1_set[:, :, 1] (phase, no log10): min={data1_set[:, :, 1].min():.6f}, max={data1_set[:, :, 1].max():.6f}")
                        print(f"      (Original phase range: [{raw_phase.min():.6f}, {raw_phase.max():.6f}] degrees)")
                    
                    # 处理每个通道（视电阻率和相位，2通道）
                    for k in range(0, out_channels):
                        # Step 5: Extract channel k: (data_dim[1], data_dim[0])
                        data11_set = np.float32(data1_set[:, :, k])
                        # Step 6: Downsample: (data_dim[1], data_dim[0]) -> (H_dsp, W_dsp)
                        # Note: After .T and transpose(1,2,0), data1_set[:,:,k] has shape (data_dim[1], data_dim[0])
                        # After downsample: (H_dsp, W_dsp) where H_dsp = data_dim[1]//dsp, W_dsp = data_dim[0]//dsp
                        data11_set = block_reduce(data11_set, block_size=label_dsp_blk, func=decimate)
                        data_dsp_dim = data11_set.shape
                        data11_set = data11_set.reshape(1, data_dsp_dim[0] * data_dsp_dim[1])
                        # Debug: Print shape for first sample
                        if i == 0 and k == 0:
                            print(f"[DataLoad_Train] Channel {k} (resistivity) after downsample: shape={data_dsp_dim}, range=[{data11_set.min():.6f}, {data11_set.max():.6f}]")
                        if i == 0 and k == 1:
                            print(f"[DataLoad_Train] Channel {k} (phase) after downsample: shape={data_dsp_dim}, range=[{data11_set.min():.6f}, {data11_set.max():.6f}]")
                        if k == 0:
                            data_set = data11_set
                        else:
                            data_set = np.append(data_set, data11_set, axis=0)
                    
                    train_label1 = raw_model
                    # 对电阻率label进行对数转换，保持与输入数据处理一致性
                    train_label1 = np.log10(train_label1)
                    train_label1 = block_reduce(train_label1, block_size=data_dsp_blk, func=np.max)
                    label_dsp_dim = train_label1.shape
                    train_label1 = train_label1.reshape(1, label_dsp_dim[0] * label_dsp_dim[1])
                    train_label1 = np.float32(train_label1)
                    
                    if valid_count == 0:
                        train_set = train_label1
                        label_set = data_set
                    else:
                        train_set = np.append(train_set, train_label1, axis=0)
                        label_set = np.append(label_set, data_set, axis=0)
                    valid_count += 1
                else:
                    # 数据无效，记录需要删除的文件
                    invalid_files_list.extend(invalid_files)
                    print(f"跳过无效数据(ID: {i})")
        # Use out_channels parameter: output has 4 channels (Both mode)
        elif out_channels == 4:
            for i in range(start, start + train_size):
                # 每处理10个数据打印一次进度，方便监控
                if (i - start) % 10 == 0 and (i - start) > 0:
                    print(f"[DataLoad_Train] 已处理 {i - start} 个样本，有效数据: {valid_count}")
                    
                # 加载原始数据用于校验
                # 确保目录路径以斜杠结尾
                te_res_dir = TE_Resistivity_Dir + '/' if not TE_Resistivity_Dir.endswith('/') and not TE_Resistivity_Dir.endswith('\\') else TE_Resistivity_Dir
                te_ph_dir = TE_Phase_Dir + '/' if not TE_Phase_Dir.endswith('/') and not TE_Phase_Dir.endswith('\\') else TE_Phase_Dir
                tm_res_dir = TM_Resistivity_Dir + '/' if not TM_Resistivity_Dir.endswith('/') and not TM_Resistivity_Dir.endswith('\\') else TM_Resistivity_Dir
                tm_ph_dir = TM_Phase_Dir + '/' if not TM_Phase_Dir.endswith('/') and not TM_Phase_Dir.endswith('\\') else TM_Phase_Dir
                res_mod_dir = Resistivity_Model_Dir + '/' if not Resistivity_Model_Dir.endswith('/') and not Resistivity_Model_Dir.endswith('\\') else Resistivity_Model_Dir
                
                filename_seis1 = te_res_dir + str(i) + '.txt'
                #print(filename_seis1)
                raw_te_resistivity = np.loadtxt(filename_seis1, encoding='utf-8')
                
                filename_seis2 = te_ph_dir + str(i) + '.txt'
                #print(filename_seis2)
                raw_te_phase = np.loadtxt(filename_seis2, encoding='utf-8')
                
                filename_seis3 = tm_res_dir + str(i) + '.txt'
                #print(filename_seis3)
                raw_tm_resistivity = np.loadtxt(filename_seis3, encoding='utf-8')
                
                filename_seis4 = tm_ph_dir + str(i) + '.txt'
                #print(filename_seis4)
                raw_tm_phase = np.loadtxt(filename_seis4, encoding='utf-8')
                
                filename1_label1 = res_mod_dir + 'zz' +str(i) + '.txt'
                #print(filename1_label1)
                raw_model = np.loadtxt(filename1_label1, encoding='utf-8')
                
                # 数据校验 - Both模式下，任何一个模式数据无效则整个数据无效
                te_valid, te_invalid_files = validate_data(raw_te_resistivity, raw_te_phase, raw_model, i, 'TE')
                tm_valid, tm_invalid_files = validate_data(raw_tm_resistivity, raw_tm_phase, raw_model, i, 'TM')
                
                if te_valid and tm_valid:
                    # 数据有效，继续处理
                    # Step 1: Reshape and process TE resistivity
                    train_data1 = np.reshape(raw_te_resistivity, (data_dim[0], data_dim[1])).T
                    train_data1 = np.log10(train_data1)
                    
                    # Step 2: Reshape and process TE phase
                    train_data2 = np.reshape(raw_te_phase, (data_dim[0], data_dim[1])).T
                    
                    # Step 3: Reshape and process TM resistivity
                    train_data3 = np.reshape(raw_tm_resistivity, (data_dim[0], data_dim[1])).T
                    train_data3 = np.log10(train_data3)
                    
                    # Step 4: Reshape and process TM phase
                    train_data4 = np.reshape(raw_tm_phase, (data_dim[0], data_dim[1])).T
                    
                    # Step 5: Combine all channels: (4, data_dim[1], data_dim[0])
                    # IMPORTANT: Channel order must match prediction
                    # data1_set[0] = TE resistivity (log10), data1_set[1] = TE phase,
                    # data1_set[2] = TM resistivity (log10), data1_set[3] = TM phase
                    data1_set = np.array([train_data1, train_data2, train_data3, train_data4])
                    # Step 6: Transpose to (data_dim[1], data_dim[0], 4)
                    data1_set = np.transpose(data1_set, (1, 2, 0))
                    
                    # 处理每个通道（4个通道：TE电阻率+TE相位+TM电阻率+TM相位）
                    for k in range(0, out_channels):
                        # Step 7: Extract channel k: (data_dim[1], data_dim[0])
                        data11_set = np.float32(data1_set[:, :, k])
                        # Step 8: Downsample: (data_dim[1], data_dim[0]) -> (H_dsp, W_dsp)
                        # Note: After .T and transpose(1,2,0), data1_set[:,:,k] has shape (data_dim[1], data_dim[0])
                        # After downsample: (H_dsp, W_dsp) where H_dsp = data_dim[1]//dsp, W_dsp = data_dim[0]//dsp
                        data11_set = block_reduce(data11_set, block_size=label_dsp_blk, func=decimate)
                        data_dsp_dim = data11_set.shape
                        data11_set = data11_set.reshape(1, data_dsp_dim[0] * data_dsp_dim[1])
                        if k == 0:
                            data_set = data11_set
                        else:
                            data_set = np.append(data_set, data11_set, axis=0)
                    
                    train_label1 = raw_model
                    # 对电阻率label进行对数转换，保持与输入数据处理一致性
                    train_label1 = np.log10(train_label1)
                    train_label1 = block_reduce(train_label1, block_size=data_dsp_blk, func=np.max)
                    label_dsp_dim = train_label1.shape
                    train_label1 = train_label1.reshape(1, label_dsp_dim[0] * label_dsp_dim[1])
                    train_label1 = np.float32(train_label1)
                    
                    if valid_count == 0:
                        train_set = train_label1
                        label_set = data_set
                    else:
                        train_set = np.append(train_set, train_label1, axis=0)
                        label_set = np.append(label_set, data_set, axis=0)
                    valid_count += 1
                else:
                    # 数据无效，记录需要删除的文件
                    if not te_valid:
                        invalid_files_list.extend(te_invalid_files)
                    if not tm_valid:
                        invalid_files_list.extend(tm_invalid_files)
                    print(f"跳过无效数据(ID: {i})")
        
        # 打印校验结果
        print(f"\n数据校验完成:")
        print(f"原始训练数据个数: {train_size}")
        print(f"有效训练数据个数: {valid_count}")
        print(f"无效数据个数: {train_size - valid_count}")
        
        if invalid_files_list:
            print(f"\n无效文件列表:")
            for file_info in invalid_files_list:
                print(f"- {file_info}")
        else:
            print("所有数据文件均有效")

        # 处理没有有效数据的情况
        if valid_count == 0:
            print("[DataLoad_Train] 警告: 没有有效数据，返回空数组和默认维度")
            # 初始化默认维度
            data_dsp_dim = (1, 1)
            label_dsp_dim = (1, 1)
            train_set = np.array([])
            label_set = np.array([])
        else:
            print(f"正在调整数据集形状...")
            # 调整train_set和label_set的形状为有效的训练数据个数
            # train_set: (valid_count, in_channels, ...) - 输入是电阻率模型（1通道）
            # label_set: (valid_count, out_channels, ...) - 输出是视电阻率和相位（2或4通道）
            train_set = train_set.reshape((valid_count, in_channels, label_dsp_dim[0] * label_dsp_dim[1]))
            label_set = label_set.reshape((valid_count, out_channels, data_dsp_dim[0] * data_dsp_dim[1]))
            
            print(f"数据集形状调整完成")
            print(f"训练集形状 (输入-电阻率模型): {train_set.shape}")
            print(f"标签集形状 (输出-视电阻率和相位): {label_set.shape}")
            
            # 验证最终的数据定义（第一个样本）
            if valid_count > 0:
                print(f"\n[DataLoad_Train] Final data verification (first sample):")
                print(f"  train_set[0] (输入-电阻率模型): shape={train_set[0].shape}, range=[{train_set[0].min():.6f}, {train_set[0].max():.6f}]")
                print(f"  label_set[0, 0, :] (输出-视电阻率, log10): range=[{label_set[0, 0, :].min():.6f}, {label_set[0, 0, :].max():.6f}]")
                if out_channels >= 2:
                    print(f"  label_set[0, 1, :] (输出-相位): range=[{label_set[0, 1, :].min():.6f}, {label_set[0, 1, :].max():.6f}]")
        
        print(f"数据加载耗时: {time.time() - start_time:.2f} 秒")

        # 根据用户要求，不返回归一化相关参数，只返回5个必要参数
        return train_set, label_set, data_dsp_dim, label_dsp_dim, valid_count
        
    except KeyboardInterrupt:
        print(f"\n[DataLoad_Train] 数据加载被用户中断!")
        print(f"已处理 {i - start + 1}/{train_size} 个样本")
        print(f"已加载 {valid_count} 个有效数据")
        raise
    except Exception as e:
        print(f"\n[DataLoad_Train] 数据加载出错: {str(e)}")
        raise


# 改进的降采样函数，直接返回块的平均值
def decimate(a, axis):
    """
    简化的降采样函数，直接返回块的平均值
    这比原来的实现更可靠，适用于block_reduce函数
    """
    return np.mean(a, axis=axis)


def updateFile(file, old_str, new_str):
    """
    将替换的字符串写到一个新的文件中，然后将原文件删除，新文件改为原来文件的名字
    :param file: 文件路径
    :param old_str: 需要替换的字符串
    :param new_str: 替换的字符串
    :return: None
    """
    with open(file, "r", encoding="utf-8") as f1, open("%s.bak" % file, "w", encoding="utf-8") as f2:
        for line in f1:
            if old_str in line:
                line = line.replace(old_str, new_str)
            f2.write(line)
    os.remove(file)
    os.rename("%s.bak" % file, file)


def normalize_data(data, min_val, max_val):
    """标准化数据到[0,1]区间
    
    参数:
    - data: 输入数据
    - min_val: 最小值
    - max_val: 最大值
    
    返回:
    - 标准化后的数据
    """
    return (data - min_val) / (max_val - min_val)

def denormalize_data(data, min_val, max_val):
    """反标准化数据
    
    参数:
    - data: 标准化后的数据
    - min_val: 原始最小值
    - max_val: 原始最大值
    
    返回:
    - 反标准化后的数据
    """
    return data * (max_val - min_val) + min_val

def get_normal_data(data1):
    """简单归一化函数，保留用于兼容性"""
    amin = 0.1
    amax = 1000000
    return normalize_data(data1, amin, amax)
