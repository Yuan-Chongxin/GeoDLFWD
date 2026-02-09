#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MT Training Script
Main training script for GeoDLFWD system
This script is called by the GUI to perform model training

Created: 2024
Author: GeoDLFWD Team
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Import configuration
try:
    import ParamConfig
except ImportError as e:
    print(f"Error: Cannot import ParamConfig. Error: {e}")
    sys.exit(1)

# Load file paths from JSON file if it exists
def load_file_paths():
    """Load file paths from JSON file created by GUI"""
    file_paths_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_file_paths.json')
    if os.path.exists(file_paths_json):
        try:
            with open(file_paths_json, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load file paths from JSON: {e}")
    return None

# Import data loading function
try:
    from func.DataLoad_Train import DataLoad_Train
except ImportError as e:
    print(f"Error: Cannot import DataLoad_Train. Error: {e}")
    sys.exit(1)

# Import model functions
try:
    from func.dinknet import DinkNet50
    from func.UnetModel import UnetModel
except ImportError as e:
    print(f"Error: Cannot import model functions. Error: {e}")
    sys.exit(1)

# Set device
device = torch.device(getattr(ParamConfig, 'Device', 'cuda') if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_model_architecture_from_checkpoint(model_path):
    """
    Infer in_channels and out_channels from checkpoint state_dict (DinkNet50/Unet).
    Returns (in_channels, out_channels) or None if detection fails.
    """
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state_dict, dict):
            return None
        # DinkNet50: firstconv.weight shape is [64, in_channels, 7, 7]
        if 'firstconv.weight' in state_dict:
            in_ch = state_dict['firstconv.weight'].shape[1]
        else:
            return None
        # finalconv3: [out_channels, 32, ...]
        if 'finalconv3.weight' in state_dict:
            out_ch = state_dict['finalconv3.weight'].shape[0]
        elif 'finalconv3.bias' in state_dict:
            out_ch = state_dict['finalconv3.bias'].shape[0]
        elif 'final.weight' in state_dict:  # UnetModel
            out_ch = state_dict['final.weight'].shape[0]
        else:
            return None
        return (in_ch, out_ch)
    except Exception:
        return None

def create_model(model_name, in_channels, out_channels):
    """Create model based on model name"""
    if 'DinkNet' in model_name or 'Dnet' in model_name:
        # DinkNet50 constructor takes (num_classes, num_channels), so swap the order
        model = DinkNet50(out_channels, in_channels)
    elif 'Unet' in model_name:
        # UnetModel constructor takes (n_classes, in_channels), so swap the order
        model = UnetModel(out_channels, in_channels)
    else:
        print(f"Warning: Unknown model name {model_name}, using DinkNet50 as default")
        # DinkNet50 constructor takes (num_classes, num_channels), so swap the order
        model = DinkNet50(out_channels, in_channels)
    
    return model.to(device)

def load_training_data():
    """Load training data using file paths from GUI or ParamConfig parameters
    
    Note: According to user specification:
    - Input: resistivity model (1 channel)
    - Output: apparent resistivity and phase
      - TE or TM mode: 2 channels (resistivity + phase)
      - Both mode: 4 channels (TE resistivity + TE phase + TM resistivity + TM phase)
    """
    # Initialize variables to prevent UnboundLocalError
    train_set = None
    label_set = None
    valid_count = 0
    
    # Get parameters from ParamConfig
    data_dim = getattr(ParamConfig, 'DataDim', [32, 32])
    model_dim = getattr(ParamConfig, 'ModelDim', [32, 32])
    data_dsp_blk = getattr(ParamConfig, 'data_dsp_blk', (1, 1))
    label_dsp_blk = getattr(ParamConfig, 'label_dsp_blk', (1, 1))
    
    # Initialize dimensions with default values to prevent UnboundLocalError
    data_dsp_dim = tuple(dim // blk for dim, blk in zip(data_dim, data_dsp_blk))
    label_dsp_dim = tuple(dim // blk for dim, blk in zip(model_dim, label_dsp_blk))
    
    MT_Mode = getattr(ParamConfig, 'MT_Mode', 'TM')
    
    # Channel configuration (MT forward modeling):
    # Input: resistivity model (1 channel)
    # Output:
    #   - TE or TM: 2 channels (apparent resistivity + phase)
    #   - TE&TM (Both): 4 channels (TE apparent resistivity + TE phase + TM apparent resistivity + TM phase)
    if MT_Mode == 'Both' or MT_Mode == 'TE&TM':
        in_channels = 1   # Input: resistivity model
        out_channels = 4  # Output: TE apparent resistivity, TE phase, TM apparent resistivity, TM phase
    else:
        # TE or TM mode: output has 2 channels
        in_channels = 1  # Input: resistivity model
        out_channels = 2  # Output: resistivity + phase
    
    print(f"Loading training data for {MT_Mode} mode...")
    print(f"Model input channels: {in_channels}, Model output channels: {out_channels}")
    
    # Try to load file paths from JSON file first
    file_paths = load_file_paths()
    
    # If file paths are available from GUI, use them directly
    if file_paths:
        print("Using file paths from Data Import tab...")
        try:
            from func.DataLoad_Train_FromFiles import DataLoad_Train_FromFiles
            
            # Get file lists based on MT_Mode
            if MT_Mode == 'TE':
                model_files = file_paths.get('model_files', [])
                resistivity_files = file_paths.get('te_resistivity_files', [])
                phase_files = file_paths.get('te_phase_files', [])
                result = DataLoad_Train_FromFiles(
                    model_files=model_files,
                    resistivity_files=resistivity_files,
                    phase_files=phase_files,
                    data_dim=data_dim,
                    model_dim=model_dim,
                    data_dsp_blk=data_dsp_blk,
                    label_dsp_blk=label_dsp_blk,
                    MT_Mode=MT_Mode
                )
                if result and len(result) >= 5 and result[0] is not None:
                    train_set, label_set, data_dsp_dim, label_dsp_dim, valid_count = result
                    # Ensure dimensions are not None
                    if data_dsp_dim is None:
                        data_dsp_dim = tuple(dim // blk for dim, blk in zip(data_dim, data_dsp_blk))
                    if label_dsp_dim is None:
                        label_dsp_dim = tuple(dim // blk for dim, blk in zip(model_dim, label_dsp_blk))
                else:
                    raise ValueError("DataLoad_Train_FromFiles returned invalid result")
            elif MT_Mode == 'TM':
                model_files = file_paths.get('model_files', [])
                resistivity_files = file_paths.get('tm_resistivity_files', [])
                phase_files = file_paths.get('tm_phase_files', [])
                result = DataLoad_Train_FromFiles(
                    model_files=model_files,
                    resistivity_files=resistivity_files,
                    phase_files=phase_files,
                    data_dim=data_dim,
                    model_dim=model_dim,
                    data_dsp_blk=data_dsp_blk,
                    label_dsp_blk=label_dsp_blk,
                    MT_Mode=MT_Mode
                )
                if result and len(result) >= 5 and result[0] is not None:
                    train_set, label_set, data_dsp_dim, label_dsp_dim, valid_count = result
                    # Ensure dimensions are not None
                    if data_dsp_dim is None:
                        data_dsp_dim = tuple(dim // blk for dim, blk in zip(data_dim, data_dsp_blk))
                    if label_dsp_dim is None:
                        label_dsp_dim = tuple(dim // blk for dim, blk in zip(model_dim, label_dsp_blk))
                else:
                    raise ValueError("DataLoad_Train_FromFiles returned invalid result")
            else:  # Both mode
                model_files = file_paths.get('model_files', [])
                te_resistivity_files = file_paths.get('te_resistivity_files', [])
                te_phase_files = file_paths.get('te_phase_files', [])
                tm_resistivity_files = file_paths.get('tm_resistivity_files', [])
                tm_phase_files = file_paths.get('tm_phase_files', [])
                result = DataLoad_Train_FromFiles(
                    model_files=model_files,
                    te_resistivity_files=te_resistivity_files,
                    te_phase_files=te_phase_files,
                    tm_resistivity_files=tm_resistivity_files,
                    tm_phase_files=tm_phase_files,
                    data_dim=data_dim,
                    model_dim=model_dim,
                    data_dsp_blk=data_dsp_blk,
                    label_dsp_blk=label_dsp_blk,
                    MT_Mode=MT_Mode
                )
                if result and len(result) >= 5 and result[0] is not None:
                    train_set, label_set, data_dsp_dim, label_dsp_dim, valid_count = result
                    # Ensure dimensions are not None
                    if data_dsp_dim is None:
                        data_dsp_dim = tuple(dim // blk for dim, blk in zip(data_dim, data_dsp_blk))
                    if label_dsp_dim is None:
                        label_dsp_dim = tuple(dim // blk for dim, blk in zip(model_dim, label_dsp_blk))
                else:
                    raise ValueError("DataLoad_Train_FromFiles returned invalid result")
        except (ImportError, Exception) as e:
            print(f"Warning: Cannot use file-based loading: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to directory-based loading...")
            file_paths = None
            # Reset variables (dimensions are already initialized above)
            train_set = None
            label_set = None
            valid_count = 0
            # Ensure dimensions are still defined (they are initialized at function start)
            # No need to recalculate, they're already set
    
    # Fall back to directory-based loading if file paths are not available
    if not file_paths:
        print("Using directory-based loading from ParamConfig...")
        train_size = int(getattr(ParamConfig, 'TrainSize', 0.8) * 100)
        TE_Resistivity_Dir = getattr(ParamConfig, 'TE_Resistivity_Dir', 'data/TE/resistivity')
        TE_Phase_Dir = getattr(ParamConfig, 'TE_Phase_Dir', 'data/TE/phase')
        TM_Resistivity_Dir = getattr(ParamConfig, 'TM_Resistivity_Dir', 'data/TM/resistivity')
        TM_Phase_Dir = getattr(ParamConfig, 'TM_Phase_Dir', 'data/TM/phase')
        Resistivity_Model_Dir = getattr(ParamConfig, 'Resistivity_Model_Dir', 'data/models')
        start = 0
        
        try:
            train_set, label_set, data_dsp_dim, label_dsp_dim, valid_count = DataLoad_Train(
                train_size=train_size,
                train_data_dir='',
                data_dim=data_dim,
                out_channels=out_channels,
                model_dim=model_dim,
                data_dsp_blk=data_dsp_blk,
                label_dsp_blk=label_dsp_blk,
                start=start,
                datafilename='',
                dataname='',
                truthfilename='',
                truthname='',
                TE_Resistivity_Dir=TE_Resistivity_Dir,
                TE_Phase_Dir=TE_Phase_Dir,
                TM_Resistivity_Dir=TM_Resistivity_Dir,
                TM_Phase_Dir=TM_Phase_Dir,
                Resistivity_Model_Dir=Resistivity_Model_Dir,
                MT_Mode=MT_Mode
            )
        except Exception as e:
            print(f"Error loading training data: {e}")
            import traceback
            traceback.print_exc()
            # Calculate default dimensions even on error
            data_dim = getattr(ParamConfig, 'DataDim', [32, 32])
            model_dim = getattr(ParamConfig, 'ModelDim', [32, 32])
            data_dsp_blk = getattr(ParamConfig, 'data_dsp_blk', (1, 1))
            label_dsp_blk = getattr(ParamConfig, 'label_dsp_blk', (1, 1))
            data_dsp_dim = tuple(dim // blk for dim, blk in zip(data_dim, data_dsp_blk))
            label_dsp_dim = tuple(dim // blk for dim, blk in zip(model_dim, label_dsp_blk))
            return None, None, None, None, 0, data_dsp_dim, label_dsp_dim
    
    print(f"Successfully loaded {valid_count} valid training samples")
    
    # Ensure data_dsp_dim and label_dsp_dim are defined and not None
    # They should be returned from DataLoad functions, but check just in case
    if 'data_dsp_dim' not in locals() or data_dsp_dim is None or 'label_dsp_dim' not in locals() or label_dsp_dim is None:
        # Calculate from ParamConfig if not returned from DataLoad functions or if they are None
        data_dim = getattr(ParamConfig, 'DataDim', [32, 32])
        model_dim = getattr(ParamConfig, 'ModelDim', [32, 32])
        data_dsp_blk = getattr(ParamConfig, 'data_dsp_blk', (1, 1))
        label_dsp_blk = getattr(ParamConfig, 'label_dsp_blk', (1, 1))
        data_dsp_dim = tuple(dim // blk for dim, blk in zip(data_dim, data_dsp_blk))
        label_dsp_dim = tuple(dim // blk for dim, blk in zip(model_dim, label_dsp_blk))
        print(f"Calculated dimensions from ParamConfig: data_dsp_dim={data_dsp_dim}, label_dsp_dim={label_dsp_dim}")
    
    # Final check: ensure train_set and label_set are valid
    if train_set is None or label_set is None:
        print("Error: Failed to load training data, train_set or label_set is None")
        return None, None, None, None, 0, data_dsp_dim, label_dsp_dim
    
    return train_set, label_set, in_channels, out_channels, valid_count, data_dsp_dim, label_dsp_dim

def train_model():
    """Main training function"""
    # Get training parameters from ParamConfig
    epochs = getattr(ParamConfig, 'Epochs', 200)
    batch_size = getattr(ParamConfig, 'BatchSize', 20)
    learning_rate = getattr(ParamConfig, 'LearnRate', 0.001)
    if learning_rate < 1e-6:
        print(f"WARNING: Learning rate {learning_rate} is very small. Training may not converge.")
        print(f"Suggested: 1e-4 to 1e-2 for Adam. Current default 0.001.")
    train_size = getattr(ParamConfig, 'TrainSize', 0.8)
    val_size = getattr(ParamConfig, 'ValSize', 0.2)
    early_stop = getattr(ParamConfig, 'EarlyStop', 30)  # 增加早停耐心值，确保与学习率衰减协调
    display_step = getattr(ParamConfig, 'DisplayStep', 10)
    model_name = getattr(ParamConfig, 'ModelName', 'DinkNet')
    models_dir = getattr(ParamConfig, 'ModelsDir', 'models/')
    
    print("=" * 60)
    print("GeoDLFWD Training Script")
    print("=" * 60)
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load training data
    # Initialize dimensions with default values to prevent UnboundLocalError
    data_dim = getattr(ParamConfig, 'DataDim', [32, 32])
    model_dim = getattr(ParamConfig, 'ModelDim', [32, 32])
    data_dsp_blk = getattr(ParamConfig, 'data_dsp_blk', (1, 1))
    label_dsp_blk = getattr(ParamConfig, 'label_dsp_blk', (1, 1))
    data_dsp_dim = tuple(dim // blk for dim, blk in zip(data_dim, data_dsp_blk))
    label_dsp_dim = tuple(dim // blk for dim, blk in zip(model_dim, label_dsp_blk))
    
    result = load_training_data()
    if result is None or len(result) < 7:
        print("Error: Failed to load training data")
        # Dimensions are already initialized above
        return
    
    try:
        train_set, label_set, in_channels, out_channels, valid_count, data_dsp_dim, label_dsp_dim = result
    except (ValueError, TypeError) as e:
        print(f"Error unpacking training data result: {e}")
        print(f"Result type: {type(result)}, Result length: {len(result) if result else 'None'}")
        # Dimensions are already initialized above
        return
    
    if train_set is None or label_set is None:
        print("Error: Failed to load training data")
        return
    
    if valid_count == 0:
        print("Error: No valid training samples found")
        print("Please check your data files or validation settings")
        return
    
    # Split data into training and validation sets using random shuffle
    # This ensures that training and validation sets have similar data distributions
    total_samples = valid_count
    train_samples = int(total_samples * train_size)
    val_samples = total_samples - train_samples
    
    # Create indices and shuffle them randomly
    indices = np.arange(total_samples)
    np.random.seed(42)  # Set random seed for reproducibility
    np.random.shuffle(indices)
    
    # Split indices into training and validation sets
    train_indices = indices[:train_samples]
    val_indices = indices[train_samples:train_samples + val_samples]
    
    # Use shuffled indices to split data, ensuring train_data[i] and train_labels[i] are paired
    train_data = train_set[train_indices]
    train_labels = label_set[train_indices]
    val_data = train_set[val_indices]
    val_labels = label_set[val_indices]
    
    print(f"Training samples: {train_samples}, Validation samples: {val_samples}")
    
    # Convert to tensors
    train_data_tensor = torch.FloatTensor(train_data).to(device)
    train_labels_tensor = torch.FloatTensor(train_labels).to(device)
    val_data_tensor = torch.FloatTensor(val_data).to(device) if val_samples > 0 else None
    val_labels_tensor = torch.FloatTensor(val_labels).to(device) if val_samples > 0 else None
    
    # Get dimensions from ParamConfig (for reference, dimensions should already be calculated above)
    data_dim = getattr(ParamConfig, 'DataDim', [32, 32])    # Data dimension (apparent resistivity & phase)
    model_dim = getattr(ParamConfig, 'ModelDim', [32, 32])  # Model dimension (resistivity model)
    # data_dsp_dim and label_dsp_dim should already be available from load_training_data() return value
    
    # Validate spatial dimensions (input and output must match for FCN)
    if data_dsp_dim != label_dsp_dim:
        print(f"WARNING: Input spatial {label_dsp_dim} != output spatial {data_dsp_dim}. "
              f"Ensure data_dim=model_dim and data_dsp_blk=label_dsp_blk for FCN compatibility.")
    
    # Print data shapes for debugging
    print(f"Before reshape - train_data_tensor shape: {train_data_tensor.shape}")
    print(f"Before reshape - train_labels_tensor shape: {train_labels_tensor.shape}")
    
    # Reshape model input: resistivity model (train_data_tensor)
    # train_set shape from DataLoad: (valid_count, 1, label_dsp_dim[0] * label_dsp_dim[1])
    # Need to reshape to: (batch, in_channels, label_dsp_dim[0], label_dsp_dim[1])
    # If tensor shape is (batch, 1, H*W), we need to flatten the last dimension first
    if len(train_data_tensor.shape) == 3 and train_data_tensor.shape[1] == in_channels:
        # Flatten the last dimension: (batch, 1, H*W) -> (batch, H*W)
        train_data_tensor = train_data_tensor.contiguous().view(train_data_tensor.shape[0], -1)
    # Now reshape to (batch, in_channels, H, W)
    train_data_tensor = train_data_tensor.contiguous().view(-1, in_channels, label_dsp_dim[0], label_dsp_dim[1])
    
    # Reshape model output: apparent resistivity and phase (train_labels_tensor)
    # label_set shape from DataLoad: (valid_count, out_channels, data_dsp_dim[0] * data_dsp_dim[1])
    # Need to reshape to: (batch, out_channels, data_dsp_dim[0], data_dsp_dim[1])
    # If tensor shape is (batch, out_channels, H*W), we need to reshape the last dimension
    if len(train_labels_tensor.shape) == 3 and train_labels_tensor.shape[1] == out_channels:
        # Reshape: (batch, out_channels, H*W) -> (batch, out_channels, H, W)
        train_labels_tensor = train_labels_tensor.contiguous().view(-1, out_channels, data_dsp_dim[0], data_dsp_dim[1])
    else:
        # Fallback: flatten and reshape
        train_labels_tensor = train_labels_tensor.contiguous().view(-1, out_channels * data_dsp_dim[0] * data_dsp_dim[1])
        train_labels_tensor = train_labels_tensor.view(-1, out_channels, data_dsp_dim[0], data_dsp_dim[1])
    
    if val_data_tensor is not None:
        # Apply same reshaping logic to validation data
        if len(val_data_tensor.shape) == 3 and val_data_tensor.shape[1] == in_channels:
            val_data_tensor = val_data_tensor.contiguous().view(val_data_tensor.shape[0], -1)
        val_data_tensor = val_data_tensor.contiguous().view(-1, in_channels, label_dsp_dim[0], label_dsp_dim[1])
        
        if len(val_labels_tensor.shape) == 3 and val_labels_tensor.shape[1] == out_channels:
            val_labels_tensor = val_labels_tensor.contiguous().view(-1, out_channels, data_dsp_dim[0], data_dsp_dim[1])
        else:
            val_labels_tensor = val_labels_tensor.contiguous().view(-1, out_channels * data_dsp_dim[0] * data_dsp_dim[1])
            val_labels_tensor = val_labels_tensor.view(-1, out_channels, data_dsp_dim[0], data_dsp_dim[1])
    
    print(f"After reshape - train_data_tensor shape: {train_data_tensor.shape}")
    print(f"After reshape - train_labels_tensor shape: {train_labels_tensor.shape}")
    # Data range check: resistivity log10 ~ [-1,3], phase ~ [0,90]. Large scale imbalance can hurt MSE.
    with torch.no_grad():
        for c in range(out_channels):
            ch_min = train_labels_tensor[:, c].min().item()
            ch_max = train_labels_tensor[:, c].max().item()
            ch_mean = train_labels_tensor[:, c].mean().item()
            name = "resistivity(log10)" if c % 2 == 0 else "phase"
            print(f"  Label ch{c} ({name}): min={ch_min:.4f}, max={ch_max:.4f}, mean={ch_mean:.4f}")
        if out_channels >= 2:
            r_std = train_labels_tensor[:, 0].std().item()
            p_std = train_labels_tensor[:, 1].std().item()
            if p_std > r_std * 5:
                print(f"  WARNING: Phase std ({p_std:.2f}) >> resistivity std ({r_std:.2f}). MSE may be dominated by phase.")
    
    # MT model: input=resistivity model (1 ch), output=apparent resistivity+phase (2 ch). Always use this.
    ReUse = getattr(ParamConfig, 'ReUse', False)
    pretrained_path = None
    if ReUse:
        if hasattr(ParamConfig, 'PreModelPath') and ParamConfig.PreModelPath:
            p = ParamConfig.PreModelPath
            pretrained_path = os.path.abspath(p) if not os.path.isabs(p) else p
        if not pretrained_path or not os.path.exists(pretrained_path):
            # Fallback: try models_dir + PreModel.pth or ModelName_best.pth
            premodelname = getattr(ParamConfig, 'PreModel', 'model')
            for candidate in [os.path.join(models_dir, f"{premodelname}.pth"),
                             os.path.join(models_dir, f"{model_name}_best.pth")]:
                cand_abs = os.path.abspath(candidate)
                if os.path.exists(cand_abs):
                    pretrained_path = cand_abs
                    print(f"Using pre-trained model (fallback): {pretrained_path}")
                    break
    
    # Create model: (1,2) for TE/TM, (1,4) for TE&TM - input=resistivity model, output=apparent resistivity+phase
    model = create_model(model_name, in_channels, out_channels)
    out_desc = "TE apparent resistivity+phase" if out_channels == 2 else "TE+TM apparent resistivity+phase (4 ch)"
    print(f"Model created: {model_name} with {in_channels} input ch (resistivity model), {out_channels} output ch ({out_desc})")
    
    # Setup optimizer and loss function
    weight_decay = getattr(ParamConfig, 'WeightDecay', 1e-4)  # 添加L2正则化参数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Use per-channel weights to balance resistivity (log10, small) vs phase (0-90, large)
    channel_weights = getattr(ParamConfig, 'LabelChannelWeights', None)
    if channel_weights is None and out_channels >= 2:
        # Phase (0-90) dominates resistivity (log10 ~-1~3). Weight phase lower.
        channel_weights = [1.0, 0.05] if out_channels == 2 else [1.0, 0.05, 1.0, 0.05]
        print(f"Using channel weights {channel_weights[:out_channels]} to balance resistivity/phase in loss")
    if channel_weights is not None and len(channel_weights) >= out_channels:
        def weighted_mse(output, target):
            loss = 0.0
            for c in range(out_channels):
                w = channel_weights[c] if c < len(channel_weights) else 1.0
                loss = loss + w * ((output[:, c] - target[:, c]) ** 2).mean()
            return loss / out_channels
        criterion = weighted_mse
    else:
        criterion = nn.MSELoss()
    
    # 设置学习率衰减策略
    lr_patience = getattr(ParamConfig, 'LRPatience', 10)  # 学习率衰减的耐心值
    lr_factor = getattr(ParamConfig, 'LRFactor', 0.5)  # 学习率衰减因子
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=lr_patience, factor=lr_factor, verbose=True
    )
    
    # Load pretrained model if ReUse is True (partial load if architecture differs)
    pretrained_loaded = False
    if ReUse and pretrained_path and os.path.exists(pretrained_path):
        try:
            print(f"Loading pretrained model from: {pretrained_path}")
            detected = get_model_architecture_from_checkpoint(pretrained_path)
            if detected and (detected[0] != in_channels or detected[1] != out_channels):
                print(f"WARNING: Pretrained has architecture in={detected[0]}, out={detected[1]} but MT requires in=1, out={out_channels}")
                print(f"Pretrained may be from different task. Loading only matching layers (firstconv/finalconv will be random).")
            checkpoint = torch.load(pretrained_path, map_location=device)
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
            else:
                state_dict = checkpoint
            model_state_dict = model.state_dict()
            filtered = {}
            for key, value in state_dict.items():
                new_key = key.replace('module.', '') if key.startswith('module.') else key
                if new_key in model_state_dict and model_state_dict[new_key].shape == value.shape:
                    filtered[new_key] = value
                elif new_key in model_state_dict:
                    print(f"Warning: Shape mismatch '{new_key}' - skipping")
            model.load_state_dict(filtered, strict=False)
            pretrained_loaded = True
            n_total = len(model_state_dict)
            n_loaded = len(filtered)
            print(f"Loaded pretrained: {n_loaded}/{n_total} parameters")
            if n_loaded == n_total:
                print("Continue training: all weights loaded successfully")
            elif n_loaded > 0:
                print("Continue training: partial load (some layers had shape mismatch)")
        except Exception as e:
            print(f"Error loading pretrained: {e}")
            import traceback
            traceback.print_exc()
            pretrained_loaded = False
    elif ReUse and (not pretrained_path or not os.path.exists(pretrained_path)):
        print(f"Warning: Pretrained file not found. Starting from scratch.")
    elif not ReUse:
        print("Starting training from scratch")
    
    # Create DataLoaders
    try:
        # Create training dataset and loader
        train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Create validation dataset and loader if we have validation samples
        val_loader = None
        if val_samples > 0 and val_data_tensor is not None and val_labels_tensor is not None:
            val_dataset = TensorDataset(val_data_tensor, val_labels_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    except Exception as e:
        print(f"Error creating DataLoaders: {e}")
        return
    
    # Check if DataLoaders are valid
    if not train_loader:
        print("Error: No valid train_loader created")
        return
    
    # Training loop
    best_val_loss = float('inf')
    no_improve_count = 0
    
    print("\nStarting training...")
    print("-" * 60)
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_loss_data = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            if 'Unet' in model_name or 'unet' in model_name:
                # UnetModel.forward needs label_dsp_dim parameter
                # Use data_dim as a fallback for label_dsp_dim
                output = model(data, getattr(ParamConfig, 'DataDim', [32, 32]))
            else:
                # DinkNet50.forward only needs data parameter
                output = model(data)
            loss = criterion(output, target)
            
            # Use loss directly as the total loss
            # Removed unused physics_loss (loss2)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_loss_data += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_data_loss = train_loss_data / len(train_loader)
        
        # Validation phase
        val_loss = 0.0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for data, target in val_loader:
                    if 'Unet' in model_name or 'unet' in model_name:
                        # UnetModel.forward needs label_dsp_dim parameter
                        # Use data_dim as a fallback for label_dsp_dim
                        output = model(data, getattr(ParamConfig, 'DataDim', [32, 32]))
                    else:
                        # DinkNet50.forward only needs data parameter
                        output = model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
        else:
            avg_val_loss = avg_train_loss
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print progress (format expected by GUI - GeoDLFWD.py)
        if epoch % display_step == 0 or epoch == 1:
            # Output format: "Epoch: X finished, Loss: X.XXXXXX, Data Loss: X.XXXXXX, Time: X.XXs"
            # This format is expected by GeoDLFWD.py TrainingThread
            print(f"Epoch: {epoch} finished, Loss: {avg_train_loss:.6f}, Data Loss: {avg_train_data_loss:.6f}, Time: {epoch_time:.2f}s")
            sys.stdout.flush()  # Ensure output is flushed immediately
            
            # Output validation loss for GUI to parse
            if val_data_tensor is not None:
                print(f"[MT_TRAIN VALIDATION] Validation Loss: {avg_val_loss:.6f}")
                sys.stdout.flush()
        
        # 更新学习率
        if val_data_tensor is not None:
            scheduler.step(avg_val_loss)
        
        # Save model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_count = 0
            
            # Save best model
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, f"{model_name}_best.pth")
            torch.save(model.state_dict(), model_path)
        else:
            no_improve_count += 1
        
        # Early stopping
        if early_stop > 0 and no_improve_count >= early_stop:
            print(f"Early stopping triggered after {epoch} epochs (no improvement for {early_stop} epochs)")
            break
    
    print("-" * 60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    try:
        train_model()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
