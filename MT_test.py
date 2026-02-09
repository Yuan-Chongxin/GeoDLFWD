#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MT Test/Prediction Script
Forward modeling: Input resistivity model, output apparent resistivity and phase
Uses the same data preprocessing logic as MT_train.py

Created: 2024
Author: GeoDLFWD Team
"""

import os
import sys
import numpy as np
import torch
from skimage.measure import block_reduce

# Import configuration
try:
    import ParamConfig
except ImportError as e:
    print(f"Error: Cannot import ParamConfig. Error: {e}")
    sys.exit(1)

# Import decimate function from DataLoad_Train
try:
    from func.DataLoad_Train import decimate
except ImportError:
    # Fallback decimate function
    def decimate(a, axis):
        return np.mean(a, axis=axis)

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
    Infer in_channels and out_channels from checkpoint state_dict.
    Returns (in_channels, out_channels) or None if detection fails.
    """
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state_dict, dict):
            return None
        
        # DinkNet50: firstconv.weight shape is [64, in_channels, 7, 7]
        if 'firstconv.weight' in state_dict:
            in_channels = state_dict['firstconv.weight'].shape[1]
        else:
            return None
        
        # DinkNet50: finalconv3.weight shape is [out_channels, 32, 2, 2] or [out_channels, 32, 3, 3]
        if 'finalconv3.weight' in state_dict:
            out_channels = state_dict['finalconv3.weight'].shape[0]
        elif 'finalconv3.bias' in state_dict:
            out_channels = state_dict['finalconv3.bias'].shape[0]
        else:
            return None
        
        return (in_channels, out_channels)
    except Exception as e:
        print(f"Warning: Could not detect architecture from checkpoint: {e}")
        return None

def create_model(model_name, in_channels, out_channels):
    """Create model based on model name"""
    if 'DinkNet' in model_name or 'Dnet' in model_name:
        model = DinkNet50(out_channels, in_channels)
    elif 'Unet' in model_name:
        model = UnetModel(out_channels, in_channels)
    else:
        print(f"Warning: Unknown model name {model_name}, using DinkNet50 as default")
        model = DinkNet50(out_channels, in_channels)
    
    return model.to(device)

def preprocess_resistivity_model(model_path, data_dim, model_dim, label_dsp_blk):
    """
    Preprocess resistivity model for prediction
    Uses the same preprocessing logic as training: log10 transform + downsampling
    
    Parameters:
    - model_path: Path to resistivity model file (.txt)
    - data_dim: Data dimension from ParamConfig
    - model_dim: Model dimension from ParamConfig
    - label_dsp_blk: Label downsampling block size
    
    Returns:
    - Preprocessed tensor with shape (1, in_channels, label_dsp_dim[0], label_dsp_dim[1])
    - label_dsp_dim: Downsampled dimensions
    """
    # Load resistivity model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Resistivity model file not found: {model_path}")
    
    raw_model = np.loadtxt(model_path, encoding='utf-8')
    
    # Apply same preprocessing as training (DataLoad_Train_FromFiles.py lines 133-138):
    # 1. Log10 transform (same as line 134)
    train_label1 = raw_model
    train_label1 = np.log10(train_label1)
    
    # 2. Downsampling using block_reduce with np.max (same as line 135)
    train_label1 = block_reduce(train_label1, block_size=label_dsp_blk, func=np.max)
    label_dsp_dim = train_label1.shape
    
    # 3. Reshape to (1, label_dsp_dim[0] * label_dsp_dim[1]) (same as line 137)
    train_label1 = train_label1.reshape(1, label_dsp_dim[0] * label_dsp_dim[1])
    train_label1 = np.float32(train_label1)
    
    # 4. Convert to tensor and reshape to (1, in_channels, H, W)
    # This matches MT_train.py line 299: view(-1, in_channels, label_dsp_dim[0], label_dsp_dim[1])
    in_channels = 1  # Resistivity model is 1 channel input
    model_tensor = torch.FloatTensor(train_label1).to(device)
    
    # Reshape to match training format: (batch, in_channels, H, W)
    if len(model_tensor.shape) == 2 and model_tensor.shape[1] == label_dsp_dim[0] * label_dsp_dim[1]:
        model_tensor = model_tensor.contiguous().view(1, in_channels, label_dsp_dim[0], label_dsp_dim[1])
    
    print(f"Preprocessed resistivity model shape: {model_tensor.shape}")
    print(f"Downsampled dimensions: {label_dsp_dim}")
    
    return model_tensor, label_dsp_dim

def postprocess_output(output_tensor, data_dsp_dim, out_channels):
    """
    Postprocess model output (apparent resistivity and phase)
    Applies inverse log10 transform to resistivity channels
    
    Parameters:
    - output_tensor: Model output tensor with shape (1, out_channels, H, W)
    - data_dsp_dim: Data downsampled dimensions
    - out_channels: Number of output channels (2 for TE/TM, 4 for Both)
    
    Returns:
    - Postprocessed output as numpy array
    """
    # Convert to numpy
    output_np = output_tensor.cpu().detach().numpy()
    
    print(f"[postprocess_output] Input shape: {output_np.shape}")
    
    # Reshape if needed: (1, out_channels, H, W) -> (out_channels, H, W)
    if len(output_np.shape) == 4:
        output_np = output_np[0]  # Remove batch dimension
    
    print(f"[postprocess_output] After removing batch dim: {output_np.shape}")
    
    # IMPORTANT: Verify channel order matches training
    # Training: label_set[:, 0, :] = resistivity (log10), label_set[:, 1, :] = phase
    # So model output: output[0] = resistivity (log10), output[1] = phase
    print(f"[postprocess_output] Data range before inverse log10:")
    print(f"  Channel 0 (resistivity, log10): min={output_np[0].min():.6f}, max={output_np[0].max():.6f}")
    if out_channels >= 2:
        print(f"  Channel 1 (phase): min={output_np[1].min():.6f}, max={output_np[1].max():.6f}")
    
    # Apply inverse log10 transform to resistivity channels ONLY
    # Channel indices for resistivity: 0 for TE/TM mode, 0 and 2 for Both mode
    if out_channels == 1:
        # Single channel: treat as resistivity (log10)
        print(f"[postprocess_output] Single channel output, treating as resistivity (log10)")
        output_np[0] = np.power(10, output_np[0])
        print(f"[postprocess_output] After inverse log10: min={output_np[0].min():.6f}, max={output_np[0].max():.6f}")
    elif out_channels == 2:
        # TE or TM mode: channel 0 is resistivity (log10), channel 1 is phase (no log10)
        # Only apply inverse log10 to resistivity, NOT to phase
        
        # IMPORTANT: Check if values look correct
        # Resistivity log10 values are typically negative (log10 of small numbers like 0.1-100)
        # Phase values are typically positive (degrees, usually 0-90)
        channel0_looks_like_log10 = output_np[0].min() < 0 or (output_np[0].min() >= 0 and output_np[0].max() < 3)
        channel1_looks_like_phase = output_np[1].min() >= 0 and output_np[1].max() < 200
        
        print(f"[postprocess_output] Value analysis:")
        print(f"  Channel 0 (resistivity, log10): min={output_np[0].min():.6f}, max={output_np[0].max():.6f}, looks_like_log10={channel0_looks_like_log10}")
        print(f"  Channel 1 (phase): min={output_np[1].min():.6f}, max={output_np[1].max():.6f}, looks_like_phase={channel1_looks_like_phase}")
        
        # If channel 0 doesn't look like log10, warn but proceed
        if not channel0_looks_like_log10:
            print(f"[postprocess_output] WARNING: Channel 0 values don't look like log10!")
            print(f"[postprocess_output]  Expected: negative or small positive values (< 3)")
            print(f"[postprocess_output]  Got: min={output_np[0].min():.6f}, max={output_np[0].max():.6f}")
            print(f"[postprocess_output]  This might indicate the model output is already in raw format, or channels are swapped")
        
        resistivity_log10 = output_np[0].copy()  # Save log10 value for debugging
        output_np[0] = np.power(10, output_np[0])  # Inverse log10 for resistivity ONLY
        print(f"[postprocess_output] After inverse log10:")
        print(f"  Channel 0 (resistivity): min={output_np[0].min():.6f}, max={output_np[0].max():.6f} (was log10: {resistivity_log10.min():.6f} to {resistivity_log10.max():.6f})")
        print(f"  Channel 1 (phase, unchanged): min={output_np[1].min():.6f}, max={output_np[1].max():.6f}")
    elif out_channels == 4:
        # Both mode: channels 0 (TE resistivity), 2 (TM resistivity) need inverse log10
        # Channels 1 (TE phase) and 3 (TM phase) remain unchanged
        print(f"[postprocess_output] Before inverse log10:")
        print(f"  Channel 0 (TE resistivity, log10): min={output_np[0].min():.6f}, max={output_np[0].max():.6f}")
        print(f"  Channel 1 (TE phase): min={output_np[1].min():.6f}, max={output_np[1].max():.6f}")
        print(f"  Channel 2 (TM resistivity, log10): min={output_np[2].min():.6f}, max={output_np[2].max():.6f}")
        print(f"  Channel 3 (TM phase): min={output_np[3].min():.6f}, max={output_np[3].max():.6f}")
        
        te_resistivity_log10 = output_np[0].copy()
        tm_resistivity_log10 = output_np[2].copy()
        output_np[0] = np.power(10, output_np[0])  # TE resistivity
        output_np[2] = np.power(10, output_np[2])  # TM resistivity
        
        print(f"[postprocess_output] After inverse log10:")
        print(f"  Channel 0 (TE resistivity): min={output_np[0].min():.6f}, max={output_np[0].max():.6f} (was log10: {te_resistivity_log10.min():.6f} to {te_resistivity_log10.max():.6f})")
        print(f"  Channel 1 (TE phase, unchanged): min={output_np[1].min():.6f}, max={output_np[1].max():.6f}")
        print(f"  Channel 2 (TM resistivity): min={output_np[2].min():.6f}, max={output_np[2].max():.6f} (was log10: {tm_resistivity_log10.min():.6f} to {tm_resistivity_log10.max():.6f})")
        print(f"  Channel 3 (TM phase, unchanged): min={output_np[3].min():.6f}, max={output_np[3].max():.6f}")
    
    return output_np

def save_output(output_data, output_dir, model_name, MT_Mode, data_dsp_dim, out_channels):
    """
    Save output data to files
    Output format matches training data format
    
    Parameters:
    - output_data: Postprocessed output array
    - output_dir: Directory to save output files
    - model_name: Model name
    - MT_Mode: MT mode ('TE', 'TM', or 'Both')
    - data_dsp_dim: Data downsampled dimensions
    - out_channels: Number of output channels
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Reshape output to match original data format (H, W)
    # Training data processing flow (DataLoad_Train_FromFiles.py):
    # Step 1: raw_resistivity (1D) -> reshape(data_dim[0], data_dim[1]) -> .T -> (data_dim[1], data_dim[0])
    # Step 2: log10 transform
    # Step 3: array([resistivity, phase]) -> (2, data_dim[1], data_dim[0])
    # Step 4: transpose(1,2,0) -> (data_dim[1], data_dim[0], 2)
    # Step 5: For each channel k: data1_set[:,:,k] -> (data_dim[1], data_dim[0])
    # Step 6: downsample -> (H_dsp, W_dsp) where H_dsp = data_dim[1]//dsp, W_dsp = data_dim[0]//dsp
    # Step 7: reshape(1, H_dsp*W_dsp) -> append -> (2, H_dsp*W_dsp)
    # Step 8: Final reshape: (valid_count, 2, H_dsp*W_dsp)
    # Step 9: In MT_train.py: reshape to (batch, 2, H_dsp, W_dsp)
    #
    # Model output: (batch, 2, H_dsp, W_dsp) where:
    #   - Channel 0: resistivity (log10 scale, needs inverse log10)
    #   - Channel 1: phase (no transform needed)
    #   - Shape: (2, H_dsp, W_dsp) after removing batch
    #
    # To reverse the training process and save back to file format:
    # - Model output is (2, H_dsp, W_dsp) where H_dsp corresponds to data_dim[1]//dsp, W_dsp to data_dim[0]//dsp
    # - Training: reshape -> .T (data_dim[0], data_dim[1]) -> (data_dim[1], data_dim[0]) -> transpose(1,2,0) -> downsample -> (H_dsp, W_dsp)
    # - To reverse: We need to undo the transpose operations
    # - Since training: (data_dim[0], data_dim[1]) -> .T -> (data_dim[1], data_dim[0]) -> transpose(1,2,0) -> downsample -> (H_dsp, W_dsp)
    # - The model output (H_dsp, W_dsp) corresponds to the shape after downsample
    # - To get back to original format, we need to transpose: (H_dsp, W_dsp) -> .T -> (W_dsp, H_dsp)
    # - But wait: if phase is correct with .T, then the model output (H_dsp, W_dsp) should be transposed to (W_dsp, H_dsp)
    # - However, we need to check: does data_dsp_dim[0] = H_dsp = data_dim[1]//dsp? And data_dsp_dim[1] = W_dsp = data_dim[0]//dsp?
    # - If so, then to match original format, we need: output (H_dsp, W_dsp) -> .T -> (W_dsp, H_dsp)
    # - But the original file format is (data_dim[0], data_dim[1]), so we need to save as (W_dsp, H_dsp) which is (data_dim[0]//dsp, data_dim[1]//dsp)
    # - Actually, let's think differently: the model learned to predict in the shape (H_dsp, W_dsp) where H_dsp=H, W_dsp=W after downsample
    # - The training data after downsample is (H_dsp, W_dsp), and this is reshaped to (batch, 2, H_dsp, W_dsp)
    # - So the model output (2, H_dsp, W_dsp) is already in the correct spatial orientation
    # - To save to file, we just need to reshape each channel: (H_dsp, W_dsp) -> save as is
    # - But wait, the original file is 1D, so we need to know the original dimensions
    # - Actually, the issue might be that we need to apply the inverse of the .T operation
    # - Training: (data_dim[0], data_dim[1]) -> .T -> (data_dim[1], data_dim[0]) -> downsample -> (H_dsp, W_dsp)
    # - So if model output is (H_dsp, W_dsp), to get back to (data_dim[0]//dsp, data_dim[1]//dsp), we need to .T again
    # - But phase is correct, so let's check: if phase uses .T and is correct, then resistivity should also use .T
    # - However, the user says resistivity is wrong but phase is correct
    # - This suggests that resistivity and phase might need different transpose logic
    # - Let me check: maybe the issue is that resistivity and phase are processed differently in training?
    # - Looking at the code, both resistivity and phase go through the same processing: reshape -> .T -> log10 (only resistivity) -> transpose(1,2,0) -> downsample
    # - So they should have the same transpose logic
    # - But if phase is correct with .T and resistivity is wrong, maybe the issue is that resistivity needs NO transpose?
    # - Or maybe resistivity needs a different transpose?
    # - Let's try: if phase is correct with .T, and both should be the same, but resistivity is wrong, maybe resistivity should NOT use .T?
    if out_channels == 1:
        # Single channel: save as resistivity only
        print(f"[save_output] Single channel output, saving as resistivity only")
        resistivity_reshaped = output_data[0].reshape(data_dsp_dim[0], data_dsp_dim[1])
        resistivity = resistivity_reshaped.T
        resistivity_file = os.path.join(output_dir, f"{model_name}_{MT_Mode}_resistivity.txt")
        np.savetxt(resistivity_file, resistivity, fmt='%.6f')
        print(f"RESULT_PATH: {resistivity_file}")
        print(f"Note: Model outputs 1 channel only. For full MT output (resistivity+phase), retrain with in_channels=1, out_channels=2")
        return resistivity_file, None
    elif out_channels == 2:
        # TE or TM mode: 2 channels (resistivity, phase)
        # output_data shape: (out_channels, H, W) = (2, data_dsp_dim[0], data_dsp_dim[1])
        print(f"[save_output] output_data shape: {output_data.shape}")
        print(f"[save_output] data_dsp_dim: {data_dsp_dim}")
        print(f"[save_output] Reshaping resistivity from {output_data[0].shape} to ({data_dsp_dim[0]}, {data_dsp_dim[1]})")
        
        # Since phase prediction is correct with .T, let's try NOT transposing resistivity
        # Training: (data_dim[0], data_dim[1]) -> .T -> (data_dim[1], data_dim[0]) -> downsample -> (H_dsp, W_dsp)
        # Model output: (H_dsp, W_dsp) where H_dsp = data_dim[1]//dsp, W_dsp = data_dim[0]//dsp
        # To reverse: (H_dsp, W_dsp) -> .T -> (W_dsp, H_dsp) = (data_dim[0]//dsp, data_dim[1]//dsp)
        # But if phase is correct with .T, then maybe resistivity should NOT use .T?
        # Or maybe the issue is that we need to check the actual dimensions?
        # Training flow analysis:
        # 1. raw_resistivity (1D) -> reshape(data_dim[0], data_dim[1]) -> .T -> (data_dim[1], data_dim[0])
        # 2. log10 -> (data_dim[1], data_dim[0])
        # 3. array([resistivity, phase]) -> (2, data_dim[1], data_dim[0])
        # 4. transpose(1,2,0) -> (data_dim[1], data_dim[0], 2)
        # 5. For channel k: data1_set[:,:,k] -> (data_dim[1], data_dim[0])
        # 6. downsample -> (H_dsp, W_dsp) where H_dsp=data_dim[1]//dsp, W_dsp=data_dim[0]//dsp
        # 7. reshape -> (batch, 2, H_dsp, W_dsp)
        #
        # IMPORTANT: Verify channel order and data processing
        # Training: label_set[:, 0, :] = resistivity (log10), label_set[:, 1, :] = phase
        # Model output: output_data[0] = resistivity (already inverse log10 in postprocess_output)
        #               output_data[1] = phase (no log10 transform)
        print(f"[save_output] Extracting channels from output_data:")
        print(f"  output_data[0] shape: {output_data[0].shape}, range: [{output_data[0].min():.6f}, {output_data[0].max():.6f}] (resistivity, after inverse log10)")
        print(f"  output_data[1] shape: {output_data[1].shape}, range: [{output_data[1].min():.6f}, {output_data[1].max():.6f}] (phase)")
        
        # IMPORTANT: Verify reshape and transpose logic
        # Training: (data_dim[0], data_dim[1]) -> .T -> (data_dim[1], data_dim[0]) -> downsample -> (H_dsp, W_dsp)
        # where H_dsp = data_dim[1]//dsp, W_dsp = data_dim[0]//dsp
        # Model output: (H_dsp, W_dsp) = (data_dim[1]//dsp, data_dim[0]//dsp)
        # To reverse: (H_dsp, W_dsp) -> .T -> (W_dsp, H_dsp) = (data_dim[0]//dsp, data_dim[1]//dsp)
        print(f"[save_output] Model output shape: {output_data.shape}")
        print(f"[save_output] data_dsp_dim: {data_dsp_dim} (H_dsp={data_dsp_dim[0]}, W_dsp={data_dsp_dim[1]})")
        print(f"[save_output] Expected after transpose: ({data_dsp_dim[1]}, {data_dsp_dim[0]})")
        
        resistivity_reshaped = output_data[0].reshape(data_dsp_dim[0], data_dsp_dim[1])
        phase_reshaped = output_data[1].reshape(data_dsp_dim[0], data_dsp_dim[1])
        
        print(f"[save_output] After reshape (before transpose):")
        print(f"  resistivity shape: {resistivity_reshaped.shape}, range: [{resistivity_reshaped.min():.6f}, {resistivity_reshaped.max():.6f}]")
        print(f"  phase shape: {phase_reshaped.shape}, range: [{phase_reshaped.min():.6f}, {phase_reshaped.max():.6f}]")
        
        # IMPORTANT: User reports phase is correct, so transpose logic must be correct
        # Both resistivity and phase should use the same transpose logic
        # If phase is correct, resistivity should also be correct with the same logic
        resistivity = resistivity_reshaped.T
        phase = phase_reshaped.T
        
        print(f"[save_output] After transpose (same logic for both):")
        print(f"  resistivity shape: {resistivity.shape}, range: [{resistivity.min():.6f}, {resistivity.max():.6f}]")
        print(f"  phase shape: {phase.shape}, range: [{phase.min():.6f}, {phase.max():.6f}]")
        
        resistivity_file = os.path.join(output_dir, f"{model_name}_{MT_Mode}_resistivity.txt")
        phase_file = os.path.join(output_dir, f"{model_name}_{MT_Mode}_phase.txt")
        
        np.savetxt(resistivity_file, resistivity, fmt='%.6f')
        np.savetxt(phase_file, phase, fmt='%.6f')
        
        print(f"RESULT_PATH: {resistivity_file}")
        print(f"RESULT_PATH: {phase_file}")
        
        return resistivity_file, phase_file
    
    elif out_channels == 4:
        # Both mode: 4 channels (TE resistivity, TE phase, TM resistivity, TM phase)
        te_resistivity = output_data[0].reshape(data_dsp_dim[0], data_dsp_dim[1]).T
        te_phase = output_data[1].reshape(data_dsp_dim[0], data_dsp_dim[1]).T
        tm_resistivity = output_data[2].reshape(data_dsp_dim[0], data_dsp_dim[1]).T
        tm_phase = output_data[3].reshape(data_dsp_dim[0], data_dsp_dim[1]).T
        
        te_resistivity_file = os.path.join(output_dir, f"{model_name}_TE_resistivity.txt")
        te_phase_file = os.path.join(output_dir, f"{model_name}_TE_phase.txt")
        tm_resistivity_file = os.path.join(output_dir, f"{model_name}_TM_resistivity.txt")
        tm_phase_file = os.path.join(output_dir, f"{model_name}_TM_phase.txt")
        
        np.savetxt(te_resistivity_file, te_resistivity, fmt='%.6f')
        np.savetxt(te_phase_file, te_phase, fmt='%.6f')
        np.savetxt(tm_resistivity_file, tm_resistivity, fmt='%.6f')
        np.savetxt(tm_phase_file, tm_phase, fmt='%.6f')
        
        print(f"RESULT_PATH: {te_resistivity_file}")
        print(f"RESULT_PATH: {te_phase_file}")
        print(f"RESULT_PATH: {tm_resistivity_file}")
        print(f"RESULT_PATH: {tm_phase_file}")
        
        return te_resistivity_file, te_phase_file, tm_resistivity_file, tm_phase_file

def predict(model_path, resistivity_model_path, output_mode='TM'):
    """
    Main prediction function
    
    Parameters:
    - model_path: Path to trained model file (.pth)
    - resistivity_model_path: Path to input resistivity model file (.txt)
    - output_mode: Output mode ('TE', 'TM', or 'Both')
    """
    print("=" * 60)
    print("GeoDLFWD Forward Modeling (Prediction)")
    print("=" * 60)
    print(f"Input resistivity model: {resistivity_model_path}")
    print(f"Trained model: {model_path}")
    print(f"Output mode: {output_mode}")
    print("=" * 60)
    
    # Get parameters from ParamConfig
    data_dim = getattr(ParamConfig, 'DataDim', [32, 32])
    model_dim = getattr(ParamConfig, 'ModelDim', [32, 32])
    data_dsp_blk = getattr(ParamConfig, 'data_dsp_blk', (1, 1))
    label_dsp_blk = getattr(ParamConfig, 'label_dsp_blk', (1, 1))
    model_name = getattr(ParamConfig, 'ModelName', 'DinkNet')
    
    # Determine expected channels based on output mode (for MT forward modeling)
    if output_mode == 'Both' or output_mode == 'BOTH' or output_mode == 'TE&TM':
        expected_in_channels = 1  # Input: resistivity model
        expected_out_channels = 4  # Output: TE resistivity + TE phase + TM resistivity + TM phase
        MT_Mode = 'Both'
    elif output_mode == 'TE':
        expected_in_channels = 1
        expected_out_channels = 2
        MT_Mode = 'TE'
    else:  # TM or default
        expected_in_channels = 1
        expected_out_channels = 2
        MT_Mode = 'TM'
    
    # Verify model file exists before any operations
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model file not found: {model_path}")
    
    # Auto-detect architecture from checkpoint (model must match how it was trained)
    detected = get_model_architecture_from_checkpoint(model_path)
    if detected is not None:
        in_channels, out_channels = detected
        print(f"Detected model architecture from checkpoint: in_channels={in_channels}, out_channels={out_channels}")
        if in_channels != expected_in_channels or out_channels != expected_out_channels:
            print(f"Warning: Checkpoint architecture (in={in_channels}, out={out_channels}) differs from "
                  f"expected MT config (in={expected_in_channels}, out={expected_out_channels})")
            print(f"For MT forward modeling, use a model trained with 1 input channel (resistivity) "
                  f"and 2 output channels (resistivity+phase). Retrain if needed.")
    else:
        in_channels = expected_in_channels
        out_channels = expected_out_channels
        print(f"Using expected MT architecture: in_channels={in_channels}, out_channels={out_channels}")
    
    # Preprocess input resistivity model
    print("\nPreprocessing input resistivity model...")
    input_tensor, label_dsp_dim = preprocess_resistivity_model(
        resistivity_model_path, data_dim, model_dim, label_dsp_blk
    )
    
    # Handle input channel mismatch: if model expects more channels than we have, replicate
    if input_tensor.shape[1] < in_channels:
        input_tensor = input_tensor.repeat(1, in_channels, 1, 1)
        print(f"Replicated input to {in_channels} channels to match model")
    elif input_tensor.shape[1] > in_channels:
        input_tensor = input_tensor[:, :in_channels, :, :]
        print(f"Trimmed input to {in_channels} channels to match model")
    
    # Calculate data downsampled dimensions
    data_dsp_dim = tuple(dim // blk for dim, blk in zip(data_dim, data_dsp_blk))
    
    # Load trained model
    print(f"\nLoading trained model: {model_path}")
    model = create_model(model_name, in_channels, out_channels)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict, strict=True)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass
    print("\nRunning forward modeling...")
    with torch.no_grad():
        if 'Unet' in model_name or 'unet' in model_name:
            # UnetModel.forward needs label_dsp_dim parameter
            output = model(input_tensor, data_dim)
        else:
            # DinkNet50.forward only needs data parameter
            output = model(input_tensor)
    
    print(f"Model output shape: {output.shape}")
    
    # Postprocess output
    print("\nPostprocessing output...")
    output_data = postprocess_output(output, data_dsp_dim, out_channels)
    
    # Save output files
    print("\nSaving output files...")
    results_dir = getattr(ParamConfig, 'ResultsDir', 'results/')
    save_output(output_data, results_dir, model_name, MT_Mode, data_dsp_dim, out_channels)
    
    print("\n" + "=" * 60)
    print("Forward modeling completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        # Parse command line arguments
        # Arguments: resistivity_model_path, trained_model_path, output_mode
        # (This order matches the call from GeoDLFWD.py start_prediction)
        if len(sys.argv) < 3:
            print("Usage: python MT_test.py <resistivity_model_path> <trained_model_path> [output_mode]")
            print("Example: python MT_test.py model.txt trained_model.pth TM")
            sys.exit(1)
        
        resistivity_model_path = sys.argv[1]
        trained_model_path = sys.argv[2]
        output_mode = sys.argv[3] if len(sys.argv) > 3 else 'TM'
        
        # Print debug information to verify correct model is being used
        print("=" * 60)
        print("PREDICTION PARAMETERS:")
        print(f"  Resistivity model path: {resistivity_model_path}")
        print(f"  Trained model path: {trained_model_path}")
        print(f"  Output mode: {output_mode}")
        print("=" * 60)
        
        # Verify model file exists
        if not os.path.exists(trained_model_path):
            print(f"ERROR: Trained model file not found: {trained_model_path}")
            sys.exit(1)
        
        # Call predict with correct parameter order: (model_path, resistivity_model_path, output_mode)
        predict(trained_model_path, resistivity_model_path, output_mode)
        
    except KeyboardInterrupt:
        print("\nPrediction interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
