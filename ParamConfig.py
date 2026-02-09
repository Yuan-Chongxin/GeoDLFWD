# -*- coding: utf-8 -*-
"""
Parameter Configuration

Created in July 2021

Author: ycx

"""


####################################################
#####            MAIN PARAMETERS               #####
####################################################

ReUse         = True      # Whether to reuse the pre-trained model
#DataDim       = [96,96]
DataDim       = [32,32] # Size of the training data
data_dsp_blk  = (1,1)     # Downsampling ratio of input
ModelDim      = [32,32] # Size of the output model
#ModelDim      = [96,96]
label_dsp_blk = (1,1)     # Downsampling ratio of output
dh            = 10        # Space interval
DataFormat    = 'depth'   # Data format: 'depth' or 'width'
Device        = 'cuda'    # Device to use for training: 'cuda' or 'cpu'


####################################################
####             NETWORK PARAMETERS             ####
####################################################

Epochs        = 200      # Number of epoch
TrainSize     = 0.80      # Training data size ratio









ValSize       = 0.20      # Validation data size ratio









BatchSize         = 20       # Number of batch size
LearnRate         = 0.001      # Learning rate
DisplayStep   = 1        # Display step
EarlyStop     = 50       # Early stop




MT_Mode = 'TM'  # MT mode: 'TE', 'TM', or 'Both'
ModelName     = 'DinkNet' # Name of the model to use
ModelsDir     = 'm:\DLTool/dl/models/'   # Directory for saving models
PreModel      = 'model'   # Pre-trained model name


####################################################
#####            DATA PARAMETERS                #####
####################################################

# Paths for training data - will be overridden by GUI input
TrainDataDir  = './train_data/'
TrainDataCount = 1000      # Training data size
TestSize      = 200       # Testing data size
ValidSize     = 200       # Validation data size

# Data processing parameters
NormalizeData = True      # Whether to normalize data
DataAugment   = False     # Whether to use data augmentation


####################################################
#####            OUTPUT PARAMETERS              #####
####################################################

ResultsDir    = 'm:\DLTool/dl/results/'  # Directory for saving results
SaveResults   = True         # Whether to save results
SaveModel     = True         # Whether to save model
SaveLoss      = True         # Whether to save loss curves


####################################################
#####            VISUALIZATION PARAMETERS       #####
####################################################

PlotLoss      = True      # Whether to plot loss curves
PlotResults   = True      # Whether to plot results
ShowProgress  = True      # Whether to show progress bar
Verbose       = False     # Whether to show verbose output

Resistivity_Model_Dir = 'M:/dzl' # Added by GUI







TE_Resistivity_Dir = 'M:/sdzl'  # Added by GUI

TM_Resistivity_Dir = 'M:/sdzl' # Added by GUI













TM_Phase_Dir = 'M:/xw' # Added by GUI






