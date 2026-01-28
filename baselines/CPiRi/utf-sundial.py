import os
import sys
# import torch
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse, masked_wape
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_adj

from baselines.CPiRi.runner import AmpRunner
from baselines.CPiRi.arch import ZMTS, Sundial
# from .runner import AmpRunner
# from .arch import ZMTS

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = 'METR-LA'  # Dataset name
# DATA_NAME = 'PEMS04'  # Dataset name
# DATA_NAME = 'PEMS08'  # Dataset name
# DATA_NAME = 'PEMS-BAY'  # Dataset name
DATA_NAME = 'SD'  # Dataset name
regular_settings = get_regular_settings(DATA_NAME)
# INPUT_LEN = regular_settings['INPUT_LEN']  # Length of input sequence
# OUTPUT_LEN = regular_settings['OUTPUT_LEN']  # Length of output sequence
INPUT_LEN = 336 # LTSF
OUTPUT_LEN = 336 # LTSF
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']  # Train/Validation/Test split ratios
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL'] # Whether to normalize each channel of the data
RESCALE = regular_settings['RESCALE'] # Whether to rescale the data
NULL_VAL = regular_settings['NULL_VAL'] # Null value in the data
# Model architecture and parameters
MODEL_ARCH = Sundial
MODEL_PARAM = {
    "input_len": INPUT_LEN,
    "input_dim": 1,
    "output_len": OUTPUT_LEN,
}
NUM_EPOCHS = 50

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = 'Sundial'
CFG.GPU_NUM = 1 # Number of GPUs to use (0 for CPU mode)
# Runner
CFG.RUNNER = AmpRunner


############################## 环境配置 ##############################

CFG.ENV = EasyDict() # 环境设置。默认值：None

# GPU 和随机种子设置
CFG.ENV.TF32 = True # 是否在 GPU 中使用 TensorFloat-32。默认值：False。


############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
# Dataset settings
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
    # 'mode' is automatically set by the runner
})

############################## Scaler Configuration ##############################
CFG.SCALER = EasyDict()
# Scaler settings
CFG.SCALER.TYPE = ZScoreScaler # Scaler class
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
})

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
# Model settings
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.MODEL.DTYPE= 'float16'

############################## Metrics Configuration ##############################

CFG.METRICS = EasyDict()
# Metrics settings
CFG.METRICS.FUNCS = EasyDict({
                                'MAE': masked_mae,
                                'MAPE': masked_mape,
                                'RMSE': masked_rmse,
                                'WAPE': masked_wape,
                            })
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = NULL_VAL

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
# CFG.TRAIN.COMPILE_MODEL = True
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae
# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.001,
    "weight_decay": 0.00001,
}
# Learning rate scheduler settings
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [1, 5, 15, 30, 40, 45],
    "gamma": 0.5
}
# NUM_ITERATIONS = int(NUM_EPOCHS * len(TimeSeriesForecastingDataset(mode='train', **CFG.DATASET.PARAM)) / CFG.TRAIN.DATA.BATCH_SIZE)
# CFG.TRAIN.LR_SCHEDULER.TYPE = "CosineWarmup"
# CFG.TRAIN.LR_SCHEDULER.PARAM = {
#     'num_warmup_steps': max(int(NUM_ITERATIONS / 100 * 1), 500), # 1%的warmup启动比例
#     'num_training_steps': NUM_ITERATIONS,
# }
CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 5.0
}
# Train data loader settings
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 32
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 16
CFG.TRAIN.DATA.PREFETCH = True
# CFG.TRAIN.DATA.PIN_MEMORY = True

# CFG.TRAIN.DATA.SHUFFLENODES = True
# CFG.TRAIN.DATA.BATCH_SAMPLER = 

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 32
# CFG.VAL.DATA.SHUFFLENODES= True

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 10
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 64
CFG.TEST.DATA.BATCH_SIZE = 8
CFG.TEST.DATA.SHUFFLENODES = True

CFG.TRAIN.COMPILE_MODEL = True
CFG.TEST.DATA.NUM_WORKERS = 0
CFG.TEST.DATA.PREFETCH = True
# CFG.TEST.DATA.PIN_MEMORY = True

# CFG.TEST.DATA.SHUFFLENODES = True
# CFG.TEST.DATA.NUN_SHUFFLENODES = 0.75
############################## Evaluation Configuration ##############################

CFG.EVAL = EasyDict()

# Evaluation parameters
CFG.EVAL.USE_GPU = False # Whether to use GPU for evaluation. Default: True


'''
        if ckpt_path == 'noload':
            logger.info('Skip loading model')
        el
\basicts\launcher.py
'''