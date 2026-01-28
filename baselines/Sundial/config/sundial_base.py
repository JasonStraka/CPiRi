# 采样概率变化

import os
import sys
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from ..arch import Sundial
# from TimeMoE.data import BLASTDatasetMixUp
from ChronosBolt.data import BLASTDatasetWoMixUp
from .runner import SundialRunner
from TimeMoE.loss import fake_loss


############################## Hot Parameters ##############################
# Dataset & Metrics configuration
# Model architecture and parameters

MODEL_ARCH = Sundial

context_length = 4096
MODEL_PARAM = {
    'model_id': "baselines/Sundial/arch",
    'from_pretrained': False,
    'context_length': context_length,
    'trust_remote_code': True,
}
DATA_NAME = "BLAST"

# N = 20_000_000
# batch size = 16*8
# 20_000_000 / 16 / 8 = 156250 iterations

NUM_ITERATIONS = 200_000 # 总轮数
VAL_ITERATION_INTERVAL = 5_000 # 每VAL_ITERATION_INTERVAL执行一次验证
NUM_ITERATIONS = 50_000 # 总轮数
VAL_ITERATION_INTERVAL = 1_000 # 每VAL_ITERATION_INTERVAL执行一次验证
num_valid_samples = 50000
# for test code
# VAL_ITERATION_INTERVAL = 50
# num_valid_samples = 50
# num_valid_samples = 1000

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = 'Sundial Base revinFalse antimask'
# CFG.MD5 = '3801837cf5cf46792e61ad63fca37a5a'
CFG.DEVICE = 'gpu'
CFG.DEVICE_NUM = 1
# Runner
CFG.RUNNER = SundialRunner

############################## 环境配置 ##############################

CFG.ENV = EasyDict() # 环境设置。默认值：None

# GPU 和随机种子设置
CFG.ENV.TF32 = True # 是否在 GPU 中使用 TensorFloat-32。默认值：False。

############################## Model Configuration ################################
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.DTYPE= 'bfloat16'
# CFG.MODEL.DTYPE= 'float32'

############################## Metrics Configuration ##############################
CFG.METRICS = EasyDict()
# Metrics settings
def fake_mae(MAE):
    return MAE
def fake_mse(MSE):
    return MSE
def fake_dif(DIF):
    return DIF
CFG.METRICS.FUNCS = EasyDict({'MAE': fake_mae, 'MSE': fake_mse, 'DIF': fake_dif})

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.COMPILE_MODEL = True
# CFG.TRAIN.COMPILE_MODEL = False
CFG.TRAIN.NUM_ITERATIONS = NUM_ITERATIONS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_ITERATIONS)])
)
CFG.TRAIN.CKPT_SAVE_STRATEGY = VAL_ITERATION_INTERVAL * 1 # 保存策略，每VAL_ITERATION_INTERVAL * 5保存一次模型
CFG.TRAIN.LOSS = fake_loss
# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "AdamW"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 5e-4, # 1e-3到6k会炸，1e-4稳定
    "betas": (0.9, 0.95),
    "fused": True,
}
# Learning rate scheduler settings
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "CosineWarmup"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    'num_warmup_steps': max(int(NUM_ITERATIONS / 100 * 1), 500), # 1%的warmup启动比例
    'num_training_steps': NUM_ITERATIONS,
}
CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 0.8
}
# Train data loader settings
CFG.TRAIN.DATA = EasyDict()
# CFG.TRAIN.DATA.BATCH_SIZE = 64
CFG.TRAIN.DATA.BATCH_SIZE = 480
CFG.TRAIN.DATA.SHUFFLE = True # has to be False
CFG.TRAIN.DATA.PIN_MEMORY = True
CFG.TRAIN.DATA.PREFETCH = True
CFG.TRAIN.GRAD_ACCUMULATION_STEPS = 1
CFG.TRAIN.DATA.NUM_WORKERS = 8

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = VAL_ITERATION_INTERVAL
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = CFG.TRAIN.DATA.BATCH_SIZE

############################## Evaluation Configuration ##############################

CFG.EVAL = EasyDict()
# Evaluation parameters
CFG.EVAL.USE_GPU = True # Whether to use GPU for evaluation. Default: True

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
# Dataset settings
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = BLASTDatasetWoMixUp
CFG.DATASET.PARAM = EasyDict({
    'context_length': context_length,
    'target_length': 720,
    'num_valid_samples': num_valid_samples
})
