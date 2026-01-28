import os
import sys
import torch
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse, masked_wape
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_adj

from baselines.CPiRi.runner import AmpRunner
from baselines.CPiRi.arch import CPiRi
# from .runner import AmpRunner
# from .arch import ZMTS

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = 'GLA'  # Dataset name
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
MODEL_ARCH = CPiRi
MODEL_PARAM = {
    "input_len": INPUT_LEN,
    "input_dim": 1,
    "output_len": OUTPUT_LEN,
    "spital_num_layers": 4,
    "from_pretrain": True
}
NUM_EPOCHS = 60

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = 'ZMTS 4'
CFG.GPU_NUM = 8 # Number of GPUs to use (0 for CPU mode)
# Runner
CFG.RUNNER = AmpRunner

# CFG.DIST_BACKEND = 'gloo'
# CFG.DIST_INIT_METHOD = 'env://'

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
CFG.TRAIN.COMPILE_MODEL = True
# CFG.TRAIN.fintune_EPOCHS = 51
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
    "lr": 0.0002,
    "weight_decay": 0.00001,
}
# Learning rate scheduler settings
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    # "milestones": [1, 5, 15, 30, 40, 45, 55],
    "milestones": [1, 10, 25, 40],
    # "milestones": [1, 5, 15, 40, 45, 55],
    "gamma": 0.5
}
# NUM_ITERATIONS = int(NUM_EPOCHS * len(TimeSeriesForecastingDataset(mode='train', **CFG.DATASET.PARAM)) / CFG.TRAIN.DATA.BATCH_SIZE)
# CFG.TRAIN.LR_SCHEDULER.TYPE = "CosineWarmup"
# CFG.TRAIN.LR_SCHEDULER.PARAM = {
#     'num_warmup_steps': max(int(NUM_ITERATIONS / 100 * 1), 500), # 1%的warmup启动比例
#     'num_training_steps': NUM_ITERATIONS,
# }
CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 3.0
}
# Train data loader settings
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 8
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 0
CFG.TRAIN.DATA.PIN_MEMORY = True
CFG.TRAIN.DATA.PREFETCH = True
CFG.TRAIN.DATA.SHUFFLENODES= True

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 5
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 16
CFG.VAL.DATA.PREFETCH = True
# CFG.VAL.DATA.SHUFFLENODES= True

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 51
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 16
CFG.TEST.DATA.NUM_WORKERS = 0
# CFG.TEST.DATA.PIN_MEMORY = True
CFG.TEST.DATA.PREFETCH = True

# CFG.TEST.DATA.SHUFFLENODES= True
############################## Evaluation Configuration ##############################

CFG.EVAL = EasyDict()

# Evaluation parameters
CFG.EVAL.USE_GPU = False # Whether to use GPU for evaluation. Default: True


if __name__ == "__main__":

    import torch
    import torch._dynamo
    torch.set_float32_matmul_precision('high')
    torch._dynamo.config.accumulated_cache_size_limit = 256
    torch._dynamo.config.cache_size_limit = 256  # 或更高
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.optimize_ddp = False
    torch.set_num_threads(24) # aviod high cpu avg usage

    bs = 128
    # bs = 32
    L = 336
    N = 207
    output_dir = 'datasets/ZZMETR-LA'
    output_dir = 'datasets/ZMETR-LA'

    import numpy as np
    import json
    from tqdm import tqdm
    data_file_path = f'datasets/{CFG.DATASET.NAME}/data.dat'
    description_file_path = f'datasets/{CFG.DATASET.NAME}/desc.json'
    with open(description_file_path, 'r') as f:
        description = json.load(f)
    data = np.memmap(data_file_path, dtype='float32', mode='r', shape=tuple(description['shape']))[:, : , 0].copy()
    # mask = np.isnan(data) | np.isinf(data)
    print(data.mean(), data.std())
    print(data[:int(34272*0.7)].mean(), data[:int(34272*0.7)].std())
    # data = mask * (data - data.mean()) / data.std()
    # valid_data = data[~mask]
    # mean, std = valid_data.mean(), valid_data.std()
    mean, std = data[:int(34272*0.7)].mean(), data[:int(34272*0.7)].std()
    print(mean, std)
    # 53.719006 20.261427
    # 54.40589 19.49425
    data = (data - mean) / std
    # data[mask] = 0  # 将无效值设为 0
    # print(mask.sum())
    model = MODEL_ARCH(**MODEL_PARAM).cuda().eval()
    model = torch.compile(model)
    print(data.shape)
    data = torch.from_numpy(data).unfold(0, L, 1) # B, N, L
    data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN and inf with 0
    print(data.shape)

    # data = data[:1000]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(os.path.join(output_dir, 'data_hidden.dat')) and os.path.exists(os.path.join(output_dir, 'data_target.dat')):
        print(f'Output files already exist in {output_dir}, del...')
        os.remove(os.path.join(output_dir, 'data_hidden.dat'))
        os.remove(os.path.join(output_dir, 'data_target.dat'))
        # rm datasets/ZMETR-LA/data_hidden.dat datasets/ZMETR-LA/data_target.dat
    fp_x = np.memmap(os.path.join(output_dir, 'data_hidden.dat'), dtype='float32', mode='w+', shape=(data.shape[0]-L, model._config.hidden_size, 207))
    print(fp_x.shape)
    fp_y = np.memmap(os.path.join(output_dir, 'data_target.dat'), dtype='float32', mode='w+', shape=(data.shape[0]-L, OUTPUT_LEN, 207))
    print(fp_y.shape)
    # (33601, 768, 207)
    # (33601, 336, 207)
    
    for i in tqdm(range((data.shape[0]-L)//bs+1)):
        future_data, history_data = data[(i)*bs+L:(i+1)*bs+L, :, :], data[(i)*bs:(i+1)*bs, :, :] # B N, L
        # print(future_data.shape, history_data.shape) # torch.Size([128, 207, 336]) torch.Size([128, 207, 336])
        history_data = history_data.transpose(1, 2).view(-1, L, N, 1).contiguous()
        # future_data = future_data.cuda().transpose(1, 2).reshape(-1, L, N, 1)
        future_data = future_data.transpose(1, 2).view(-1, L, N).contiguous()
        # history_data = history_data.transpose(1, 2).reshape(-1, L, N, 1)
        # future_data = future_data.transpose(1, 2).reshape(-1, L, N, 1)
        _b = future_data.shape[0]
        history_data = history_data[:_b]
        # history_data = (history_data - mean) / std
        # print(future_data.shape, history_data.shape)
        with torch.inference_mode():
        # with torch.no_grad():
        # 混合精度会导致计算问题！
            # with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            #     res = model.forward_hid(history_data, future_data, revin=False)
                res = model.forward_hid(history_data.cuda(), revin=False)
        # print(res['hidden'].shape, future_data.shape) # torch.Size([128, 768, 207]) torch.Size([128, 336, 207])
        fp_x[(i)*bs:(i)*bs+_b] = res['hidden'][:].cpu().numpy() # B, C, N
        fp_x.flush()
        # fp_y[(i)*bs:(i)*bs+_b] = res['target'][:].squeeze(-1).cpu().numpy()
        fp_y[(i)*bs:(i)*bs+_b] = future_data.numpy() # B, L, N
        fp_y.flush()
        # print((i)*bs, (i+1)*bs, res['target'].shape)
        # torch.cuda.empty_cache()

    del fp_x, fp_y