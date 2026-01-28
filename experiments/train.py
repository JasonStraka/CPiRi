# Run a baseline model in BasicTS framework.
# pylint: disable=wrong-import-position
import time
# time.sleep(60 * 20)
import os
import sys
from argparse import ArgumentParser
from multiprocessing import cpu_count
cpu_num = cpu_count() # 自动获取最大核心数目
cpu_num = 40
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)

sys.path.append(os.path.abspath(__file__ + '/../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch._dynamo

torch.set_float32_matmul_precision('high')
torch._dynamo.config.accumulated_cache_size_limit = 256
torch._dynamo.config.cache_size_limit = 256  # 或更高
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.optimize_ddp = False

import basicts

torch.set_num_threads(40) # aviod high cpu avg usage

def parse_args():
    parser = ArgumentParser(description='Run time series forecasting model in BasicTS framework!')
    parser.add_argument('-c', '--cfg', default='baselines/Crossformer/METR-LA_LTSF.py', help='training config')
    parser.add_argument('-g', '--gpus', default='0', help='visible gpus')
    return parser.parse_args()

def main():
    args = parse_args()
    basicts.launch_training(args.cfg, args.gpus, node_rank=0)


if __name__ == '__main__':
    main()
