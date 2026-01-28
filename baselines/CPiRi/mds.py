import os
import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))
from basicts.data import TimeSeriesForecastingDataset


import logging
import numpy as np
from torch.utils.data import Dataset, Sampler
from typing import List, Dict, Iterator
import random
import math

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from typing import List, Dict, Iterator
import random
import math

# CombinedTimeSeriesDataset 类保持不变 (使用优化后的版本)
class CombinedTimeSeriesDataset(Dataset):
    """
    组合多个时间序列数据集，并优化了内存使用。
    """
    def __init__(self, dataset_params_list: List[Dict], mode: str, logger: logging.Logger = None, max_value = 100.0):
        if mode == 'test':
            dataset_params_list = [dataset_params_list[0]]
        self.datasets = [
            TimeSeriesForecastingDataset(mode=mode, logger=logger, **params)
            for params in dataset_params_list
        ]
        for ds in self.datasets:
            logger.info(f"Subdataset {ds.dataset_name} loaded max: {ds.data.max()} mean: {ds.data.mean()} min: {ds.data.min()} std: {ds.data.std()} shape: {ds.data.shape}.")
            ds.data[ds.data > max_value] = 0.0
            logger.info(f"Subdataset {ds.dataset_name} loaded max: {ds.data.max()} mean: {ds.data.mean()} min: {ds.data.min()} std: {ds.data.std()} shape: {ds.data.shape}.")
        
        self.dataset_lengths = [len(ds) for ds in self.datasets]
        # 累积长度，用于快速索引
        self.cumulative_lengths = np.cumsum([0] + self.dataset_lengths)
        self.total_length = int(self.cumulative_lengths[-1])

    def __len__(self):
        return self.total_length

    def __getitem__(self, global_idx: int) -> Dict:
        if not 0 <= global_idx < self.total_length:
            raise IndexError("Global index out of range")
        
        ds_idx = np.searchsorted(self.cumulative_lengths, global_idx, side='right') - 1
        local_idx = global_idx - self.cumulative_lengths[ds_idx]
        
        return self.datasets[ds_idx][local_idx]

    def get_dataset_info(self) -> List[Dict]:
        return [
            {
                'start_idx': int(self.cumulative_lengths[i]),
                'end_idx': int(self.cumulative_lengths[i+1]),
                'dataset_idx': i,
                'num_samples': ds_len
            }
            for i, ds_len in enumerate(self.dataset_lengths)
        ]

class SameDatasetBatchSampler(Sampler[List[int]]):
    """
    批采样器的新实现，遵循“先生成所有批次，再全局打乱”的逻辑。

    它首先为每个子数据集生成所有批次，并将它们收集到一个列表中。
    然后，如果启用了shuffle，它会打乱这个包含所有批次的列表。
    这种方式确保了批次间的随机性，使模型可以交错地看到来自不同数据集的批次。
    """
    def __init__(
        self,
        dataset: CombinedTimeSeriesDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.dataset_info = dataset.get_dataset_info()

    def __iter__(self) -> Iterator[List[int]]:
        # 步骤 1: 创建一个列表，用于汇集所有子数据集的所有批次
        all_batches = []
        
        # 步骤 2: 为每个子数据集生成批次
        for info in self.dataset_info:
            # 获取当前子数据集的全局索引
            indices = list(range(info['start_idx'], info['end_idx']))
            
            # 如果需要，打乱子数据集内部的样本顺序
            if self.shuffle:
                random.shuffle(indices)
            
            # 将打乱后的索引分批
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                
                # 根据drop_last决定是否保留最后一个不完整的批次
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                
                all_batches.append(batch)
        
        # 步骤 3: 全局打乱所有生成的批次
        # 这确保了来自不同数据集的批次是随机交错的
        if self.shuffle:
            random.shuffle(all_batches)
        
        # 返回最终批次列表的迭代器
        return iter(all_batches)

    def __len__(self) -> int:
        """长度计算逻辑不变，因为它只关心最终的批次总数。"""
        total_batches = 0
        for info in self.dataset_info:
            num_samples = info['num_samples']
            if self.drop_last:
                total_batches += num_samples // self.batch_size
            else:
                total_batches += math.ceil(num_samples / self.batch_size)
        return total_batches

