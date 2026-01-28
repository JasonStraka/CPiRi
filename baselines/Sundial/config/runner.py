from typing import Dict

import torch

from basicts.runners.base_utsf_runner import BaseUniversalTimeSeriesForecastingRunner


class SundialRunner(BaseUniversalTimeSeriesForecastingRunner):

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.context_length = cfg['MODEL']['PARAM']['context_length']

    def forward(self, data: Dict, train=True, **kwargs) -> Dict:
        inputs, labels, mask, target_mask = data['inputs'], data['labels'], data['mask'], data['target_mask']
        inputs = self.to_running_device(inputs)
        target = self.to_running_device(labels)
        mask = self.to_running_device(mask)
        target_mask = self.to_running_device(target_mask)
        loss,  _mae, _mse, _trend_diff = self.model(context=inputs, target=target, mask=mask, target_mask=target_mask, training=train) # NOTE: TimeMoE integrates the loss calculation in the forward method
        if train: # 非训练、验证的时候算吧,太慢了
            return {'loss': loss, 'MAE': torch.tensor(0.0), 'MSE': torch.tensor(0.0), 'DIF': torch.tensor(0.0)}
        return {'loss': loss, 'MAE': _mae, 'MSE': _mse, 'DIF': _trend_diff}
    
    # def train_iters(self, iteration, dataloader):
    #     loss = super().train_iters(iteration, dataloader)
    #     if iteration % 100 == 0:
    #     # if iteration % 1 == 0:
    #         self.print_iteration_meters('train')
    #     return loss
