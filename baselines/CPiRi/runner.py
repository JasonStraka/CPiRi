import os
import sys
from typing import Dict, Optional, Tuple, Union
sys.path.append(os.path.abspath(__file__ + '/../../..'))
import torch
from basicts.runners import SimpleTimeSeriesForecastingRunner
from easytorch.device import _DEVICE_TYPE

from packaging import version
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from easytorch.utils import (TimePredictor, get_local_rank, get_logger,
                             is_master, master_only, set_env)

from torch.utils.data import Dataset, DataLoader
from easytorch.utils.data_prefetcher import DataLoaderX
from easytorch.core.data_loader import build_data_loader_ddp
import json
import numpy as np
from tqdm import tqdm

from easytorch.core.checkpoint import (backup_last_ckpt, clear_ckpt, load_ckpt,
                                       save_ckpt)

def build_data_loader_with_sampler(dataset: Dataset, data_cfg: Dict):
    """Build dataloader from `data_cfg`
    `data_cfg` is part of config which defines fields about data, such as `CFG.TRAIN.DATA`

    structure of `data_cfg` is
    {
        'BATCH_SIZE': (int, optional) batch size of data loader (default: ``1``),
        'SHUFFLE': (bool, optional) data reshuffled option (default: ``False``),
        'NUM_WORKERS': (int, optional) num workers for data loader (default: ``0``),
        'PIN_MEMORY': (bool, optional) pin_memory option (default: ``False``),
        'PREFETCH': (bool, optional) set to ``True`` to use `DataLoaderX` (default: ``False``),
    }

    Args:
        dataset (Dataset): dataset defined by user
        data_cfg (Dict): data config

    Returns:
        data loader
    """
    sampler = data_cfg.get('BATCH_SAMPLER', None)
    if sampler is not None:
        sampler = sampler(dataset, batch_size=data_cfg.get('BATCH_SIZE', 1), shuffle=data_cfg.get('SHUFFLE', False))
        return (DataLoaderX if data_cfg.get('PREFETCH', False) else DataLoader)(
            dataset,
            collate_fn=data_cfg.get('COLLATE_FN', None),
            num_workers=data_cfg.get('NUM_WORKERS', 0),
            pin_memory=data_cfg.get('PIN_MEMORY', False),
            batch_sampler = sampler
        )
    return (DataLoaderX if data_cfg.get('PREFETCH', False) else DataLoader)(
        dataset,
        collate_fn=data_cfg.get('COLLATE_FN', None),
        batch_size=data_cfg.get('BATCH_SIZE', 1),
        shuffle=data_cfg.get('SHUFFLE', False),
        num_workers=data_cfg.get('NUM_WORKERS', 0),
        pin_memory=data_cfg.get('PIN_MEMORY', False),
        batch_sampler = sampler
    )

class AmpRunner(SimpleTimeSeriesForecastingRunner):

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        
        # automatic mixed precision (amp)
        self.model_dtype = cfg.get('MODEL.DTYPE', 'float32')
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.model_dtype]
        self.use_amp = self.model_dtype in ['float16', 'bfloat16']
        if self.use_amp: assert _DEVICE_TYPE == 'gpu', 'AMP only supports CUDA.'
        self.amp_context = torch.amp.autocast(device_type='cuda', dtype=ptdtype, enabled=self.use_amp)
        # GradScaler will scale up gradients and some of them might become inf, which may cause lr_scheduler throw incorrect warning information. See:
        # https://discuss.pytorch.org/t/userwarning-detected-call-of-lr-scheduler-step-before-optimizer-step-in-pytorch-1-1-0-and-later-you-should-call-them-in-the-opposite-order-optimizer-step-before-lr-scheduler-step/88295/6
        self.amp_scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.logger.info(f'use_amp: {self.use_amp}')

        self.SHUFFLE_Nodes = cfg.get('TRAIN.DATA.SHUFFLENODES', False)
        self.logger.info(f'TRAIN.DATA.SHUFFLENODES: {self.SHUFFLE_Nodes}')
        self.SHUFFLE_Nodes_val = cfg.get('VAL.DATA.SHUFFLENODES', False)
        self.logger.info(f'VAL.DATA.SHUFFLENODES: {self.SHUFFLE_Nodes_val}')
        self.SHUFFLE_Nodes_test = cfg.get('TEST.DATA.SHUFFLENODES', False)
        self.logger.info(f'TEST.DATA.SHUFFLENODES: {self.SHUFFLE_Nodes_test}')
        self.NUN_SHUFFLENODES = cfg.get('TEST.DATA.NUN_SHUFFLENODES', 1.0)
        self.logger.info(f'NUN_SHUFFLENODES: {self.NUN_SHUFFLENODES}')

        self.fintune_EPOCHS = cfg.get('TRAIN.fintune_EPOCHS', None)
        if self.fintune_EPOCHS is not None:
            self.logger.info(f'fintune_EPOCHS requires_grad_ after {self.fintune_EPOCHS-1}')

        self.if_compile = cfg.get('TRAIN.COMPILE_MODEL', False)

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """Training iteration process.

        Args:
            epoch (int): Current epoch.
            iter_index (int): Current iteration index.
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader.

        Returns:
            torch.Tensor: Loss value.
        """
        if self.fintune_EPOCHS is not None and self.fintune_EPOCHS == epoch and iter_index == 0:
            self.model.model.requires_grad_(True)
            self.logger.info('fintune_EPOCHS requires_grad_!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        if self.SHUFFLE_Nodes:
            # data = self.shuffle_nodes(data, rand_index=True)
            data = self.shuffle_nodes(data)

        if self.use_amp:
            with self.amp_context:
                forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)
                if self.cl_param:
                    cl_length = self.curriculum_learning(epoch=epoch)
                    forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
                    forward_return['target'] = forward_return['target'][:, :cl_length, :, :]
                # for k,v in forward_return.items():
                #     print(k, v.shape)
                loss = self.metric_forward(self.loss, forward_return)
                if 'raw_prediction' in forward_return.keys():
                    raw_loss = self.loss(forward_return['prediction']-forward_return['raw_prediction'], forward_return['target']-forward_return['raw_prediction'], self.null_val)
                    # loss = loss - raw_loss if loss > raw_loss else loss # ?
                    loss = (loss + raw_loss) / 2
        else:
            forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)
            if self.cl_param:
                cl_length = self.curriculum_learning(epoch=epoch)
                forward_return['prediction'] = forward_return['prediction'][:, :cl_length, :, :]
                forward_return['target'] = forward_return['target'][:, :cl_length, :, :]
            # for k,v in forward_return.items():
            #     print(k, v.shape)
            loss = self.metric_forward(self.loss, forward_return)

        self.update_epoch_meter('train/loss', loss.item())

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f'train/{metric_name}', metric_item.item())
        return loss
    
    def backward(self, loss: torch.Tensor):
        """Backward and update params.

        Args:
            loss (torch.Tensor): loss
        """
        self.optim.zero_grad()
        if self.use_amp:
            self.amp_scaler.scale(loss).backward()
            self.amp_scaler.unscale_(self.optim)
            # grad_norm = sum(
            #             param.grad.data.norm(2).item() ** 2 for param in self.model.parameters() if param.grad is not None
            #         ) ** 0.5
            if self.clip_grad_param is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), **self.clip_grad_param)
            self.amp_scaler.step(self.optim)
            self.amp_scaler.update()

        else:
            loss.backward()
            if self.clip_grad_param is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), **self.clip_grad_param)
            self.optim.step()

    def shuffle_nodes(self, input_data: Dict, rand_index=False) -> Dict:
        input_data['target'] = self.to_running_device(input_data['target'])  # Shape: [B, L, N, C]
        input_data['inputs'] = self.to_running_device(input_data['inputs'])    # Shape: [B, L, N, C]
        # if self.SHUFFLE_Nodes:
        with torch.no_grad():
            if rand_index:
                _index = torch.randperm(torch.randint(low=input_data['target'].size(2)//2, high=input_data['target'].size(2), size=torch.Size([1])).item(), device=input_data['target'].device)
            else:
                _num = int(self.NUN_SHUFFLENODES * input_data['target'].size(2))
                _index = torch.randperm(_num, device=input_data['target'].device)
                _index = torch.cat([_index, torch.arange(_num, input_data['target'].size(2), device=input_data['target'].device)], dim=0) if _num < input_data['target'].size(2) else _index
                # _index = torch.randperm(input_data['target'].size(2), device=input_data['target'].device)
            # if len(input_data['inputs'].shape)==3:
            #     input_data['target'] = input_data['target'][:, :, _index]
            #     input_data['inputs'] = input_data['inputs'][:, :, _index]
            # else:
            #     input_data['target'] = input_data['target'][:, :, _index, :]
            #     input_data['inputs'] = input_data['inputs'][:, :, _index, :]
            input_data['target'].data = input_data['target'].index_select(2, _index).contiguous()
            input_data['inputs'].data = input_data['inputs'].index_select(2, _index).contiguous()

        return input_data

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        """Validation iteration process.

        Args:
            iter_index (int): Current iteration index.
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader.
        """

        if self.SHUFFLE_Nodes_val:
            data = self.shuffle_nodes(data)
        # with torch.inference_mode(): # 会导致问题！！
        # with self.amp_context:
        
        if self.use_amp:
            with self.amp_context:
                forward_return = self.forward(data=data, epoch=None, iter_num=iter_index, train=False)
                loss = self.metric_forward(self.loss, forward_return)
        else:
            forward_return = self.forward(data=data, epoch=None, iter_num=iter_index, train=False)
            loss = self.metric_forward(self.loss, forward_return)
        self.update_epoch_meter('val/loss', loss.item())

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f'val/{metric_name}', metric_item.item())


    def build_model(self, cfg: Dict) -> nn.Module:
        """Build model.

        Initialize model by calling ```self.define_model```,
        Moves model to the GPU.

        If DDP is initialized, initialize the DDP wrapper.

        Args:
            cfg (Dict): config

        Returns:
            model (nn.Module)
        """

        self.logger.info('Building model.')
        model = self.define_model(cfg)
        model = self.to_running_device(model)

        # complie model
        if cfg.get('TRAIN.COMPILE_MODEL', False):
            # get current torch version
            current_version = torch.__version__
            # torch.compile() is only available in torch>=2.0
            if version.parse(current_version) >= version.parse('2.0'):
                self.logger.info('Compile model with torch.compile')
                model = torch.compile(model, dynamic=False) # 
                if hasattr(model, 'preload'):
                    model.preload()
                self.logger.info('Compiled')
            else:
                self.logger.warning(f'torch.compile requires PyTorch 2.0 or higher. Current version: {current_version}. Skipping compilation.')

        # DDP
        if torch.distributed.is_initialized():
            model = DDP(
                model,
                device_ids=[get_local_rank()],
                find_unused_parameters=cfg.get('MODEL.DDP_FIND_UNUSED_PARAMETERS', False)
            )
        return model
    
    def postprocessing(self, input_data: Dict) -> Dict:
        """Postprocess data.

        Args:
            input_data (Dict): Dictionary containing data to be processed.

        Returns:
            Dict: Processed data.
        """
        # if 'hidden' in input_data:
        #     _x, _y = input_data['hidden'], input_data['target']
        #     input_data['scaled_target'] = _y

        if len(input_data['inputs'].shape)==3: # 如果是训练hidden加速
            # rescale data
            if self.scaler is not None and self.scaler.rescale:
                input_data['prediction'] = self.scaler.inverse_transform(input_data['prediction'])
                input_data['target'] = self.scaler.inverse_transform(input_data['target'])
                # print('rescale!!! prediction and target')
        else:
            # rescale data
            if self.scaler is not None and self.scaler.rescale:
                input_data['prediction'] = self.scaler.inverse_transform(input_data['prediction'])
                input_data['target'] = self.scaler.inverse_transform(input_data['target'])
                input_data['inputs'] = self.scaler.inverse_transform(input_data['inputs'])

            # subset forecasting
            if self.target_time_series is not None:
                input_data['target'] = input_data['target'][:, :, self.target_time_series, :]
                input_data['prediction'] = input_data['prediction'][:, :, self.target_time_series, :]

        # TODO: add more postprocessing steps as needed.
        return input_data
    
    def preprocessing(self, input_data: Dict) -> Dict:
        """Preprocess data.

        Args:
            input_data (Dict): Dictionary containing data to be processed.

        Returns:
            Dict: Processed data.
        """
        if len(input_data['inputs'].shape)==3: # 如果是训练hidden加速
            # print(f'input_data target shape: {input_data["target"].shape}') # [64, 336, 207]
            # print(f'input_data inputs shape: {input_data["inputs"].shape}') # [32, 768, 207])
            # print('not transform')
            input_data['target'] = input_data['target'].unsqueeze(-1)
        else:
            if self.scaler is not None:
                input_data['target'] = self.scaler.transform(input_data['target'])
                input_data['inputs'] = self.scaler.transform(input_data['inputs'])
        # TODO: add more preprocessing steps as needed.
        return input_data
        
    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        """
        Selects target features based on the target features specified in the configuration.

        Args:
            data (torch.Tensor): Model prediction data with shape [B, L, N, C1].

        Returns:
            torch.Tensor: Data with selected target features and shape [B, L, N, C2].
        """

        if self.target_features is not None:
            data = data[:, :, :, self.target_features]
        return data
    
    def build_train_data_loader(self, cfg: Dict) -> DataLoader:
        """Build train dataset and dataloader.
        Build dataset by calling ```self.build_train_dataset```,
        build dataloader by calling ```build_data_loader``` or
        ```build_data_loader_ddp``` when DDP is initialized

        Args:
            cfg (Dict): config

        Returns:
            train data loader (DataLoader)
        """

        self.logger.info('Building training data loader.')
        dataset = self.build_train_dataset(cfg)
        if torch.distributed.is_initialized():
            return build_data_loader_ddp(dataset, cfg['TRAIN.DATA'])
        else:
            return build_data_loader_with_sampler(dataset, cfg['TRAIN.DATA'])


    def build_val_data_loader(self, cfg: Dict) -> DataLoader:
        """Build val dataset and dataloader.
        Build dataset by calling ```self.build_train_dataset```,
        build dataloader by calling ```build_data_loader```.

        Args:
            cfg (Dict): config

        Returns:
            val data loader (DataLoader)
        """

        self.logger.info('Building val data loader.')
        dataset = self.build_val_dataset(cfg)
        return build_data_loader_with_sampler(dataset, cfg['VAL.DATA'])
    
    def build_test_data_loader(self, cfg: Dict) -> DataLoader:
        """
        Build the test data loader.

        Args:
            cfg (Dict): Configuration dictionary.

        Returns:
            DataLoader: The test data loader.
        """

        dataset = self.build_test_dataset(cfg)
        return build_data_loader_with_sampler(dataset, cfg['TEST']['DATA'])
    
    
    @torch.no_grad()
    @master_only
    def test(self, train_epoch: Optional[int] = None, save_metrics: bool = False, save_results: bool = False) -> Dict:
        """Test process.
        
        Args:
            train_epoch (Optional[int]): Current epoch if in training process.
            save_metrics (bool): Save the test metrics. Defaults to False.
            save_results (bool): Save the test results. Defaults to False.
        """

        prediction, target, inputs = [], [], []

        for data in tqdm(self.test_data_loader):
            if self.SHUFFLE_Nodes_test:
                data = self.shuffle_nodes(data)
            with self.amp_context:
                forward_return = self.forward(data, epoch=None, iter_num=None, train=False)

                loss = self.metric_forward(self.loss, forward_return)
            self.update_epoch_meter('test/loss', loss.item())

            if not self.if_evaluate_on_gpu:
                forward_return['prediction'] = forward_return['prediction'].detach().cpu()
                forward_return['target'] = forward_return['target'].detach().cpu()
                forward_return['inputs'] = forward_return['inputs'].detach().cpu()

            prediction.append(forward_return['prediction'])
            target.append(forward_return['target'])
            inputs.append(forward_return['inputs'])

        prediction = torch.cat(prediction, dim=0)
        target = torch.cat(target, dim=0)
        inputs = torch.cat(inputs, dim=0)

        returns_all = {'prediction': prediction, 'target': target, 'inputs': inputs}
        metrics_results = self.compute_evaluation_metrics(returns_all)

        # save
        if save_results:
            # save returns_all to self.ckpt_save_dir/test_results.npz
            test_results = {k: v.cpu().numpy() for k, v in returns_all.items()}
            np.savez(os.path.join(self.ckpt_save_dir, 'test_results.npz'), **test_results)

        if save_metrics:
            # save metrics_results to self.ckpt_save_dir/test_metrics.json
            with open(os.path.join(self.ckpt_save_dir, 'test_metrics.json'), 'w') as f:
                json.dump(metrics_results, f, indent=4)

        return returns_all
    # def forward(self, data: Dict, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> Dict:
    #     """
    #     Performs the forward pass for training, validation, and testing. 

    #     Args:
    #         data (Dict): A dictionary containing 'target' (future data) and 'inputs' (history data) (normalized by self.scaler).
    #         epoch (int, optional): Current epoch number. Defaults to None.
    #         iter_num (int, optional): Current iteration number. Defaults to None.
    #         train (bool, optional): Indicates whether the forward pass is for training. Defaults to True.

    #     Returns:
    #         Dict: A dictionary containing the keys:
    #               - 'inputs': Selected input features.
    #               - 'prediction': Model predictions.
    #               - 'target': Selected target features.

    #     Raises:
    #         AssertionError: If the shape of the model output does not match [B, L, N].
    #     """
    #     print(self.target_features, self.forward_features, self.target_time_series)
    #     print(f'input_data target shape: {data["target"].shape} inputs shape: {data["inputs"].shape}') # [64, 336, 207]
    #     data = self.preprocessing(data)
    #     print(f'preprocessing target shape: {data["target"].shape} inputs shape: {data["inputs"].shape}') # [64, 336, 207]

    #     # Preprocess input data
    #     future_data, history_data = data['target'], data['inputs']
    #     history_data = self.to_running_device(history_data)  # Shape: [B, L, N, C]
    #     future_data = self.to_running_device(future_data)    # Shape: [B, L, N, C]
    #     batch_size, length, num_nodes = future_data.shape[:3]

    #     # Select input features
    #     history_data = self.select_input_features(history_data)
    #     future_data_4_dec = self.select_input_features(future_data)

    #     if not train:
    #         # For non-training phases, use only temporal features
    #         future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

    #     # Forward pass through the model
    #     print(f'forward history_data shape: {history_data.shape} future_data_4_dec shape: {future_data_4_dec.shape} {train}') # [64, 336, 207]
    #     model_return = self.model(history_data=history_data, future_data=future_data_4_dec,
    #                               batch_seen=iter_num, epoch=epoch, train=train)

    #     # Parse model return
    #     if isinstance(model_return, torch.Tensor):
    #         model_return = {'prediction': model_return}
    #     if 'inputs' not in model_return:
    #         model_return['inputs'] = self.select_target_features(history_data)
    #     if 'target' not in model_return:
    #         model_return['target'] = self.select_target_features(future_data)

    #     print(f'res target shape: {model_return["target"].shape} inputs shape: {model_return["inputs"].shape}') # [64, 336, 207]
    #     # Ensure the output shape is correct
    #     assert list(model_return['prediction'].shape)[:3] == [batch_size, length, num_nodes], \
    #         "The shape of the output is incorrect. Ensure it matches [B, L, N, C]."

    #     model_return = self.postprocessing(model_return)
    #     print(f'postprocessing target shape: {model_return["target"].shape} inputs shape: {model_return["inputs"].shape}') # [64, 336, 207]

    #     return model_return
    
    def load_model(self, ckpt_path: str = None, strict: bool = True) -> None:
        """Load model state dict.
        if param `ckpt_path` is None, load the last checkpoint in `self.ckpt_save_dir`,
        else load checkpoint from `ckpt_path`

        Args:
            ckpt_path (str, optional): checkpoint path, default is None
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
        """

        try:
            checkpoint_dict = load_ckpt(self.ckpt_save_dir, ckpt_path=ckpt_path, logger=self.logger)
            def deal_checkpoint_compile(state_dict):
                keys_list = list(state_dict.keys())
                for key in keys_list:
                    if 'orig_mod.' in key:
                        deal_key = key.replace('_orig_mod.', '')
                        state_dict[deal_key] = state_dict[key]
                        del state_dict[key]
                return state_dict
            if not self.if_compile:
                self.logger.info(checkpoint_dict['model_state_dict'].keys())
                checkpoint_dict['model_state_dict'] = deal_checkpoint_compile(checkpoint_dict['model_state_dict'])
                self.logger.info(self.model.state_dict().keys())
                self.logger.info(checkpoint_dict['model_state_dict'].keys())
            if isinstance(self.model, DDP):
                self.model.module.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)
            else:
                self.model.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)
        except (IndexError, OSError) as e:
            raise OSError('Ckpt file does not exist') from e