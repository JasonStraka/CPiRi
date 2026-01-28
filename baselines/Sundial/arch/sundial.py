import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from .modeling_sundial import SundialForPrediction
from .flow_loss import FlowLoss
from .configuration_sundial import SundialConfig
from .ts_generation_mixin import TSGenerationMixin


class Sundial(nn.Module):
    def __init__(self, model_id: str, from_pretrained: bool,
                    context_length: int,
                    trust_remote_code: bool):
        
        super().__init__()
        
        self.model_type = 'causal' # TimeMoE is a causal model
        self.context_length = context_length
        
        if from_pretrained:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
            )
        else:
            kwargs = {}
            kwargs['torch_dtype'] = 'float32'
            # kwargs['attn_implementation'] = 'flash_attention_2'
            config, model_kwargs = SundialConfig.from_pretrained(
                                    pretrained_model_name_or_path=model_id,
                                    return_unused_kwargs=True,
                                    **kwargs)
            # print(f'Using attention implementation: {kwargs.get("attn_implementation", "original")}')
            self.model = SundialForPrediction(config)
            from safetensors.torch import load_model, save_model
            load_model(self.model, "baselines/Sundial/ckpt/model.safetensors")
            # download from： model = AutoModelForCausalLM.from_pretrained('thuml/sundial-base-128m', trust_remote_code=True) 
        self.chunk_size = self.model.config.input_token_len
        self.output_token_len = self.model.config.output_token_lens[-1]

    def forward(self, context: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, target_mask: torch.Tensor, training=True):

        # print(mask.sum(), target_mask.sum())
        # print('===========================', context.size(), target.size(), mask.size(), target_mask.size()) 
        # torch.Size([64, 4096]) torch.Size([64, 720]) torch.Size([64, 4096]) torch.Size([64, 720])
        labels = torch.cat([context[:, self.chunk_size:], target], dim=-1).detach() # torch.Size([64, 4800])
        # loss_masks = torch.cat([mask[:, self.chunk_size:], target_mask], dim=-1)
        # loss_masks = loss_masks.unfold(dimension=-1, size=self.output_token_len, step=self.chunk_size).any(dim=-1).detach() # torch.Size([64, 256, 720])  torch.Size([64, 256]) 
        # mask = self.mask_pre(mask).detach() # torch.Size([64, 256])
        # mask = self.mask_pre(torch.isnan(context)).detach() # torch.Size([64, 256])
        # context, labels = context.nan_to_num(), labels.nan_to_num()
        context = context.nan_to_num()
        
        # 注意，这里attention_mask和BLAST数据集中的mask是反过来的，需要进行取反~，否则loss会nan；但是loss_masks不需要
        # attention_mask将输入1设为负无穷，True就是对应nan
        # 数据集中已经归一化，这里不需要再次revin
        output, _mae, _mse, _trend_diff = self.model(input_ids=context, labels=labels, attention_mask=None, loss_masks=None, mask_y=None, revin=False, training = training)
        # output, _mae, _mse = self.model(input_ids=context, labels=labels, attention_mask=~mask, loss_masks=loss_masks, mask_y=None, revin=False)
        # output = self.model(input_ids=context, labels=target, loss_masks=target_mask)

        loss, _ = output.loss, output.logits # _ is the logits

        return loss, _mae, _mse, _trend_diff

    def generate(self, context: torch.Tensor, prediction_length: int, normalize: bool = False):
        if normalize:
            mean, std = context.mean(dim=-1, keepdim=True), context.std(dim=-1, keepdim=True)
            std[std == 0] = 1
            context = (context - mean) / std
        
        predictions = self.model.generate(
            input_ids=context,
            max_new_tokens=prediction_length
        )
        predictions = predictions[:, -prediction_length:]

        if normalize:
            predictions = predictions * std + mean

        return predictions
    
    def mask_pre(self, mask):
        B, L = mask.shape
        # mask = np.logical_not(np.isnan(inputs)) not any all
        compressed_mask = mask.reshape(B, L // self.chunk_size, self.chunk_size).any(dim=2)
        # print(mask.size(), compressed_mask.size())
        return compressed_mask
