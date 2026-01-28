import os
import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))
from baselines.Sundial.arch.modeling_sundial import SundialForPrediction, SundialDecoderLayer
from baselines.Sundial.arch.configuration_sundial import SundialConfig
import torch
from torch import nn
import tqdm

class CPiRi(nn.Module):

    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.output_len = model_args["output_len"]
        self.spital_num_layers = model_args.get("spital_num_layers", 4)
        self.revin = model_args.get("revin", False)
        self.from_pretrain = model_args.get("from_pretrain", True)
        
        self.model_type = 'causal'
        self.context_length = self.input_len
        kwargs = {}
        kwargs['torch_dtype'] = 'float32'
        # kwargs['attn_implementation'] = 'flash_attention_2'
        config, model_kwargs = SundialConfig.from_pretrained(
                                pretrained_model_name_or_path="baselines/Sundial/arch",
                                return_unused_kwargs=True,
                                **kwargs)
        # print(f'Using attention implementation: {kwargs.get("attn_implementation", "original")}')
        self._config = config
        self.model = SundialForPrediction(config)
        from safetensors.torch import load_model, save_model
        if self.from_pretrain:
            load_model(self.model, "baselines/Sundial/ckpt/model.safetensors")
            self.model.requires_grad_(False)
        self.chunk_size = self.model.config.input_token_len
        self.output_token_len = self.model.config.output_token_lens[-1]
        self.spital_att = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads, dim_feedforward=config.intermediate_size, batch_first=True, norm_first=True, activation="gelu", dropout=0.3), num_layers=self.spital_num_layers, enable_nested_tensor=False, norm=nn.LayerNorm(config.hidden_size))

        for name, param in self.model.named_parameters():
            if 'spital_att' not in name:
                param.requires_grad = False

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None, epoch: int = None, train: bool = True, **kwargs) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """
        input_data = history_data[..., range(self.input_dim)]
        batch_size, L, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous() # B, N, L, C
        input_data = input_data.view(batch_size * num_nodes, -1)
        
        if self.revin:
            means = input_data.mean(1, keepdim=True).detach()
            stdev = input_data.std(dim=1, keepdim=True, unbiased=False).detach()
            stdev = torch.where(stdev > 1e-2, stdev, torch.tensor(1.0, device=input_data.device))
            input_data = (input_data - means) / stdev
        hidden = self.model.model(input_data).last_hidden_state
        last_hidden_state = hidden[:, -1, :]
        
        hidden = self.spital_att(last_hidden_state.view(batch_size, num_nodes, -1)).view(batch_size * num_nodes, -1)
        
        if train:
            prediction = self.model.flow_loss.sample(hidden, num_samples=1).squeeze(1)[:, :self.output_len]
        else:
            prediction = self.model.flow_loss.sample(hidden, 20).mean(dim=1)[:, :self.output_len]
            
        if self.revin:
            prediction = prediction * stdev + means
        prediction = prediction.view(batch_size, num_nodes, -1, 1).transpose(1, 2).contiguous()
        
        return prediction

class Sundial(nn.Module):

    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.output_len = model_args["output_len"]
        self.revin = model_args.get("revin", False)
        
        # self.model_type = 'causal' # TimeMoE is a causal model
        self.context_length = self.input_len
        kwargs = {}
        kwargs['torch_dtype'] = 'float32'
        # kwargs['attn_implementation'] = 'flash_attention_2'
        config, model_kwargs = SundialConfig.from_pretrained(
                                pretrained_model_name_or_path="baselines/Sundial/arch",
                                return_unused_kwargs=True,
                                **kwargs)
        self._config = config
        self.model = SundialForPrediction(config)
        from safetensors.torch import load_model, save_model
        load_model(self.model, "baselines/Sundial/ckpt/model.safetensors")
        self.model.requires_grad_(False)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None, epoch: int = None, train: bool = True, **kwargs) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """
        input_data = history_data[..., range(self.input_dim)]
        batch_size, L, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous() # B, N, L, C
        input_data = input_data.view(batch_size * num_nodes, -1)
        
        if self.revin:
            means = input_data.mean(1, keepdim=True).detach()
            stdev = input_data.std(dim=1, keepdim=True, unbiased=False).detach()
            stdev = torch.where(stdev > 1e-2, stdev, torch.tensor(1.0, device=input_data.device))
            input_data = (input_data - means) / stdev
        hidden = self.model.model(input_data).last_hidden_state[:, -1, :]
        if train:
            prediction = self.model.flow_loss.sample(hidden, num_samples=1).squeeze(1)[:, :self.output_len]
        else:
            prediction = self.model.flow_loss.sample(hidden, 20).mean(dim=1)[:, :self.output_len]

        if self.revin:
            prediction = prediction * stdev + means
        prediction = prediction.view(batch_size, num_nodes, -1, 1).transpose(1, 2).contiguous()

        return prediction


if __name__ == "__main__":
    model = CPiRi(input_len=336, input_dim=1, output_len=336).cuda()
    from torchinfo import summary
    summary(model, (1, 336, 207, 3))
    with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        print(model(torch.rand([128, 336, 207, 3]).cuda(), train=False).shape)
        print(torch.cuda.memory_summary())
        torch.cuda.empty_cache()
        print(model(torch.rand([128, 336, 207, 3]).cuda()).shape)
        print(torch.cuda.memory_summary())
        torch.cuda.empty_cache()
        import time
        for batch in [1, 8, 16, 32, 64, 128]: # 128 最快
            data = torch.rand([batch, 336, 207, 1]).cuda()
            _times = 5
            model(data)
            _t = time.time()
            for i in range(_times):
                model(data)
            print(batch, (time.time() - _t)/batch/_times)
            torch.cuda.empty_cache()