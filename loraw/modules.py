import math
import torch
from torch import nn


class LoRAModule(nn.Module):
    def __init__(
        self,
        lora_name,
        original_module: nn.Module,
        multiplier=1.0,
        lora_dim=16,
        alpha=1.0,
        dropout=None,
        module_dropout=None
    ):
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        self.multiplier = multiplier
        self.original_module = original_module
        self.dropout = dropout
        self.module_dropout = module_dropout

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
    
    def init_weights(self):
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        # Module dropout (skip lora module)
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.original_module(x)

        # Down to low-rank
        lx = self.lora_down(x)

        # Regular dropout
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        # Back up to full-rank
        lx = self.lora_up(lx)

        # Add scaled residual to original
        return self.original_module(x) + lx * self.scale * self.multiplier
    
    def inject(self, parent_module):
        # Replace original module with lora module
        parent_module._modules[self.lora_name.split("/")[-1]] = self
        # Move original params to lora module
        self.weight = nn.Parameter(data=self.original_module.weight.clone().detach(), requires_grad=False)
        self.original_module.weight = self.weight
            
    def quantize(self):
        self.original_module = torch.ao.quantization.quantize_dynamic(
            self.original_module,
            {nn.Linear},
            dtype=torch.qint8
        )
        del self.weight

    def dump_weights(self):
        # Update original module weights
        updated = self.weight.clone().detach() + self.lora_up.weight.clone().detach() @ self.lora_down.weight.clone().detach() * self.scale
        self.weight.data = updated

        # Reinit lora weights
        self.init_weights()

        # Update quantized module if present
        if self.original_module_q is not None:
            self.quantize()


class LoRALinear(LoRAModule):
    def __init__(
        self,
        lora_name,
        original_module: nn.Module,
        multiplier=1,
        lora_dim=16,
        alpha=1,
        dropout=None,
        module_dropout=None,
    ):
        super().__init__(
            lora_name,
            original_module,
            multiplier,
            lora_dim,
            alpha,
            dropout,
            module_dropout,
        )
        in_dim = original_module.in_features
        out_dim = original_module.out_features
        self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
        self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)
        self.init_weights()


class LoRAConv1d(LoRAModule):
    def __init__(
        self,
        lora_name,
        original_module: nn.Module,
        multiplier=1,
        lora_dim=16,
        alpha=1,
        dropout=None,
        module_dropout=None,
    ):
        super().__init__(
            lora_name,
            original_module,
            multiplier,
            lora_dim,
            alpha,
            dropout,
            module_dropout,
        )
        in_dim = original_module.in_channels
        out_dim = original_module.out_channels
        kernel_size = original_module.kernel_size
        stride = original_module.stride
        padding = original_module.padding
        self.lora_down = torch.nn.Conv1d(
            in_dim, self.lora_dim, kernel_size, stride, padding, bias=False
        )
        self.lora_up = torch.nn.Conv1d(self.lora_dim, out_dim, 1, 1, bias=False)
        self.init_weights()
