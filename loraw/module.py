import math
import torch
from torch import nn


class LoRAWModule(nn.Module):
    def __init__(
        self,
        lora_name,
        original_module: nn.Module,
        multiplier=1.0,
        lora_dim=16,
        alpha=1.0,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
    ):
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        self.multiplier = multiplier
        self.original_module = original_module
        self.original_forward = original_module.forward
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim

    def forward(self, x):
        lx = self.lora_down(x)
        lx = self.lora_up(lx)
        return self.original_forward(x) + lx * self.scale * self.multiplier

    def inject(self, parent_module):
        parent_module._modules[self.lora_name.split("/")[-1]] = self

    def inject_forward(self):
        self.original_module.forward = self.forward
        del self.original_module


class LoRAWLinear(LoRAWModule):
    def __init__(
        self,
        lora_name,
        original_module: nn.Module,
        multiplier=1,
        lora_dim=16,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
    ):
        super().__init__(
            lora_name,
            original_module,
            multiplier,
            lora_dim,
            alpha,
            dropout,
            rank_dropout,
            module_dropout,
        )
        in_dim = original_module.in_features
        out_dim = original_module.out_features
        self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
        self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)


class LoRAWConv1d(LoRAWModule):
    def __init__(
        self,
        lora_name,
        original_module: nn.Module,
        multiplier=1,
        lora_dim=16,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
    ):
        super().__init__(
            lora_name,
            original_module,
            parent_module,
            multiplier,
            lora_dim,
            alpha,
            dropout,
            rank_dropout,
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


# Original implementation for reference
class LoRAWModuleForward(nn.Module):
    def __init__(
        self,
        lora_name,
        orig_module: nn.Module,
        multiplier=1.0,
        lora_dim=16,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
    ):
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        self.multiplier = multiplier
        self.orig_module = orig_module  # remove in applying
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        self.lora_up = None
        self.lora_down = None

        module_type = orig_module.__class__.__name__

        if module_type == "Conv1d":
            in_dim = orig_module.in_channels
            out_dim = orig_module.out_channels
            kernel_size = orig_module.kernel_size
            stride = orig_module.stride
            padding = orig_module.padding
            self.lora_down = torch.nn.Conv1d(
                in_dim, self.lora_dim, kernel_size, stride, padding, bias=False
            )
            self.lora_up = torch.nn.Conv1d(self.lora_dim, out_dim, 1, 1, bias=False)
        elif module_type == "Linear":
            in_dim = orig_module.in_features
            out_dim = orig_module.out_features
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

    def activate(self):
        self.orig_forward = self.orig_module.forward
        self.orig_module.forward = self.forward
        del self.orig_module

    def forward(self, x):
        orig_forwarded = self.orig_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return orig_forwarded

        lx = self.lora_down(x)

        # normal dropout
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        # rank dropout
        if self.rank_dropout is not None and self.training:
            mask = (
                torch.rand((lx.size(0), self.lora_dim), device=lx.device)
                > self.rank_dropout
            )
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)  # for Text Encoder
            elif len(lx.size()) == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)  # for Conv2d
            lx = lx * mask

            # scaling for rank dropout: treat as if the rank is changed
            # maskから計算することも考えられるが、augmentation的な効果を期待してrank_dropoutを用いる
            scale = self.scale * (
                1.0 / (1.0 - self.rank_dropout)
            )  # redundant for readability
        else:
            scale = self.scale

        lx = self.lora_up(lx)

        return orig_forwarded + lx * self.multiplier * scale
