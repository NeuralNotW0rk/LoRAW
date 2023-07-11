# LoRA network module
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import math
import os
from typing import List, Tuple, Union
import numpy as np
import torch


class AudioLoRAModule(torch.nn.Module):
    def __init__(
        self,
        lora_name,
        orig_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
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

        if orig_module.__class__.__name__ == "Conv1d":
            in_dim = orig_module.in_channels
            out_dim = orig_module.out_channels
            kernel_size = orig_module.kernel_size
            stride = orig_module.stride
            padding = orig_module.padding
            self.lora_down = torch.nn.Conv1d(
                in_dim, self.lora_dim, kernel_size, stride, padding, bias=False
            )
            self.lora_up = torch.nn.Conv1d(self.lora_dim, out_dim, 1, 1, bias=False)
        else:
            in_dim = orig_module.in_features
            out_dim = orig_module.out_features
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

    def apply_to(self):
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


class AudioLoRANetwork(torch.nn.Module):
    NUM_OF_BLOCKS = 12
    # Only target self attention blocks by default
    UNET1D_TARGET_REPLACE_MODULE = ["SelfAttention1d"]
    LORA_PREFIX_UNET = "lora_unet"

    def __init__(
        self,
        unet,
        target_modules=UNET1D_TARGET_REPLACE_MODULE,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        module_class=AudioLoRAModule,
        verbose=False,
    ):
        super().__init__()

        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha

        def create_modules(
            root_module: torch.nn.Module, target_replace_modules
        ) -> torch.nn.ModuleList:
            prefix = AudioLoRANetwork.LORA_PREFIX_UNET
            loras = torch.nn.ModuleList()
            skipped = torch.nn.ModuleList()
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv1d = child_module.__class__.__name__ == "Conv1d"

                        if is_linear or is_conv1d:
                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")

                            lora = module_class(
                                lora_name,
                                child_module,
                                self.multiplier,
                                self.lora_dim,
                                self.alpha,
                            )
                            loras.append(lora)
            return loras, skipped

        self.unet_loras, skipped = create_modules(unet, target_modules)
        print(f"create LoRA for U-Net1D: {len(self.unet_loras)} modules.")

        if verbose and len(skipped) > 0:
            print(
                f"because block_lr_weight is 0 or dim (rank) is 0, {len(skipped)} LoRA modules are skipped / block_lr_weightまたはdim (rank)が0の為、次の{len(skipped)}個のLoRAモジュールはスキップされます:"
            )
            for name in skipped:
                print(f"\t{name}")

        self.up_lr_weight: List[float] = None
        self.down_lr_weight: List[float] = None
        self.mid_lr_weight: float = None
        self.block_lr = False

        # assertion
        names = set()
        for lora in self.unet_loras:
            assert (
                lora.lora_name not in names
            ), f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.unet_loras:
            lora.multiplier = self.multiplier

    def apply_to(self):
        for lora in self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    def is_mergeable(self):
        return True

    def save_weights(self, file, dtype):

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        torch.save(state_dict, file)

    def load_weights(self, file):
        weights_sd = torch.load(file, map_location="cpu")

        info = self.load_state_dict(weights_sd, False)
        return info
