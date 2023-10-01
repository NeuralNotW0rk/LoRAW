import torch
from torch import nn
from typing import List

from .module import *


class LoRAWNetwork(nn.Module):
    def __init__(
        self,
        target_map,
        id,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
    ):
        super().__init__()
        self.id = id
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.dropout = dropout
        self.lora_map = nn.ModuleDict()
        # Scan model and create loras for respective modules
        for name, module_dict in target_map.items():
            module = module_dict["module"]
            module_type = module.__class__.__name__
            if module_type == "Linear":
                self.lora_map[name] = LoRAWLinear(name, module)
            elif module_type == "Conv1d":
                self.lora_map[name] = LoRAWConv1d(name, module)

    def activate(self, target_map):
        for name, lora in self.lora_map.items():
            lora.inject(target_map[name]["parent"])
        print(f"Injected {len(self.lora_map)} LoRAW modules into model")

    def activate_forward(self):
        for _, lora in self.lora_map.items():
            lora.inject_forward()
        print(f"Forwarded {len(self.lora_map)} LoRAW modules into model")

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.unet_loras:
            lora.multiplier = self.multiplier

    def is_mergeable(self):
        return True

    def save_weights(self, file, dtype=torch.float16):
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
