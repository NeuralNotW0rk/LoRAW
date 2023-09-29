import torch
from torch import nn
from typing import List

from .module import *

TARGETABLE_MODULES = ["Linear", "Conv1d"]


def scan_model(model, target_blocks, whitelist=None, blacklist=None):
    # Find all modules that are in targeted blocks
    # If a whitelist is specified, modules must have at least one whitelisted ancestor
    # If a blacklist is specified, modules must have no blacklisted ancestors
    target_blocks = set(target_blocks)
    whitelist = set(whitelist) if whitelist is not None else None
    blacklist = set(blacklist) if blacklist is not None else None
    module_map = {}
    for parent_name, parent_module in model.named_modules():
        split_name = set(parent_name.split("."))
        if (
            parent_module.__class__.__name__ in target_blocks
            and (whitelist is None or not split_name.isdisjoint(whitelist))
            and (blacklist is None or split_name.isdisjoint(blacklist))
        ):
            for child_name, child_module in parent_module.named_modules():
                if child_module.__class__.__name__ in TARGETABLE_MODULES:
                    # Since '.' is not allowed, replace with '/' (makes it look like a path)
                    id = f"{parent_name}.{child_name}".replace(".", "/")
                    module_map[id] = (child_module, parent_module)
    print(f'Found {len(module_map)} candidates for LoRAW replacement')
    return module_map


class LoRAWNetwork(nn.Module):
    def __init__(
        self,
        model,
        target_blocks=["Attention"],
        component_whitelist=None,
        component_blacklist=None,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
    ):
        super().__init__()

        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.dropout = dropout
        self.lora_map = nn.ModuleDict()
        # Scan model and create loras for respective modules
        for name, (module, parent) in scan_model(
            model,
            target_blocks=target_blocks,
            whitelist=component_whitelist,
            blacklist=component_blacklist,
        ).items():
            module_type = module.__class__.__name__
            if module_type == "Linear":
                self.lora_map[name] = (LoRAWLinear(name, module, parent))
            elif module_type == "Conv1d":
                self.lora_map[name] = (LoRAWConv1d(name, module, parent))

    def activate(self):
        for lora in self.lora_map.values():
            lora.inject()
        print(f"Injected {len(self.lora_map)} LoRAW modules into model")

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
