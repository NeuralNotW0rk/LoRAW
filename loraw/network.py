import torch
from torch import nn
from typing import List
from enum import Enum

from .module import *

class TargetableModules(Enum):
    Linear = LoRAWLinear
    Conv1d = LoRAWConv1d


class LoRAWNetwork(nn.Module):
    def __init__(
        self,
        target_map,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
    ):
        super().__init__()
        self.active = False
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.dropout = dropout
        self.lora_map = nn.ModuleDict()
        # Scan model and create loras for respective modules
        for name, module_dict in target_map.items():
            module = module_dict["module"]
            self.lora_map[name] = TargetableModules[module.__class__.__name__].value(name, module)

    def activate(self, target_map):
        for name, lora in self.lora_map.items():
            lora.inject(target_map[name]["parent"])
        self.active = True
        print(f"Injected {len(self.lora_map)} LoRAW modules into model")

    def activate_forward(self):
        for _, lora in self.lora_map.items():
            lora.inject_forward()
        self.active = True
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


class LoRAWWrapper:
    def __init__(
        self,
        target_model,
        target_blocks=["Attention"],
        component_whitelist=None,
        lora_dim=16,
        alpha=1.0,
        dropout=None,
        multiplier=1.0,
    ):
        self.target_model = target_model
        self.target_blocks = target_blocks
        self.component_whitelist = component_whitelist
        self.active = False
        self.trainable = False

        # Gather candidates for replacement
        self.target_map = scan_model(
            target_model, target_blocks, whitelist=component_whitelist
        )

        # Construct LoRAW network
        self.net = LoRAWNetwork(
            self.target_map,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            multiplier=multiplier,
        )

    def activate(self):
        assert not self.active, "LoRAW is already active"
        self.net.activate(self.target_map)
        self.active = True

    def configure_optimizers(self):
        return optim.Adam([*self.net.parameters()], lr=self.lr)

    def prepare_for_training(self, training_wrapper, lr=None):
        assert self.active, "LoRAW must be activated before training preparation"

        # Freeze target model
        self.target_model.requires_grad_(False)

        # Unfreeze lora modules
        self.net.requires_grad_(True)

        # Move lora to training device
        self.net.to(device=training_wrapper.device)

        # Replace optimizer to use lora parameters
        if lr is None:
            self.lr = training_wrapper.lr
        else:
            self.lr = lr
        training_wrapper.configure_optimizers = self.configure_optimizers()
        self.trainable = True


def scan_model(model, target_blocks, whitelist=None, blacklist=None):
    # Find all modules that are in targeted blocks
    # If a whitelist is specified, modules must have at least one whitelisted ancestor
    # If a blacklist is specified, modules must have no blacklisted ancestors
    target_blocks = set(target_blocks)
    whitelist = set(whitelist) if whitelist is not None else None
    blacklist = set(blacklist) if blacklist is not None else None
    module_map = {}
    for parent_name, parent_module in model.named_modules():
        parent_name_split = set(parent_name.split("."))
        if (
            parent_module.__class__.__name__ in target_blocks
            and (whitelist is None or not parent_name_split.isdisjoint(whitelist))
            and (blacklist is None or parent_name_split.isdisjoint(blacklist))
        ):
            for child_name, child_module in parent_module.named_modules():
                if child_module.__class__.__name__ in TargetableModules.__members__:
                    for name in child_name.split(".")[:-1]:
                        parent_module = parent_module._modules[name]
                    # Since '.' is not allowed, replace with '/' (makes it look like a path)
                    id = f"{parent_name}.{child_name}".replace(".", "/")
                    module_map[id] = {"module": child_module, "parent": parent_module}
    print(f"Found {len(module_map)} candidates for LoRAW replacement")
    return module_map
