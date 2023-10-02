import torch
from torch import nn
from torch import optim
from enum import Enum

from .module import LoRAWLinear, LoRAWConv1d

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
        self.loraw_modules = nn.ModuleDict()
        # Scan model and create loraws for respective modules
        for name, info in target_map.items():
            module = info["module"]
            self.loraw_modules[name] = TargetableModules[module.__class__.__name__].value(name, module)

    def activate(self, target_map):
        for name, module in self.loraw_modules.items():
            module.inject(target_map[name]["parent"])
        self.active = True
        print(f"Injected {len(self.loraw_modules)} LoRAW modules into model")

    def activate_forward(self):
        for _, module in self.loraw_modules.items():
            module.inject_forward()
        self.active = True
        print(f"Forwarded {len(self.loraw_modules)} LoRAW modules into model")

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for _, module in self.loraw_modules.items():
            module.multiplier = self.multiplier

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
        self.is_active = False
        self.is_trainable = False

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

        # Get a list of bottom-level loraw modues, excluding the originals
        self.residual_modules = nn.ModuleDict()
        for name, module in self.net.loraw_modules.items():
            self.residual_modules[f'{name}/lora_up'] = module.lora_up
            self.residual_modules[f'{name}/lora_down'] = module.lora_down

    def activate(self):
        assert not self.is_active, "LoRAW is already active"
        self.net.activate(self.target_map)
        self.is_active = True

    def configure_optimizers(self):
        return optim.Adam([*self.residual_modules.parameters()], lr=self.lr)

    def prepare_for_training(self, training_wrapper, lr=None):
        assert self.is_active, "LoRAW must be activated before training preparation"

        # Freeze target model
        for param in self.target_model.parameters():
            param.requires_grad = False

        # Unfreeze loraw modules
        for param in self.residual_modules.parameters():
            param.requires_grad = True

        # Move loraw to training device
        self.net.to(device=training_wrapper.device)

        # Replace optimizer to use loraw parameters
        if lr is None:
            self.lr = training_wrapper.lr
        else:
            self.lr = lr
        training_wrapper.configure_optimizers = self.configure_optimizers
        self.is_trainable = True


def scan_model(model, target_blocks, whitelist=None, blacklist=None):
    # Find all targetable modules that are in targeted blocks
    # If a whitelist is specified, modules must have at least one whitelisted ancestor
    # If a blacklist is specified, modules must have no blacklisted ancestors
    target_blocks = set(target_blocks)
    whitelist = set(whitelist) if whitelist is not None else None
    blacklist = set(blacklist) if blacklist is not None else None
    module_map = {}
    for ancestor_name, ancestor_module in model.named_modules():
        ancestor_set = set(ancestor_name.split("."))
        if (
            ancestor_module.__class__.__name__ in target_blocks
            and (whitelist is None or not ancestor_set.isdisjoint(whitelist))
            and (blacklist is None or ancestor_set.isdisjoint(blacklist))
        ):
            for decendant_name, decendant_module in ancestor_module.named_modules():
                if decendant_module.__class__.__name__ in TargetableModules.__members__:
                    # Get parent if child is not a direct decendant
                    for name in decendant_name.split(".")[:-1]:
                        ancestor_module = ancestor_module._modules[name]
                    # Since '.' is not allowed, replace with '/' (makes it look like a path)
                    id = f"{ancestor_name}.{decendant_name}".replace(".", "/")
                    module_map[id] = {"module": decendant_module, "parent": ancestor_module}
    print(f"Found {len(module_map)} candidates for LoRAW replacement")
    return module_map
