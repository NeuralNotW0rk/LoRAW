import torch
from torch import nn
from torch import optim
from enum import Enum

from .modules import LoRAWLinear, LoRAWConv1d


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
        self.lora_modules = nn.ModuleDict()
        # Scan model and create loras for respective modules
        for name, info in target_map.items():
            module = info["module"]
            self.lora_modules[name] = TargetableModules[
                module.__class__.__name__
            ].value(name, module)

    def activate(self, target_map):
        for name, module in self.lora_modules.items():
            module.inject(target_map[name]["parent"])
        self.active = True
        print(f"Injected {len(self.lora_modules)} LoRAW modules into model")

    def activate_forward(self):
        for _, module in self.lora_modules.items():
            module.inject_forward()
        self.active = True
        print(f"Forwarded {len(self.lora_modules)} LoRAW modules into model")

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for _, module in self.lora_modules.items():
            module.multiplier = self.multiplier


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

        # Get a list of bottom-level lora modues, excluding the originals
        self.residual_modules = nn.ModuleDict()
        for name, module in self.net.lora_modules.items():
            self.residual_modules[f"{name}/lora_up"] = module.lora_up
            self.residual_modules[f"{name}/lora_down"] = module.lora_down

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

        # Unfreeze lora modules
        for param in self.residual_modules.parameters():
            param.requires_grad = True

        # Move lora to training device
        self.net.to(device=training_wrapper.device)

        # Replace optimizer to use lora parameters
        if lr is None:
            self.lr = training_wrapper.lr
        else:
            self.lr = lr
        training_wrapper.configure_optimizers = self.configure_optimizers
        self.is_trainable = True

    def save_weights(self, path, dtype=torch.float16):
        torch.save(self.residual_modules.state_dict(), path)

    def load_weights(self, path):
        weights = torch.load(path, map_location="cpu")
        info = self.residual_modules.load_state_dict(weights, False)
        return info

    def merge_weights(self, path, multiplier=1.0):
        weights = torch.load(path, map_location="cpu")
        for name, weight in weights.items():
            param = self.residual_modules.state_dict()[name]
            param.copy_(param + weight * multiplier) 

    def extract_diff(self, tuned_model):
        lora_weights = calculate_svds(self.net.lora_modules, tuned_model, self.net.lora_modules.keys(), rank=self.net.lora_dim)
        for name, (up_weight, down_weight) in lora_weights.items():
            self.residual_modules[f"{name}/lora_up"].weight.copy_(up_weight)
            self.residual_modules[f"{name}/lora_down"].weight.copy_(down_weight)


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
                    module_map[id] = {
                        "module": decendant_module,
                        "parent": ancestor_module,
                    }
    print(f"Found {len(module_map)} candidates for LoRAW replacement")
    return module_map

CLAMP_QUANTILE = 0.99
MIN_DIFF = 1e-6

def calculate_svds(model_original, model_tuned, lora_names, rank):
    map_o = {name.replace('.', '/'): module for name, module in model_original.named_modules()}
    map_t = {name.replace('.', '/'): module for name, module in model_tuned.named_modules()}

    # Get diffs
    diffs = {}
    for name in lora_names:
        diff = map_t[name].weight - map_o[name].weight
        diff = diff.float()
        diffs[name] = diff

    # Calculate SVD
    lora_weights = {}
    with torch.no_grad():
        for lora_name, mat in list(diffs.items()):
            conv1d = len(mat.size()) == 3
            kernel_size = None if not conv1d else mat.size()[2]
            out_dim, in_dim = mat.size()[0:2]

            mat_padded = torch.zeros(max(out_dim, rank), max(in_dim, rank), kernel_size)
            mat_padded[:out_dim, :in_dim] = mat

            if conv1d:
                if kernel_size != 1:
                    mat_padded = mat_padded.flatten(start_dim=1)
                else:
                    mat_padded = mat_padded.squeeze()

            U, S, Vh = torch.linalg.svd(mat_padded)

            U = U[:, :rank]
            S = S[:rank]
            U = U @ torch.diag_embed(S)

            Vh = Vh[:rank, :]

            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi_val = torch.quantile(dist, CLAMP_QUANTILE)
            low_val = -hi_val

            U = U.clamp(low_val, hi_val)
            Vh = Vh.clamp(low_val, hi_val)

            if conv1d:
                U = U.reshape(max(out_dim, rank), rank, 1)
                Vh = Vh.reshape(rank, max(in_dim, rank), kernel_size)

            U = U.to("cpu").contiguous()
            Vh = Vh.to("cpu").contiguous()

            # Record lora_up and lora_down weights
            lora_weights[lora_name] = (U[:out_dim], Vh[:, :in_dim])

    return lora_weights
    