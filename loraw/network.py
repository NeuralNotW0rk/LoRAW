import torch
from torch import nn
from torch import optim
from enum import Enum

from stable_audio_tools.models.utils import load_ckpt_state_dict

from .modules import LoRALinear, LoRAConv1d
from .util import *
from .attributes import *



class TargetableModules(Enum):
    Linear = LoRALinear
    Conv1d = LoRAConv1d


def scan_model(model, whitelist=None, blacklist=None):
    # If a whitelist is specified, modules must have at least one whitelisted ancestor
    whitelist = set(whitelist) if whitelist is not None else None
    # If a blacklist is specified, modules must have no blacklisted ancestors
    blacklist = set(blacklist) if blacklist is not None else None
    module_map = {}
    for decendant_name, decendant_module in model.named_modules():
        if decendant_module.__class__.__name__ in TargetableModules.__members__:
            ancestor_set = set(decendant_name.split("."))
            if (
                (whitelist is None or not ancestor_set.isdisjoint(whitelist))
                and (blacklist is None or ancestor_set.isdisjoint(blacklist))
            ):
                # Get parent if child is not a direct decendant
                ancestor_module = model
                for name in decendant_name.split(".")[:-1]:
                    ancestor_module = ancestor_module._modules[name]
                # Since '.' is not allowed, replace with '/' (makes it look like a path)
                id = decendant_name.replace(".", "/")
                module_map[id] = {
                    "module": decendant_module,
                    "parent": ancestor_module,
                }
    print(f"Found {len(module_map)} candidates for LoRA replacement")
    return module_map

    
def scan_model_by_block(model, target_blocks, whitelist=None, blacklist=None):
    # Find all targetable modules that are in targeted blocks
    target_blocks = set(target_blocks)
    # If a whitelist is specified, modules must have at least one whitelisted ancestor
    whitelist = set(whitelist) if whitelist is not None else None
    # If a blacklist is specified, modules must have no blacklisted ancestors
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
    print(f"Found {len(module_map)} candidates for LoRA replacement")
    return module_map


class LoRANetwork(nn.Module):
    def __init__(
        self,
        target_map,
        multiplier=1.0,
        lora_dim=16,
        alpha=16,
        dropout=None,
        module_dropout=None,
        decompose=False
    ):
        super().__init__()
        self.active = False
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.dropout = dropout
        self.module_dropout = module_dropout
        self.lora_modules = nn.ModuleDict()
        # Scan model and create loras for respective modules
        for name, info in target_map.items():
            module = info["module"]
            self.lora_modules[name] = TargetableModules[
                module.__class__.__name__
            ].value(
                name,
                module,
                multiplier=multiplier,
                lora_dim=lora_dim,
                alpha=alpha,
                dropout=dropout,
                module_dropout=module_dropout,
                decompose=decompose
            )

    def activate(self, target_map):
        for name, module in self.lora_modules.items():
            module.inject(target_map[name]["parent"])
        self.active = True
        print(f"Injected {len(self.lora_modules)} LoRA modules into model")

    def activate_forward(self):
        for _, module in self.lora_modules.items():
            module.inject_forward()
        self.active = True
        print(f"Forwarded {len(self.lora_modules)} LoRA modules into model")

    def quantize_base(self):
        for _, module in self.lora_modules.items():
            module.quantize()
        print(f'Base model weights quantized')
        
    def update_base(self):
        for name, module in self.lora_modules.items():
            module.dump_weights()
        print(f"Base model weights updated and LoRA modules reinitialized")

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for _, module in self.lora_modules.items():
            module.multiplier = self.multiplier
            

class LoRAWrapper:
    def __init__(
        self,
        target_model,
        model_type=None,
        component_whitelist=None,
        multiplier=1.0,
        lora_dim=16,
        alpha=16,
        dropout=None,
        module_dropout=None,
        decompose=False,
        lr=None,
    ):
        self.target_model = target_model
        self.model_type = model_type
        self.component_whitelist = component_whitelist
        self.lr = lr

        self.is_active = False
        self.is_trainable = False
        self.is_quantized = False

        # Gather candidates for replacement
        self.target_map = scan_model(
            target_model, whitelist=component_whitelist
        )

        # Construct LoRA network
        self.net = LoRANetwork(
            self.target_map,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            module_dropout=module_dropout,
            decompose=decompose
        )

        # Get a list of bottom-level lora modules, excluding the originals
        self.residual_modules = nn.ModuleDict()
        for name, module in self.net.lora_modules.items():
            self.residual_modules[f"{name}/lora_down"] = module.lora_down
            self.residual_modules[f"{name}/lora_up"] = module.lora_up
            if module.dora_mag is not None:
                self.residual_modules[f"{name}/dora_mag"] = module.dora_mag

    def activate(self):
        assert not self.is_active, "LoRA is already active"
        self.net.activate(self.target_map)
        self.is_active = True
    
    def quantize(self):
        assert not self.is_trainable, "Quantization must be performed before training preparation"
        self.net.quantize_base()
        self.target_model.to('cuda')
        self.is_quantized = True

    def configure_optimizers(self):
        return optim.Adam([*self.residual_modules.parameters()], lr=self.lr)

    def prepare_for_training(self, training_wrapper):
        assert self.is_active, "LoRA must be activated before training preparation"

        # Freeze target model
        for param in self.target_model.parameters():
            param.requires_grad = False

        # Unfreeze lora modules
        for param in self.residual_modules.parameters():
            param.requires_grad = True

        # Move lora to training device
        self.net.to(device=training_wrapper.device)

        # Replace optimizer to use lora parameters TODO: implement more robust lr stuff
        training_wrapper.configure_optimizers = self.configure_optimizers

        # Trim ema model if present TODO: generalize beyond diffusion models
        if hasattr(training_wrapper, 'diffusion_ema') and training_wrapper.diffusion_ema is not None:
            trim_ema(training_wrapper.diffusion, training_wrapper.diffusion_ema)

        self.is_trainable = True

    def save_weights(self, path, dtype=torch.float16):
        torch.save(self.residual_modules.state_dict(), path)

    def load_weights(self, residual_weights, multiplier=1.0):
        resized = 0
        for key, weight in residual_weights.items():
            if key.endswith('lora_down.weight'):
                rank = weight.size(0)
            elif key.endswith('lora_up.weight'):
                rank = weight.size(1)
            module = self.net.lora_modules['/'.join(key.split('/')[:-1])]
            if module.lora_dim != rank:
                module.resize(rank)
                resized += 1
            self.residual_modules[key.split('.')[0]].weight.data = (weight * multiplier).detach()
        print(f'{resized} LoRA modules resized')

    def merge_weights(self, residual_weights, multiplier=1.0):
        for name, weight in residual_weights.items():
            param = self.residual_modules.state_dict()[name]
            param.copy_(param + residual_weights * multiplier)

    def extract_diff(self, tuned_model):
        lora_weights = calculate_svds(
            self.target_model,
            tuned_model,
            self.net.lora_modules.keys(),
            lora_dim=self.net.lora_dim,
        )
        for name, (down_weight, up_weight) in lora_weights.items():
            self.residual_modules[f"{name}/lora_down"].weight.data = down_weight.detach()
            self.residual_modules[f"{name}/lora_up"].weight.data = up_weight.detach()


class LoRAMerger:
    def __init__(
        self,
        target_model,
        model_type=None,
        component_whitelist=None
    ):
        self.target_model = target_model
        self.model_type = model_type
        self.component_whitelist = component_whitelist

        # Gather candidates for replacement
        self.target_map = scan_model(
            target_model, whitelist=component_whitelist
        )

        self.backup = {name: module.weight.data.detach().to('cpu') for name, module in self.target_map.items()}

        self.lora_paths = {}

    def add_lora(self, name, path):
        self.lora_dicts[name] = path
        print(f'LoRA registered: {name}')

    def merge(self, lora_multipliers=None):
        # If no multipliers specified, set all to 1.0
        if lora_multipliers is None:
            lora_multipliers = {name: 1.0 for name in self.lora_dicts.keys()}

        for lora_name, mul in lora_multipliers.items():
            state_dict = load_ckpt_state_dict(self.lora_paths[lora_name])
        

    
    def restore(self):
        for name, module in self.target_map:
            module.weight.data = self.backup[name].clone().detach().to(self.target_model.device)
        print('Base model weights restored from backup')
    


    

def create_lora_from_config(config, model):
    lora_config = config["lora"]

    model_type = config["model_type"]

    component_whitelist = lora_config.get("component_whitelist", None)
    assert component_whitelist is not None, "Must specify component whitelist in config"

    multiplier = lora_config.get("multiplier", None)
    assert multiplier is not None, "Must specify multiplier in config"

    rank = lora_config.get("rank", None)
    assert rank is not None, "Must specify rank in config"

    alpha = lora_config.get("alpha", None)
    assert alpha is not None, "Must specify alpha in config"

    dropout = lora_config.get("dropout", None)
    if dropout == 0: dropout = None

    module_dropout = lora_config.get("module_dropout", None)
    if module_dropout == 0: module_dropout = None

    lr = lora_config.get("lr", None)

    weight_decompose = lora_config.get("weight_decompose", False)

    lora = LoRAWrapper(
        model,
        model_type=model_type,
        component_whitelist=component_whitelist,
        multiplier=multiplier,
        lora_dim=rank,
        alpha=alpha,
        dropout=dropout,
        module_dropout=module_dropout,
        lr=lr,
        decompose=weight_decompose
    )

    return lora
