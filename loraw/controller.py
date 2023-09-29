from torch import nn
from torch import optim
from ema_pytorch import EMA

from .network import LoRAWNetwork

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
                    module_map[id] = {"module": child_module, "parent": parent_module}
    print(f"Found {len(module_map)} candidates for LoRAW replacement")
    return module_map


class LoRAWController:
    def __init__(self) -> None:
        self.lr = 0
        self.lora_ema = None

    def create_loraw(
        self,
        target_model,
        target_blocks=["Attention"],
        component_whitelist=["downsamples", "upsamples"],
        lora_dim=16,
        alpha=1.0,
        dropout=None,
        multiplier=1.0,
    ):
        self.target_map = scan_model(
            target_model, target_blocks, whitelist=component_whitelist
        )
        self.lora = LoRAWNetwork(
            self.target_map,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            multiplier=multiplier,
        )

    def activate(self):
        self.lora.activate(self.target_map)

    def configure_optimizer_patched(self):
        return optim.Adam([*self.lora.parameters()], lr=self.lr)

    def on_before_zero_grad_patched(self, *args, **kwargs):
        self.lora_ema.update()

    def prepare_training(self, training_wrapper):
        # Move lora to training device
        self.lora.to(device=training_wrapper.device)

        # Freeze main diffusion model
        self.target_model.requires_grad_(False)
        self.lora.requires_grad_(True)

        # Replace optimizer to use lora parameters
        self.lr = training_wrapper.lr
        training_wrapper.configure_optimizers = self.configure_optimizer_patched

        # Replace ema update
        # training_wrapper.on_before_zero_grad = self.on_before_zero_grad_patched
