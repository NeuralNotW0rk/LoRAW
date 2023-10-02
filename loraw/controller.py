from torch import nn
from torch import optim
from ema_pytorch import EMA

from .network import LoRAWNetwork


class LoRAWController:
    def __init__(self) -> None:
        self.loraws = []
        self.lr = None

    def create_loraw_wrapper(
        self,
        target_model,
        target_blocks=["Attention"],
        component_whitelist=["downsamples", "upsamples"],
        lora_dim=16,
        alpha=1.0,
        dropout=None,
        multiplier=1.0,
    ) -> LoRAWWrapper:

        loraw = LoRAWWrapper(target_model, target_blocks=target_blocks, )
        self.loraws.append(loraw)
        return loraw

    def activate_all(self, train=False):
        for loraw in self.loraws:
            loraw.net.activate(loraw.target_map)
            loraw.train = train

    def configure_optimizers(self):
        optims = []
        for loraw in self.loraws:
            if loraw.trainable:
                optims.append(loraw.configure_optimizers())
        return optims

    def prepare_all_for_training(self, training_wrapper, lr=None):

        # Freeze target model
        self.target_model.requires_grad_(False)

        for loraw in self.loraws.values():
            # Move lora to training device
            loraw.net.to(device=training_wrapper.device)

            # Unfreeze lora modules
            if loraw.trainable:
                loraw.net.requires_grad_(True)

        # Replace optimizer to use lora parameters
        if lr is None:
            self.lr = training_wrapper.lr
        else:
            self.lr = lr
        training_wrapper.configure_optimizers = self.configure_optimizers()
