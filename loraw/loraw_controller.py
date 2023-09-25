from torch import optim
from ema_pytorch import EMA

from .loraw_network import LoRAWNetwork
from .loraw_module import LoRAWModule

class LoRAWController:
    def __init__(self, target_model, target_config) -> None:
        self.target_model = target_model
        self.target_config = target_config

        self.lr = 0
        self.lora_ema = None

    def create_diffuser_lora(
        self,
        lora_dim=16,
        alpha=1,
        dropout=None,
    ):
        self.lora = LoRAWNetwork(
            net=self.target_model,
            target_subnets=["downsamples", "upsamples"],
            target_modules=["Attention"],
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            multiplier=1.0,
            module_class=LoRAWModule,
            verbose=False,
        )
        self.lora.activate()

    def configure_optimizer_patched(self):
        return optim.Adam([*self.lora.parameters()], lr=self.lr)

    def on_before_zero_grad_patched(self, *args, **kwargs):
        self.lora_ema.update()

    def prepare_training(self, training_wrapper=None):
            # Freeze main diffusion model
            self.target_model.requires_grad_(False)
            self.lora.requires_grad_(True)

            # Replace optimizer to use lora parameters
            self.lr = training_wrapper.lr
            training_wrapper.configure_optimizers = self.configure_optimizer_patched

            # Replace ema update
            # training_wrapper.on_before_zero_grad = self.on_before_zero_grad_patched
