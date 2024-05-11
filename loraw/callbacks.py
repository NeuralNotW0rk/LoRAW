import os
import torch
import pytorch_lightning as pl
from weakref import proxy

from .network import LoRAWrapper

# Save lora weights only
class LoRAModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, lora: LoRAWrapper, **kwargs):
        super().__init__(**kwargs)
        self.lora = lora

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.lora.save_weights(filepath)

        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))


# Update and save base model
class ReLoRAModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, lora: LoRAWrapper, checkpoint_every_n_updates=1, **kwargs):
        super().__init__(**kwargs)
        self.lora = lora
        self.checkpoint_every_n_updates = checkpoint_every_n_updates
        self.updates = 0

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        self.lora.net.update_base()

        if self.updates % self.checkpoint_every_n_updates == 0:
            super()._save_checkpoint(trainer, filepath)

        self.updates += 1


# Update base model with lora weights (no checkpoint saving)
class ReLoRAUpdateCallback(pl.Callback):

    def __init__(self, lora: LoRAWrapper, update_every=1000, **kwargs):
        super().__init__(**kwargs)
        self.lora = lora
        self.update_every = update_every

    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):        
        if (trainer.global_step - 1) % self.update_every != 0:
            return

        self.lora.net.update_base()





