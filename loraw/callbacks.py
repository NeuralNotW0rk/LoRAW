import pytorch_lightning as pl
import os
from weakref import proxy

from .network import LoRAWrapper


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

class ReLoRAUpdateCallback(pl.Callback):
    def __init__(self, lora: LoRAWrapper, update_every=1000):
        super().__init__(**kwargs)
        self.lora = lora
        self.update_every = update_every

    @torch.no_grad()
    def on_train_batch_end(self, trainer):        
        if (trainer.global_step - 1) % self.update_every != 0:
            return

        self.lora.net.update_base()





