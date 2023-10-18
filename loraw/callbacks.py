import pytorch_lightning as pl
from weakref import proxy

from .network import LoRAWWrapper


class LoRAWModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, loraw: LoRAWWrapper, **kwargs):
        super().__init__(**kwargs)
        self.loraw = loraw

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        self.loraw.save_weights(filepath)

        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))
