# LoRAW
Low Rank Adaptation for Waveforms based on:
- https://github.com/cloneofsimo/lora
- https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py

Highly experimental still

# Usage

## Construction
Create a loraw using the LoRAWWrapper class. For example using a conditional diffusion model for which we only want to target the down and up unet components:
```Python
from LoRAW.network import LoRAWWrapper

loraw = LoRAWWrapper(
    target_model,
    target_blocks=["Attention"],
    component_whitelist=["downsamples", "upsamples"],
    lora_dim=16,
    alpha=1.0,
    dropout=None,
    multiplier=1.0
)
```

## Activation
If you want to load weights into the target model, be sure to do so first as activation will alter the structure and confuse state_dict copying
```Python
loraw.activate()
```

## Loading and saving weights
`loraw.load_weights(path)` and `loraw.save_weights(path)` are for simple file IO. `loraw.merge_weights(path)` can be used to add more checkpoints without overwriting the current state.

## Training
With harmonai-tools, after activation, you can simply call
```Python
loraw.prepare_for_training(training_wrapper)
```

For training to work manually, you need to:
- Set all original weights to `requires_grad = False`
- Set loraw weights set to `requires_grad = True` (easily accessed with `loraw.residual_modules.parameters()`)
- Update the optimizer to use the loraw parameters (the same parameters as the previous step)