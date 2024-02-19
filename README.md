# LoRAW
Low Rank Adaptation for Waveforms based on:
- https://github.com/cloneofsimo/lora
- https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py

Highly experimental still

# Usage

## Construction
Create a loraw using the LoRAWWrapper class. For example using a conditional diffusion model for which we only want to target the down and up unet components:
```Python
from loraw.network import LoRAWWrapper

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
If using stable-audio-tools, you can create a LoRAW based on your model config:
```Python
from loraw.network import create_loraw_from_config

loraw = create_loraw_from_config(model_config, target_model)
```
For this to work, you need to add a loraw section to the model config. For example:
```JSON
{
    "model_type": "diffusion_cond"
    // ... args, model, training, etc. ...
	"loraw": {
        "target_blocks": ["Attention", "ConvBlock1d"],
        "component_whitelist": ["downsamples", "upsamples"],
        "multiplier": 1.0,
        "rank": 16,
        "alpha": 1.0,
        "dropout": 0,
        "module_dropout": 0
    }
}
```

## Activation
If you want to load weights into the target model, be sure to do so first as activation will alter the structure and confuse state_dict copying
```Python
loraw.activate()
```

## Loading and saving weights
`loraw.load_weights(path)` and `loraw.save_weights(path)` are for simple file IO. `loraw.merge_weights(path)` can be used to add more checkpoints without overwriting the current state.

## Training
With stable-audio-tools, after activation, you can simply call
```Python
loraw.prepare_for_training(training_wrapper)
```

For training to work manually, you need to:
- Set all original weights to `requires_grad = False`
- Set loraw weights set to `requires_grad = True` (easily accessed with `loraw.residual_modules.parameters()`)
- Update the optimizer to use the loraw parameters (the same parameters as the previous step)

# Example
See `scripts/train.py` for a modified version of stable audio tool's training script.

Modify your model config as shown above, and use `--use-loraw true` when running in CLI to enable