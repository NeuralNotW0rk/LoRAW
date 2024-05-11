# LoRAW
Low Rank Adaptation for Waveforms

Designed to be used with Stable Audio Tools

Highly experimental still

# Usage

## Construction
Create a loraw using the LoRAWrapper class. For example using a conditional diffusion model for which we only want to target the transformer component:
```Python
from loraw.network import LoRAWrapper

lora = LoRAWrapper(
    target_model,
    target_blocks=["Attention"],
    component_whitelist=["transformer"],
    lora_dim=16,
    alpha=16,
    dropout=None,
    multiplier=1.0
)
```
If using stable-audio-tools, you can create a LoRA based on your model config:
```Python
from loraw.network import create_lora_from_config

lora = create_lora_from_config(model_config, target_model)
```
For this to work, you need to add a lora section to the model config. For example:
```JSON
{
    "model_type": "diffusion_cond"
    // ... args, model, training, etc. ...
    "lora": {
        "target_blocks": ["Attention"],
        "component_whitelist": ["transformer"],
        "multiplier": 1.0,
        "rank": 16,
        "alpha": 16,
        "dropout": 0,
        "module_dropout": 0,
        "lr": 1e-4
    }
}
```

## Activation
If you want to load weights into the target model, be sure to do so first as activation will alter the structure and confuse state_dict copying
```Python
lora.activate()
```

## Loading and saving weights
`lora.load_weights(path)` and `lora.save_weights(path)` are for simple file IO. `lora.merge_weights(path)` can be used to add more checkpoints without overwriting the current state.

## Training
With stable-audio-tools, after activation, you can simply call
```Python
lora.prepare_for_training(training_wrapper)
```

For training to work manually, you need to:
- Set all original weights to `requires_grad = False`
- Set lora weights set to `requires_grad = True` (easily accessed with `lora.residual_modules.parameters()`)
- Update the optimizer to use the lora parameters (the same parameters as the previous step)

# Example
See `examples/train.py` for a modified version of stable audio tool's training script.

Modify your model config as shown above, and use `--use-lora true` when running in CLI to enable

# References
- https://github.com/cloneofsimo/lora
- https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py