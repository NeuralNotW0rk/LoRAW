import torch
from tqdm import tqdm

CLAMP_QUANTILE = 0.99
MIN_DIFF = 1e-6


# Get a dict of lora weights calculated using singular value decomposition on the difference between an untuned and tuned model
def calculate_svds(model_original, model_tuned, lora_names, lora_dim):
    map_o = {
        name.replace(".", "/"): module for name, module in model_original.named_modules()
    }
    map_t = {
        name.replace(".", "/"): module for name, module in model_tuned.named_modules()
    }

    # Calculate SVD
    lora_weights = {}
    for name in tqdm(lora_names):
        with torch.no_grad():
            in_dim = map_t[name].in_features
            out_dim = map_t[name].out_features
            rank = min(in_dim, out_dim, lora_dim)

            residual = map_t[name].weight.data - map_o[name].weight.data
            residual.float()

            U, S, Vh = torch.linalg.svd(residual)

            U = U[:, :rank]
            S = S[:rank]
            U = U @ torch.diag(S)
            Vh = Vh[:rank, :]

            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi_val = torch.quantile(dist, CLAMP_QUANTILE)
            low_val = -hi_val

            U = U.clamp(low_val, hi_val)
            Vh = Vh.clamp(low_val, hi_val)

            # Record lora_down and lora_up weights
            lora_weights[name] = (Vh[:, :in_dim], U[:out_dim])

    return lora_weights


# Remove redundancy by sharing weights between main and ema models which will not be updated during lora training
def trim_ema(model, ema_model):
    for (name, module), module_ema in zip(
        model.named_modules(), ema_model.modules()
    ):
        if (
            hasattr(module, "weight")
            and not name.endswith("lora_down")
            and not name.endswith("lora_up")
        ):
            module_ema.weight = module.weight
