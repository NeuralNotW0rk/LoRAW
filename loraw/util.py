import torch
import ema_pytorch

CLAMP_QUANTILE = 0.99
MIN_DIFF = 1e-6


# Get a dict of lora weights calculated using singular value decomposition on the difference between am untuned and tuned model
def calculate_svds(model_original, model_tuned, lora_names, rank):
    map_o = {
        name.replace(".", "/"): module
        for name, module in model_original.named_modules()
    }
    map_t = {
        name.replace(".", "/"): module for name, module in model_tuned.named_modules()
    }

    # Get diffs
    diffs = {}
    for name in lora_names:
        diff = map_t[name].weight - map_o[name].weight
        diff = diff.float()
        diffs[name] = diff

    # Calculate SVD
    lora_weights = {}
    with torch.no_grad():
        for lora_name, mat in list(diffs.items()):
            conv1d = len(mat.size()) == 3
            kernel_size = None if not conv1d else mat.size()[2]
            out_dim, in_dim = mat.size()[0:2]

            mat_padded = torch.zeros(max(out_dim, rank), max(in_dim, rank), kernel_size)
            mat_padded[:out_dim, :in_dim] = mat

            if conv1d:
                if kernel_size != 1:
                    mat_padded = mat_padded.flatten(start_dim=1)
                else:
                    mat_padded = mat_padded.squeeze()

            U, S, Vh = torch.linalg.svd(mat_padded)

            U = U[:, :rank]
            S = S[:rank]
            U = U @ torch.diag_embed(S)

            Vh = Vh[:rank, :]

            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi_val = torch.quantile(dist, CLAMP_QUANTILE)
            low_val = -hi_val

            U = U.clamp(low_val, hi_val)
            Vh = Vh.clamp(low_val, hi_val)

            if conv1d:
                U = U.reshape(max(out_dim, rank), rank, 1)
                Vh = Vh.reshape(rank, max(in_dim, rank), kernel_size)

            U = U.to("cpu").contiguous()
            Vh = Vh.to("cpu").contiguous()

            # Record lora_down and lora_up weights
            lora_weights[lora_name] = (Vh[:, :in_dim], U[:out_dim])

    return lora_weights


# Remove redundancy by sharing weights between online and ema models which will not be updated during lora training
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
