import torch

# Key prefixes and delimiters
prefixes = {
    'default': 'model/model/',
    'comfyui': 'lora_unet_'
}
delimiters = {
    'default': '/',
    'comfyui': '_'
}
dora_name = {
    'default': 'dora_mag',
    'comfyui': 'dora_scale'
}

def main(args):
    path_in = args.path_in
    path_out = path_in if args.path_out is None else args.path_out
    target = 'default' if args.target is None else args.target
    type = 'ckpt' if args.format is None else args.format

    lora = torch.load(path_in, map_location='cpu')
    # Naive target detection: default keys use '/' instead of '_'
    target_original = 'default' if '/' in list(lora.keys())[0] else 'comfyui'
    print(f'Detected target application: {target_original}')
    
    new_dict = None

    if target_original != target:
        print(f'Converting for use with {target}')
        new_dict = {}
        for name, tensor in lora.items():
            new_name = name.replace(prefixes[target_original], prefixes[target])
            new_name = new_name.replace(delimiters[target_original], prefixes[target])
            new_name = new_name.replace(dora_name[target_original], dora_name[target])
            new_dict[new_name] = tensor

    if new_dict is not None:
        torch.save(new_dict, path_out)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert lora checkpoint format')
    parser.add_argument('--path-in', type=str, help='Path to lora checkpoint', required=True)
    parser.add_argument('--path-out', type=str, help='Path to save converted lora checkpoint', required=False)
    parser.add_argument('--target', type=str, help='Application to save lora checkpoint for [default, comfyui]', required=False)
    parser.add_argument('--format', type=str, help='Format to save lora checkpoint [ckpt, safetensors]', required=False)
    args = parser.parse_args()
    main(args)

    