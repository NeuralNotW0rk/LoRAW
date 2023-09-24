
import torch
from torch import nn
from typing import List

from .loraw_module import LoRAWModule

class LoRAWNetwork(nn.Module):
    def __init__(
        self,
        net,
        target_subnets=None,
        target_modules=[
            'SelfAttention1d'
        ],
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        module_class=LoRAWModule,
        verbose=False,
    ):
        super().__init__()

        self.lora_map = {}
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.dropout = dropout

        def create_modules(
            root_name, root_module: nn.Module, target_replace_modules
        ) -> nn.ModuleList:
            loras = nn.ModuleList()
            skipped = nn.ModuleList()
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv1d = child_module.__class__.__name__ == "Conv1d"

                        if is_linear or is_conv1d:
                            lora_name = "lora.{root_name}.{name}.{child_name}"
                            lora_name = lora_name.replace(".", "_")

                            lora = module_class(
                                lora_name,
                                child_module,
                                multiplier=self.multiplier,
                                lora_dim=self.lora_dim,
                                alpha=self.alpha,
                                dropout=self.dropout
                            )
                            loras.append(lora)
            return loras, skipped

        for subnet_name in target_subnets:
            if hasattr(net.model, subnet_name):
                subnet = getattr(net.model, subnet_name)
                self.lora_map[subnet_name], _ = create_modules(subnet_name, subnet, target_modules)
                print(f"Created LoRAW for {subnet_name}: {len(self.lora_map[subnet_name])} modules.")

                '''
                if verbose and len(skipped) > 0:
                    print(
                        f"because block_lr_weight is 0 or dim (rank) is 0, {len(skipped)} LoRA modules are skipped / block_lr_weightまたはdim (rank)が0の為、次の{len(skipped)}個のLoRAモジュールはスキップされます:"
                    )
                    for name in skipped:
                        print(f"\t{name}")

                self.up_lr_weight: List[float] = None
                self.down_lr_weight: List[float] = None
                self.mid_lr_weight: float = None
                self.block_lr = False

                # assertion
                names = set()
                for lora in self.unet_loras:
                    assert (
                        lora.lora_name not in names
                    ), f"duplicated lora name: {lora.lora_name}"
                    names.add(lora.lora_name)
                '''
            else:
                print(f'Skipping {subnet_name}: not present in this network')
        


    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.unet_loras:
            lora.multiplier = self.multiplier

    def activate(self):
        for subnet_name, subnet in self.lora_map.items():
            for lora in subnet:
                lora.activate()
                self.add_module(lora.lora_name, lora)
            print(f'Injected {len(subnet)} LoRAW modules into {subnet_name}')

    def is_mergeable(self):
        return True

    def save_weights(self, file, dtype=torch.float16):

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        torch.save(state_dict, file)

    def load_weights(self, file):
        weights_sd = torch.load(file, map_location="cpu")

        info = self.load_state_dict(weights_sd, False)
        return info
