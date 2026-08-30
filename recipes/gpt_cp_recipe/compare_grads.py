#!/usr/bin/env python3

import torch
import torch.distributed.tensor
def combine(layer_name, cp_0, cp_1):
     cp_0_l = cp_0[layer_name]
     cp_1_l = cp_1[layer_name]
     if cp_0_l is not None and cp_1_l is not None:
         return torch.cat([cp_0_l.to_local().cpu(), cp_1_l.to_local().cpu()])
     if cp_0_l is not None:
         return cp_0_l.to_local().cpu()
     return cp_1_l.to_local().cpu()


cp_0 = torch.load("cp_2_grads_rnk_0_0.pt", weights_only=False)
cp_1 = torch.load("cp_2_grads_rnk_1_0.pt", weights_only=False)
combined_cp = {key: combine(key, cp_0, cp_1) for key in cp_0.keys()}

base = {k: v.to_local().cpu() for k, v in torch.load("cp_1_grads_rnk_0_0.pt", weights_only=False).items()}

assert set(combined_cp.keys()) == set(base.keys())

for k in combined_cp.keys():
    print(f"Testing {k}")
    torch.testing.assert_close(combined_cp[k], base[k])