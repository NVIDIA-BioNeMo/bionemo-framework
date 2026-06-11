# Training Models

Actively supported model training workflows live in `models/` and `recipes/`.

| Workflow                      | Location                                 |
| ----------------------------- | ---------------------------------------- |
| ESM-2 native PyTorch + TE     | `recipes/esm2_native_te`                 |
| ESM-2 with Accelerate + TE    | `recipes/esm2_accelerate_te`             |
| Geneformer + TE               | `recipes/geneformer_native_te_mfsdp_fp8` |
| Evo2 with Megatron Bridge     | `recipes/evo2_megatron`                  |
| EDEN with Megatron Bridge     | `recipes/eden_megatron`                  |
| TE-optimized AMPLIFY model    | `models/amplify`                         |
| TE-optimized ESM-2 model      | `models/esm2`                            |
| TE-optimized Geneformer model | `models/geneformer`                      |

Use each directory's README as the setup source of truth.
