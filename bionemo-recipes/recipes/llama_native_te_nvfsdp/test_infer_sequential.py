#!/usr/bin/env python3

import pytest
import torch
import pickle
import os
from pathlib import Path
from transformers import AutoConfig, LlamaForCausalLM
from model import NVLlamaForCausalLM

@pytest.mark.parametrize("model_name", ["meta-llama/Llama-2-7b-hf"])
def test_hf_and_te_llama_equivalence_sequential(model_name: str, tol=0.25):
    """Sequential test that loads one model at a time and saves outputs to pickle files"""
    
    print(f"Testing {model_name} (Sequential with Pickle Comparison)")
    print("=" * 60)
    
    device = torch.device("cuda:0")
    
    # Clear cache first
    torch.cuda.empty_cache()
    
    config = AutoConfig.from_pretrained(model_name, torch_dtype=torch.float16)
    
    # Optional: Reduce layers for memory safety
    # config.num_hidden_layers = 16  # Uncomment to reduce layers
    
    # Create dummy input (fixed seed for reproducibility)
    torch.manual_seed(42)
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(1, 32), device=device)
    input_ids_cpu = input_ids.cpu()  # Save CPU version for later
    
    print(f"Input created: {input_ids.shape} (seed=42 for reproducibility)")
    
    # Create output directory
    output_dir = Path("model_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # ===== STEP 1: Load and test NV model =====
    print("\nStep 1: Loading NVLlama model...")
    nv_model = NVLlamaForCausalLM.from_pretrained(
        model_name, config=config, torch_dtype=torch.float16
    ).to(device)
    nv_model.eval()
    print(f"NVLlama loaded on GPU")
    
    # Get NV model output
    with torch.no_grad():
        nv_logits = nv_model(input_ids).logits
    print(f"NVLlama forward pass: {nv_logits.shape}")
    
    # Save NV results to pickle
    nv_results = {
        "logits": nv_logits.cpu(),
        "model_name": model_name,
        "model_type": "NVLlama",
        "input_ids": input_ids_cpu,
        "config_layers": config.num_hidden_layers,
        "config_hidden": config.hidden_size
    }
    
    nv_pickle_path = output_dir / "nv_model_output.pkl"
    with open(nv_pickle_path, "wb") as f:
        pickle.dump(nv_results, f)
    print(f"NVLlama results saved to {nv_pickle_path}")
    
    # Clean up NV model
    del nv_model, nv_logits
    torch.cuda.empty_cache()
    print("NVLlama unloaded, GPU memory cleared")
    
    # ===== STEP 2: Load and test HF model =====
    print("\nStep 2: Loading HuggingFace model...")
    hf_model = LlamaForCausalLM.from_pretrained(
        model_name, config=config, torch_dtype=torch.float16
    ).to(device)
    hf_model.eval()
    print(f"HuggingFace loaded on GPU")
    
    # Get HF model output
    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits
    print(f"HuggingFace forward pass: {hf_logits.shape}")
    
    # Save HF results to pickle
    hf_results = {
        "logits": hf_logits.cpu(),
        "model_name": model_name,
        "model_type": "HuggingFace",
        "input_ids": input_ids_cpu,
        "config_layers": config.num_hidden_layers,
        "config_hidden": config.hidden_size
    }
    
    hf_pickle_path = output_dir / "hf_model_output.pkl"
    with open(hf_pickle_path, "wb") as f:
        pickle.dump(hf_results, f)
    print(f"HuggingFace results saved to {hf_pickle_path}")
    
    # Clean up HF model
    del hf_model, hf_logits
    torch.cuda.empty_cache()
    print("HuggingFace unloaded, GPU memory cleared")
    
    # ===== STEP 3: Load and compare results =====
    print("\nStep 3: Loading pickled results for comparison...")
    
    with open(nv_pickle_path, "rb") as f:
        nv_data = pickle.load(f)
    
    with open(hf_pickle_path, "rb") as f:
        hf_data = pickle.load(f)
    
    nv_logits_loaded = nv_data["logits"]
    hf_logits_loaded = hf_data["logits"]
    
    print(f"Results loaded from disk")
    print(f"   NV logits: {nv_logits_loaded.shape}")
    print(f"   HF logits: {hf_logits_loaded.shape}")
    
    # ===== STEP 4: Compare results =====
    print("\nStep 4: Comparing model outputs...")
    
    # Check shapes
    assert hf_logits_loaded.shape == nv_logits_loaded.shape, \
        f"Shape mismatch: HF {hf_logits_loaded.shape} vs NV {nv_logits_loaded.shape}"
    print(f"Shapes match: {hf_logits_loaded.shape}")
    
    # Check input consistency
    assert torch.equal(nv_data["input_ids"], hf_data["input_ids"]), \
        "Input mismatch between saved results"
    print(f"Input consistency verified")
    
    # Compare outputs
    diff = torch.abs(hf_logits_loaded - nv_logits_loaded)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    print(f"Difference statistics:")
    print(f"   Max absolute diff: {max_diff:.6f}")
    print(f"   Mean absolute diff: {mean_diff:.6f}")
    print(f"   Tolerance: {tol}")
    
    # Final comparison
    outputs_match = torch.allclose(hf_logits_loaded, nv_logits_loaded, atol=tol, rtol=tol)
    
    if outputs_match:
        print(f"Models match within tolerance {tol}")
        print("\nSequential test PASSED!")
    else:
        print(f"Models differ more than tolerance {tol}")
        print(f"   Consider increasing tolerance or investigating differences")
        raise AssertionError(f"Models differ more than tol={tol}")
    
    # Clean up pickle files
    print(f"\nCleaning up pickle files...")
    nv_pickle_path.unlink()
    hf_pickle_path.unlink()
    print(f"Temporary files cleaned up")

if __name__ == "__main__":
    test_hf_and_te_llama_equivalence_sequential("meta-llama/Llama-2-7b-hf")
