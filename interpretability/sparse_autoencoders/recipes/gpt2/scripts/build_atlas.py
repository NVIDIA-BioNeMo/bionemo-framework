import os
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from sae.analysis import compute_feature_stats, compute_feature_umap, save_feature_atlas
from safetensors.torch import load_file
from transformer_lens import HookedTransformer


# Where to write the parquets — defaults to the dashboard's bundled data dir.
OUT = Path(os.environ.get("GPT2_SAE_DATA", Path(__file__).resolve().parents[1] / "dashboard_data"))
OUT.mkdir(parents=True, exist_ok=True)

REPO = "jbloom/GPT2-Small-SAEs-Reformatted"
HP = "blocks.7.hook_resid_pre"
model = HookedTransformer.from_pretrained("gpt2")
model.eval()
w = load_file(hf_hub_download(REPO, f"{HP}/sae_weights.safetensors"))
W_enc, W_dec, b_enc, b_dec = [w[k].float() for k in ("W_enc", "W_dec", "b_enc", "b_dec")]


class SAEWrap(nn.Module):
    """Minimal nn.Module wrapper over the Bloom SAE weights for the atlas builder."""

    def __init__(s):
        """Register the SAE weight/bias buffers and a decoder Linear."""
        super().__init__()
        s.register_buffer("W_enc", W_enc)
        s.register_buffer("W_dec", W_dec)
        s.register_buffer("b_enc", b_enc)
        s.register_buffer("b_dec", b_dec)
        s.hidden_dim = W_enc.shape[1]
        s.input_dim = W_enc.shape[0]
        s.decoder = nn.Linear(s.hidden_dim, s.input_dim, bias=False)
        s.decoder.weight.data = W_dec.t().contiguous()  # (input_dim, hidden_dim)

    def encode(s, x):
        """Residual activations -> ReLU SAE codes."""
        return F.relu((x - s.b_dec) @ s.W_enc + s.b_enc)

    def decode(s, c):
        """SAE codes -> reconstructed residual activations."""
        return c @ s.W_dec + s.b_dec


sae = SAEWrap()

# activations: cache to disk so re-runs are fast
if os.path.exists(str(OUT / "acts.pt")):
    X = torch.load(str(OUT / "acts.pt"))
else:
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    lines = [t for t in ds["text"] if len(t.strip()) > 40][:120]
    acts = []
    with torch.no_grad():
        for t in lines:
            _, cache = model.run_with_cache(t[:600], names_filter=HP)
            acts.append(cache[HP][0].float().cpu())
    X = torch.cat(acts, 0)[:6000]
    torch.save(X, str(OUT / "acts.pt"))
print(f"  activations {tuple(X.shape)}")

stats, top_examples = compute_feature_stats(sae, X, device="cpu", top_k=10, batch_size=4096, show_progress=False)
print(f"  stats: {len(stats)} features")
geom = compute_feature_umap(sae, compute_clusters=False)
print(f"  umap: {geom.umap_x.shape}")
save_feature_atlas(stats, geom, str(OUT / "features_atlas.parquet"), top_examples=top_examples)

t = pq.read_table(str(OUT / "features_atlas.parquet"))
print(f"  ✅ features_atlas.parquet: {t.num_rows} rows, cols={t.column_names}")
df = t.to_pandas()
alive = (df["activation_freq"] > 0).sum()
print(f"  alive features: {alive}/{len(df)}   x/y range: x[{df.x.min():.1f},{df.x.max():.1f}]")
print(df[["feature_id", "activation_freq", "max_activation", "x", "y"]].head(3).to_string(index=False))
