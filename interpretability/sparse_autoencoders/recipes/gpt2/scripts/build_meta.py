import os
from pathlib import Path

import pyarrow.parquet as pq
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from sae.analysis import export_text_features_parquet
from sae.collector import TokenActivationCollector
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
    """Minimal nn.Module wrapper over the Bloom SAE weights for the metadata builder."""

    def __init__(s):
        """Register the SAE weight/bias buffers."""
        super().__init__()
        for n, t in [("W_enc", W_enc), ("W_dec", W_dec), ("b_enc", b_enc), ("b_dec", b_dec)]:
            s.register_buffer(n, t)
        s.hidden_dim = W_enc.shape[1]

    def encode(s, x):
        """Residual activations -> ReLU SAE codes."""
        return F.relu((x - s.b_dec) @ s.W_enc + s.b_enc)


sae = SAEWrap()


def encode_fn(text):
    """Encode one text -> (str tokens, SAE codes) for the collector."""
    _, cache = model.run_with_cache(text, names_filter=HP)
    x = cache[HP][0].float().cpu()
    codes = sae.encode(x)
    labels = model.to_str_tokens(text)
    return labels, codes


ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
lines = [t[:600] for t in ds["text"] if len(t.strip()) > 40][:120]

col = TokenActivationCollector(encode_fn, n_features=sae.hidden_dim)
print("  collecting over corpus ...")
res = col.collect(lines)
print(f"  collected: {len(res.feature_stats)} features")
export_text_features_parquet(res, output_dir=str(OUT), n_examples=5)

for f in (str(OUT / "feature_metadata.parquet"), str(OUT / "feature_examples.parquet")):
    t = pq.read_table(f)
    print(f"  ✅ {f}: {t.num_rows} rows, cols={t.column_names}")
