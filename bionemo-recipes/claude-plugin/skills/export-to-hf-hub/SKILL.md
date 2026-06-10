---
name: export-to-hf-hub
description: >
  Export a TransformerEngine model to HuggingFace Hub format.
  Triggers when user asks to export, publish, upload to HuggingFace,
  or create a model card.
allowed-tools: Read, Grep, Glob, Write, Edit, Bash, Agent
argument-hint: '[model-path] [hub-id]'
---

# Export TE Model to HuggingFace Hub

You are creating an export pipeline that converts a HuggingFace model to TE format and packages it for distribution on HuggingFace Hub.

## Reference Files

- `reference/export_esm2.py` — Complete export script example

## Steps

### 1. Load and Convert

```python
model_hf = AutoModelForMaskedLM.from_pretrained(model_id)
model_te = convert_hf_to_te(model_hf)
model_te.save_pretrained(export_path)
```

### 2. Save Tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.save_pretrained(export_path)
```

### 3. Patch config.json with AUTO_MAP

```python
import json

with open(export_path / "config.json", "r") as f:
    config = json.load(f)
config["auto_map"] = {
    "AutoConfig": "model_file.NVConfig",
    "AutoModel": "model_file.NVModel",
    "AutoModelForMaskedLM": "model_file.NVModelForMaskedLM",
}
with open(export_path / "config.json", "w") as f:
    json.dump(config, f, indent=2, sort_keys=True)
```

### 4. Copy Model Code as Remote Code

```python
import shutil

shutil.copy("modeling_te.py", export_path / "model_file.py")
```

### 5. Smoke Test

```python
model = AutoModelForMaskedLM.from_pretrained(export_path, trust_remote_code=True)
```

### 6. Upload to Hub

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(folder_path=export_path, repo_id="org/model-name")
```
