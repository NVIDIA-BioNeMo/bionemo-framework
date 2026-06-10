import time
import sys
import IPython

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    AutoConfig,
)
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils.dataclasses import FP8RecipeKwargs

class HyperParameters:
    def __init__(self):
        self.mixed_precision = "bf16"

        # Set to Meta Llama 2 by default.
        self.model_name = "meta-llama/Llama-2-7b-hf"

        self.dataset_name = "timdettmers/openassistant-guanaco"
        self.dataset_text_field = "text"
        self.learning_rate = 1.41e-5
        self.batch_size = 8
        self.max_seq_length = 256
        self.gradient_accumulation_steps = 1
        self.num_warmup_steps = 5
        self.num_training_steps = 10

        # This is either provided by the user or it will be set when the
        # model weights are downloaded.
        self.weights_cache_dir = ""


hyperparams = HyperParameters()


def get_dataloaders(accelerator: Accelerator, hyperparams):
    dataset = load_dataset(hyperparams.dataset_name, split="train")
    tokenizer = AutoTokenizer.from_pretrained(hyperparams.model_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            padding=False,
            max_length=hyperparams.max_seq_length,
            return_overflowing_tokens=False,
            return_length=False,
        )
        return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

    with accelerator.main_process_first():
        dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    # Simply pad to the multiple of 16 for both FP8 and BF16 precision
    pad_to_multiple_of = 16
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    dataloader_params = {
        "batch_size": hyperparams.batch_size,
        "collate_fn": data_collator,
        "drop_last": True,
    }
    train_dataloader = DataLoader(dataset, **dataloader_params)
    return train_dataloader


def ensure_model_is_downloaded(hyperparams):
    assert hyperparams.model_name in [
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Llama-2-7b-hf",
    ], "Only Meta Llama 2 7B and Meta Llama 3 8B models are supported!"

    # Login using Huggingface Hub API
    from huggingface_hub import login

    try:
        login(hyperparams.hf_access_token)
    except Exception as e:
        if "Invalid token passed!" in str(e):
            print(
                "Please pass a valid HF Access Token! More info at"
                " https://huggingface.co/docs/hub/en/security-tokens."
            )
        else:
            print(f"Exception is {e}")

    # Download the model if it doesn't exist
    from huggingface_hub import snapshot_download

    supplied_cache_dir = (
        hyperparams.weights_cache_dir if hyperparams.weights_cache_dir != "" else None
    )
    hyperparams.weights_cache_dir = snapshot_download(
        repo_id=hyperparams.model_name, cache_dir=supplied_cache_dir
    )

    print(f"Model cache directory : {hyperparams.weights_cache_dir}")


def init_baseline_model(hyperparams):
    # Download and cache the weights
    ensure_model_is_downloaded(hyperparams)

    # Init the model
    config = AutoConfig.from_pretrained(hyperparams.weights_cache_dir)
    # make sure to use flash_attention to do iso comparison with TELlamaModel
    config._attn_implementation = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(
        hyperparams.weights_cache_dir,
        config=config,
        torch_dtype=torch.bfloat16,
    )

    model = model.cuda()
    # Needed for the cases when using TELlamaForCausalLM. So adding here for 1:1 comparison
    model.config.use_cache = False

    return model


def init_te_llama_model(hyperparams):
    # Download and cache the weights
    ensure_model_is_downloaded(hyperparams)

    # Init the model
    config = AutoConfig.from_pretrained(hyperparams.weights_cache_dir)
    config._attn_implementation = "flash_attention_2"
    model = TELlamaForCausalLM.from_pretrained_local(
        hyperparams.weights_cache_dir,
        config=config,
        torch_dtype=torch.bfloat16,
    )
    model = model.cuda()
    # Needed for the cases when using TELlamaForCausalLM
    model.config.use_cache = False

    return model


def wrap_with_accelerator(model, hyperparams):
    # Create FP8 kwarg handler if required
    fp8_kwarg_handler = (
        [FP8RecipeKwargs(backend="te")] if hyperparams.mixed_precision == "fp8" else None
    )

    # Init HF accelerator that's used for training
    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=hyperparams.gradient_accumulation_steps,
        mixed_precision=hyperparams.mixed_precision,
        kwargs_handlers=fp8_kwarg_handler,
    )
    # accelerator.print(f'State: {accelerator.state}')
    train_dataloader = get_dataloaders(accelerator, hyperparams)

    # Wrap model, optimizer/scheduler, dataloaders in accelerate
    optimizer = AdamW(params=model.parameters(), lr=hyperparams.learning_rate, fused=True)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=hyperparams.num_training_steps,
    )
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    return accelerator, model, optimizer, train_dataloader, lr_scheduler


def finetune_model(model, hyperparams, accelerator, train_dataloader, optimizer, lr_scheduler):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    train_dataloader = enumerate(train_dataloader)

    # Warmup iters
    for _ in range(hyperparams.num_warmup_steps):
        step, batch = next(train_dataloader)
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    # Get the timers ready
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()

    start.record()
    # Training iters
    for _ in range(hyperparams.num_training_steps):
        step, batch = next(train_dataloader)
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
    torch.cuda.synchronize()
    end.record()
    accelerator.end_training()

    print(
        f"{hyperparams.num_training_steps} finetuning steps complete!\nAverage time taken per step:"
        f" {(start.elapsed_time(end)/hyperparams.num_training_steps):.0f} milliseconds"
    )


#######################
# te_llama.py
#######################
import os
import re
import gc
from contextlib import contextmanager

import torch

import transformer_engine as te
from transformer_engine.pytorch.attention.rope import RotaryPositionEmbedding

import transformers
from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaForCausalLM,
    LlamaRMSNorm,
    LlamaConfig,
)
from transformers.modeling_utils import _add_variant, load_state_dict
from transformers.utils import WEIGHTS_INDEX_NAME
from transformers.utils.hub import get_checkpoint_shard_files


@contextmanager
def replace_decoder(te_decoder_cls):
    """
    Replace `LlamaDecoderLayer` with custom `TELlamaDecoderLayer`.
    """
    original_llama_decoder_cls = transformers.models.llama.modeling_llama.LlamaDecoderLayer
    transformers.models.llama.modeling_llama.LlamaDecoderLayer = te_decoder_cls
    try:
        yield
    finally:
        transformers.models.llama.modeling_llama.LlamaDecoderLayer = original_llama_decoder_cls


class TELlamaDecoderLayer(te.pytorch.TransformerLayer):
    """
    Wrapper class over TE's `TransformerLayer`. This makes the wrapper very
    similar to HF's `LlamaDecoderLayer` and easier to replace it in the code.

    Args:
        config: LlamaConfig
        args: positional args (for compatibility with `LlamaDecoderLayer`)
        kwargs: keyword args (for compatibility with `LlamaDecoderLayer`)
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            bias=False,
            layernorm_epsilon=config.rms_norm_eps,
            hidden_dropout=0,
            attention_dropout=0,
            fuse_qkv_params=False,
            normalization="RMSNorm",
            activation="swiglu",
            attn_input_format="bshd",
            num_gqa_groups=config.num_key_value_heads,
        )
        te_rope = RotaryPositionEmbedding(config.hidden_size // config.num_attention_heads)
        self.te_rope_emb = te_rope(max_seq_len=config.max_position_embeddings).cuda()

    def forward(self, hidden_states, *args, attention_mask, **kwargs):
        """
        Custom forward to make sure we only pass relevant arguments to the
        forward pass of the `TransformerLayer`. Also, make sure the output
        format matches the output of the HF's `LlamaDecoderLayer`.
        """
        return super().forward(
                hidden_states, attention_mask=attention_mask, rotary_pos_emb=self.te_rope_emb
        )


class TELlamaForCausalLM:
    """
    Causal LM created with `LlamaModel`. The underlying `LlamaDecoderLayer`
    class is monkey-patched with `TELlamaDecoderLayer` class before
    initializing the causal LM with `LlamaForCausalLM`.

    Args:
        config: LlamaConfig
    """

    def __new__(cls, config: LlamaConfig):
        with replace_decoder(te_decoder_cls=TELlamaDecoderLayer):
            llama_for_causal_lm = LlamaForCausalLM(config)
        return llama_for_causal_lm

    @classmethod
    def from_pretrained_local(cls, pretrained_model_name_or_path, *args, config, **kwargs):
        """
        Custom method adapted from `from_pretrained` method in HuggingFace
        Transformers repo: https://github.com/huggingface/transformers/blob/f497f564bb76697edab09184a252fc1b1a326d1e/src/transformers/modeling_utils.py#L2579
        """
        # Before loading the model, set the default dtype for torch
        torch.set_default_dtype(kwargs["torch_dtype"])

        # Load the vanilla model weights
        vanilla_model = cls(config)
        subfolder = ""
        variant = None
        if os.path.isfile(
            os.path.join(
                pretrained_model_name_or_path,
                subfolder,
                _add_variant("model.safetensors.index.json", variant),
            )
        ):
            # Load from a sharded PyTorch checkpoint
            archive_file = os.path.join(
                pretrained_model_name_or_path,
                subfolder,
                _add_variant("model.safetensors.index.json", variant),
            )
            is_sharded = True
        elif os.path.isfile(
            os.path.join(
                pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant)
            )
        ):
            # Load from a sharded PyTorch checkpoint
            archive_file = os.path.join(
                pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant)
            )
            is_sharded = True
        else:
            raise AssertionError("Only sharded PyTorch ckpt format supported at the moment")

        resolved_archive_file, _ = get_checkpoint_shard_files(
            pretrained_model_name_or_path,
            archive_file,
        )

        # If the checkpoint is not sharded, it's a trivial sharding case
        if not is_sharded:
            assert not isinstance(resolved_archive_file, list)
            resolved_archive_file = [resolved_archive_file]

        for shard_file in resolved_archive_file:
            state_dict = load_state_dict(shard_file)
            # replace_params copies parameters relevant only to TransformerEngine
            replace_params(state_dict, vanilla_model.state_dict(), config)
            # load_state_dict copies parameters other than those in TransformerEngine
            vanilla_model.load_state_dict(state_dict, strict=False)

            # Force mem release. Taken from huggingface code
            del state_dict
            gc.collect()

        return vanilla_model


def replace_params(hf_state_dict, te_state_dict, config):
    # collect all layer prefixes to update
    all_layer_prefixes = set()
    for param_key in hf_state_dict.keys():
        layer_prefix_pat = "model.layers.\d+."
        m = re.match(layer_prefix_pat, param_key)
        if m is not None:
            all_layer_prefixes.add(m.group())

    for layer_prefix in all_layer_prefixes:
        # When loading weights into models with less number of layers, skip the
        # copy if the corresponding layer doesn't exist in HF model
        if layer_prefix + "input_layernorm.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "self_attention.layernorm_qkv.layer_norm_weight"].data[
                :
            ] = hf_state_dict[layer_prefix + "input_layernorm.weight"].data[:]

        if layer_prefix + "self_attn.q_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "self_attention.layernorm_qkv.query_weight"].data[:] = (
                hf_state_dict[layer_prefix + "self_attn.q_proj.weight"].data[:]
            )

        if layer_prefix + "self_attn.k_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "self_attention.layernorm_qkv.key_weight"].data[:] = (
                hf_state_dict[layer_prefix + "self_attn.k_proj.weight"].data[:]
            )

        if layer_prefix + "self_attn.v_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "self_attention.layernorm_qkv.value_weight"].data[:] = (
                hf_state_dict[layer_prefix + "self_attn.v_proj.weight"].data[:]
            )

        if layer_prefix + "self_attn.o_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "self_attention.proj.weight"].data[:] = hf_state_dict[
                layer_prefix + "self_attn.o_proj.weight"
            ].data[:]

        if layer_prefix + "post_attention_layernorm.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "layernorm_mlp.layer_norm_weight"].data[:] = hf_state_dict[
                layer_prefix + "post_attention_layernorm.weight"
            ].data[:]

        # It may happen that gate_proj.weight and up_proj.weight will be in the different files, so we need to
        # load them separately.
        if layer_prefix + "mlp.gate_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "layernorm_mlp.fc1_weight"].data[
                : config.intermediate_size
            ] = hf_state_dict[layer_prefix + "mlp.gate_proj.weight"].data

        if layer_prefix + "mlp.up_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "layernorm_mlp.fc1_weight"].data[
                config.intermediate_size :
            ] = hf_state_dict[layer_prefix + "mlp.up_proj.weight"].data

        if layer_prefix + "mlp.down_proj.weight" in hf_state_dict:
            te_state_dict[layer_prefix + "layernorm_mlp.fc2_weight"].data[:] = hf_state_dict[
                layer_prefix + "mlp.down_proj.weight"
            ].data[:]
    return all_layer_prefixes

##################
# main.py
##################

# Provide Huggingface Access Token
hyperparams.hf_access_token = ""
assert hyperparams.hf_access_token, "Provide a HF API Access Token!"

# Provide a directory to cache weights in to avoid downloading them every time.
# (By default, weights are cached in `~/.cache/huggingface/hub/models`)
hyperparams.weights_cache_dir = ""

# For Llama 2, uncomment this line (also set by default)
hyperparams.model_name = "meta-llama/Llama-2-7b-hf"

# For Llama 3, uncomment this line
# hyperparams.model_name = "meta-llama/Meta-Llama-3-8B"

hyperparams.mixed_precision = "bf16"


# Init the model and accelerator wrapper
model = init_te_llama_model(hyperparams)

# accelerator, model, optimizer, train_dataloader, lr_scheduler = wrap_with_accelerator(model, hyperparams)
#model = init_baseline_model(hyperparams)
#model = LlamaModel.from_pretrained("meta-llama/Llama-2-7b-hf")

print(model)
#print(f"TEST LLAMA MODEL {'\n'.join(model.state_dict().keys())}")

# # Finetune the model
# finetune_model(model, hyperparams, accelerator, train_dataloader, optimizer, lr_scheduler)

###################
##################
#################
# Inference
# model.eval()

# # Create a dummy input of zeros
# # Normally you would pass token IDs from the tokenizer, but here you want zeros

# device = torch.device(f"cuda:0")
# input_ids = torch.zeros((1, 10), dtype=torch.long, device=device)  # batch_size=1, seq_len=10

# # Run inference (no gradients needed)
# print(f"input ids: {type(input_ids)}")
# with torch.no_grad():
#     outputs = model(input_ids, output_hidden_states=True)

# # outputs.last_hidden_state is [batch_size, seq_len, hidden_size]
# print(outputs.logits[0])
# print(outputs.logits[0].shape)


config = AutoConfig.from_pretrained(
    hyperparams.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
)
# config.max_seq_length = args.max_seq_length
# config.micro_batch_size = args.micro_batch_size
from recipes.llama_native_te_accelerate.model import NVLlamaForCausalLM

model1 = NVLlamaForCausalLM.from_pretrained(hyperparams.model_name, config=config, trust_remote_code=True)

state_dict1 = model.state_dict()
state_dict2 = model1.state_dict()

# Compare layers
# for name, param1 in state_dict1.items():
#     if name not in state_dict2:
#         print(f"Layer {name} missing in model2")
#         continue

#     param2 = state_dict2[name]
#     if not torch.equal(param1, param2):
#         diff = torch.abs(param1 - param2).sum().item()
#         print(f"Layer {name} differs (sum abs diff = {diff:.4f})")

# model.eval()
# tokenizer = AutoTokenizer.from_pretrained(hyperparams.model_name)
# model_inputs = tokenizer(["The secret to baking a good cake is a "], return_tensors="pt")
# model_inputs.to(torch.device("cuda:0"))

# generated_ids = model.generate(**model_inputs, max_length=100)
# print(generated_ids)
# print(tokenizer.batch_decode(generated_ids)[0])
