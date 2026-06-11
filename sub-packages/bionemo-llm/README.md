# bionemo-llm

The Bionemo Large Language Model (LLM) submodule contains common code used in submodules that train LLMs on biological
datasets. This includes data masking and collate functions, the
bio-BERT common architecture code, loss functions, and other NeMo / Megatron-LM compatibility functions. Sub-packages
should only depend on `bionemo-llm` if they need access to NeMo and Megatron-LM.
