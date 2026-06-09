# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SAE: Generic Sparse Autoencoder Package.

A domain-agnostic implementation of Sparse Autoencoders (SAEs) for
interpretability research. Provides multiple SAE architectures, training
utilities, and evaluation metrics.

Main Components:
    - architectures: SAE implementations (ReLU-L1, Top-K)
    - training: Training loop and configuration
    - eval: Evaluation metrics (reconstruction, loss recovered, dead latents)
    - utils: Utility functions (device, seed, memory)
"""

from .activation_store import (
    ActivationStore,
    ActivationStoreConfig,
    load_activations,
    save_activations,
)
from .analysis import (
    ClusterInfo,
    FeatureGeometry,
    FeatureLogits,
    FeatureStats,
    TopExample,
    build_cluster_label_prompt,
    compute_cluster_centroids,
    compute_feature_logits,
    compute_feature_stats,
    compute_feature_umap,
    export_text_features_parquet,
    launch_dashboard,
    save_cluster_labels,
    save_feature_atlas,
)
from .architectures import MoESAE, ReLUSAE, ShardedTopKSAE, SparseAutoencoder, TopKSAE
from .autointerp import (
    DEFAULT_PROMPT_TEMPLATE,
    TOKEN_PROMPT_TEMPLATE,
    AnthropicClient,
    AutoInterpreter,
    FeatureExamples,
    FeatureInterpretation,
    FeatureSampler,
    LLMClient,
    LLMResponse,
    NIMClient,
    NVIDIAInternalClient,
    OpenAIClient,
    OpenAICompatibleClient,
)
from .collector import (
    CollectorResult,
    TokenActivationCollector,
    TokenExample,
)
from .eval import (
    DeadLatentTracker,
    EvalResults,
    LossRecoveredResult,
    SparsityMetrics,
    compute_loss_recovered,
    compute_reconstruction_metrics,
    evaluate_loss_recovered,
    evaluate_sae,
    evaluate_sparsity,
)
from .kernels import HAS_TRITON, TritonDecoderAutograd
from .perf_logger import PerfLogger
from .process_group_manager import ProcessGroupManager
from .streaming import StreamingActivationDataset, StreamingConfig, make_streaming_dataloader
from .training import ParallelConfig, Trainer, TrainingConfig, WandbConfig
from .utils import get_device, set_seed


__version__ = "0.1.0"

__all__ = [
    "DEFAULT_PROMPT_TEMPLATE",
    "HAS_TRITON",
    "TOKEN_PROMPT_TEMPLATE",
    "ActivationStore",
    "ActivationStoreConfig",
    "AnthropicClient",
    "AutoInterpreter",
    "ClusterInfo",
    "CollectorResult",
    "DeadLatentTracker",
    "EvalResults",
    "FeatureExamples",
    "FeatureGeometry",
    "FeatureInterpretation",
    "FeatureLogits",
    "FeatureSampler",
    "FeatureStats",
    "LLMClient",
    "LLMResponse",
    "LossRecoveredResult",
    "MoESAE",
    "NIMClient",
    "NVIDIAInternalClient",
    "OpenAIClient",
    "OpenAICompatibleClient",
    "ParallelConfig",
    "PerfLogger",
    "ProcessGroupManager",
    "ReLUSAE",
    "ShardedTopKSAE",
    "SparseAutoencoder",
    "SparsityMetrics",
    "StreamingActivationDataset",
    "StreamingConfig",
    "TokenActivationCollector",
    "TokenExample",
    "TopExample",
    "TopKSAE",
    "Trainer",
    "TrainingConfig",
    "TritonDecoderAutograd",
    "WandbConfig",
    "build_cluster_label_prompt",
    "compute_cluster_centroids",
    "compute_feature_logits",
    "compute_feature_stats",
    "compute_feature_umap",
    "compute_loss_recovered",
    "compute_reconstruction_metrics",
    "evaluate_loss_recovered",
    "evaluate_sae",
    "evaluate_sparsity",
    "export_text_features_parquet",
    "get_device",
    "launch_dashboard",
    "load_activations",
    "make_streaming_dataloader",
    "save_activations",
    "save_cluster_labels",
    "save_feature_atlas",
    "set_seed",
]
