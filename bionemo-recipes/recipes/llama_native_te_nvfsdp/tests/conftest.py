# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""
Pytest configuration for deterministic tests.

Sets global seeds to make all tests reproducible.
"""

import pytest
import torch
import random
import numpy as np


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set all random seeds for deterministic tests.
    
    This fixture runs automatically before each test.
    """
    seed = 42
    
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    yield
    
    # No cleanup needed

