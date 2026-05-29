# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Constants and enums shared across SCDL format specification and implementation.

This module provides a single source of truth for:
- Array identifiers and their filesystem mappings
- Data type specifications
- Backend implementations
- File and metadata constants
"""

from enum import Enum, IntEnum


class FileNames(str, Enum):
    """All files in SCDL archive.

    This enum contains both array data files and special metadata files.
    For arrays, use the `array_name` property to get the canonical header name.
    """

    # Array data files
    DATA = "data.npy"
    ROWPTR = "row_ptr.npy"
    COLPTR = "col_ptr.npy"
    NEIGHBOR_INDICES = "neighbor_indices.npy"
    NEIGHBOR_INDICES_PTR = "neighbor_indptr.npy"
    NEIGHBOR_VALUES = "neighbor_values.npy"
    METADATA = "metadata.json"
    FEATURES = "features"
    VAR_FEATURES = "var_features"
    OBS_FEATURES = "obs_features"
    VERSION = "version.json"
    HEADER = "header.sch"


class ArrayDType(IntEnum):
    """Numpy dtype specification for arrays in SCDL archives.

    Integer values are used in the binary format for efficient storage.
    """

    UINT8_ARRAY = 1
    UINT16_ARRAY = 2
    UINT32_ARRAY = 3
    UINT64_ARRAY = 4
    FLOAT16_ARRAY = 5
    FLOAT32_ARRAY = 6
    FLOAT64_ARRAY = 7
    STRING_ARRAY = 8
    FIXED_STRING_ARRAY = 9

    @property
    def numpy_dtype_string(self) -> str:
        """Get the corresponding NumPy dtype string."""
        dtype_map = {
            self.UINT8_ARRAY: "uint8",
            self.UINT16_ARRAY: "uint16",
            self.UINT32_ARRAY: "uint32",
            self.UINT64_ARRAY: "uint64",
            self.FLOAT16_ARRAY: "float16",
            self.FLOAT32_ARRAY: "float32",
            self.FLOAT64_ARRAY: "float64",
            self.STRING_ARRAY: "string",
            self.FIXED_STRING_ARRAY: "fixed_string",
        }
        return dtype_map[self]

    @classmethod
    def from_numpy_dtype(cls, dtype) -> "ArrayDType":
        """Convert a numpy dtype to ArrayDType enum.

        Args:
            dtype: numpy dtype object or string representation

        Returns:
            Corresponding ArrayDType enum value

        Raises:
            ValueError: If dtype is not supported
        """
        # Convert dtype object to string if needed
        if isinstance(dtype, type) and hasattr(dtype, "__name__"):
            # Handle numpy type classes like np.float32, np.uint32
            dtype_str = dtype.__name__
        elif hasattr(dtype, "name"):
            # Handle numpy dtype instances
            dtype_str = dtype.name
        elif hasattr(dtype, "dtype"):
            dtype_str = dtype.dtype.name
        else:
            dtype_str = str(dtype)

        # Map numpy dtype strings to ArrayDType enums
        dtype_map = {
            "uint8": cls.UINT8_ARRAY,
            "uint16": cls.UINT16_ARRAY,
            "uint32": cls.UINT32_ARRAY,
            "uint64": cls.UINT64_ARRAY,
            "float16": cls.FLOAT16_ARRAY,
            "float32": cls.FLOAT32_ARRAY,
            "float64": cls.FLOAT64_ARRAY,
            "object": cls.STRING_ARRAY,  # Object arrays often contain strings
            "str": cls.STRING_ARRAY,
            "<U": cls.FIXED_STRING_ARRAY,  # Unicode string arrays
        }

        # Handle variations and aliases
        if dtype_str.startswith("<U") or dtype_str.startswith(">U"):
            return cls.FIXED_STRING_ARRAY
        elif dtype_str.startswith("<f") or dtype_str.startswith(">f"):
            if "4" in dtype_str:
                return cls.FLOAT32_ARRAY
            elif "8" in dtype_str:
                return cls.FLOAT64_ARRAY
            elif "2" in dtype_str:
                return cls.FLOAT16_ARRAY
        elif dtype_str.startswith(("<u", ">u")):
            if "1" in dtype_str:
                return cls.UINT8_ARRAY
            elif "2" in dtype_str:
                return cls.UINT16_ARRAY
            elif "4" in dtype_str:
                return cls.UINT32_ARRAY
            elif "8" in dtype_str:
                return cls.UINT64_ARRAY
        elif dtype_str.startswith(("<i", ">i")):
            raise ValueError(f"Signed integer dtypes are not supported: {dtype_str}")

        # Try direct mapping
        if dtype_str in dtype_map:
            return dtype_map[dtype_str]

        # Default fallback for common types
        if "float32" in dtype_str or "f4" in dtype_str:
            return cls.FLOAT32_ARRAY
        elif "float64" in dtype_str or "f8" in dtype_str:
            return cls.FLOAT64_ARRAY
        # Do not silently map signed ints; require explicit handling upstream
        elif "int32" in dtype_str or "i4" in dtype_str or "int64" in dtype_str or "i8" in dtype_str:
            raise ValueError(f"Signed integer dtypes are not supported: {dtype_str}")

        raise ValueError(f"Unsupported numpy dtype: {dtype_str} (original: {dtype})")


class Backend(IntEnum):
    """Backend implementations for SCDL archives.

    Defines how array data is stored and accessed.
    """

    MEMMAP_V0 = 1
    CHUNKED_MEMMAP_V0 = 2  # Chunked memmap for large datasets with remote access support


class Mode(str, Enum):
    """Valid modes for file I/O operations.

    The write append mode is 'w+' while the read append mode is 'r+'.
    """

    CREATE_APPEND = "w+"
    READ_APPEND = "r+"
    READ = "r"
    CREATE = "w"


class NeighborSamplingStrategy(str, Enum):
    """Valid sampling strategies for neighbor selection."""

    RANDOM = "random"
    FIRST = "first"


# Centralized dtype family orderings, for use in upscaling/validation across modules
INT_ORDER = ["uint8", "uint16", "uint32", "uint64"]
FLOAT_ORDER = ["float16", "float32", "float64"]
