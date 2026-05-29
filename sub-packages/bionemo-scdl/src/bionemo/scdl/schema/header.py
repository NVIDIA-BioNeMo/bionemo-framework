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


"""SCDL Archive Header Implementation.

This module provides comprehensive header serialization/deserialization for SCDL archives,
implementing the formal specification defined in scdl-schema.md.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

from bionemo.scdl.util.scdl_constants import ArrayDType, Backend

from .headerutil import BinaryHeaderCodec, Endianness, HeaderSerializationError
from .magic import SCDL_MAGIC_NUMBER
from .version import CurrentSCDLVersion, SCDLVersion


class ArrayInfo:
    """Information about an array in the SCDL archive.

    Represents metadata for a single array as defined in the SCDL schema specification.
    """

    def __init__(self, name: str, length: int, dtype: ArrayDType, shape: Optional[Tuple[int, ...]] = None):
        """Initialize array information.

        Args:
            name: Filename of the array
            length: Number of elements in the array
            dtype: Data type of the array elements
            shape: Optional shape tuple for multidimensional arrays
        """
        self.name = name
        self.length = length
        self.dtype = dtype
        self.shape = shape

    def serialize(self, codec: BinaryHeaderCodec) -> bytes:
        """Serialize this ArrayInfo to binary format.

        Args:
            codec: Binary codec for serialization

        Returns:
            Binary representation following SCDL schema

        Raises:
            HeaderSerializationError: If validation fails
        """
        # Validate before serialization (per schema requirements)
        self._validate()

        data = b""

        # name_len + name
        data += codec.pack_string(self.name)

        # length (uint64)
        data += codec.pack_uint64(self.length)

        # dtype (uint32 enum value)
        data += codec.pack_uint32(int(self.dtype))

        # has_shape + optional shape data
        if self.shape is not None:
            data += codec.pack_uint8(1)  # has_shape = true
            data += codec.pack_uint32(len(self.shape))  # shape_dims
            for dim in self.shape:
                data += codec.pack_uint32(dim)  # shape array
        else:
            data += codec.pack_uint8(0)  # has_shape = false

        return data

    def _validate(self) -> None:
        """Validate ArrayInfo according to SCDL schema requirements.

        Raises:
            HeaderSerializationError: If validation fails
        """
        # Schema requirement: All string lengths must be > 0
        if not self.name or len(self.name.strip()) == 0:
            raise HeaderSerializationError("Array name cannot be empty (schema requirement)")

        # Additional reasonable validations
        if self.length < 0:
            raise HeaderSerializationError(f"Array length cannot be negative: {self.length}")

        if self.shape is not None:
            if len(self.shape) == 0:
                raise HeaderSerializationError("Shape cannot be empty when specified")
            for i, dim in enumerate(self.shape):
                if dim <= 0:
                    raise HeaderSerializationError(f"Shape dimension {i} must be positive: {dim}")

        # Validate UTF-8 encoding
        try:
            self.name.encode("utf-8")
        except UnicodeEncodeError as e:
            raise HeaderSerializationError(f"Array name contains invalid UTF-8: {e}")

    @classmethod
    def deserialize(cls, codec: BinaryHeaderCodec, data: bytes, offset: int = 0) -> Tuple["ArrayInfo", int]:
        """Deserialize ArrayInfo from binary data.

        Args:
            codec: Binary codec for deserialization
            data: Binary data containing serialized ArrayInfo
            offset: Starting offset in data

        Returns:
            Tuple of (ArrayInfo instance, bytes consumed)

        Raises:
            HeaderSerializationError: If data is invalid
        """
        current_offset = offset

        # Read name
        name, name_bytes = codec.unpack_string(data[current_offset:])
        current_offset += name_bytes

        # Read length
        length = codec.unpack_uint64(data[current_offset : current_offset + 8])
        current_offset += 8

        # Read dtype
        dtype_value = codec.unpack_uint32(data[current_offset : current_offset + 4])
        current_offset += 4

        try:
            dtype = ArrayDType(dtype_value)
        except ValueError:
            raise HeaderSerializationError(f"Invalid ArrayDType value: {dtype_value}")

        # Read optional shape
        has_shape = codec.unpack_uint8(data[current_offset : current_offset + 1])
        current_offset += 1

        shape = None
        if has_shape:
            shape_dims = codec.unpack_uint32(data[current_offset : current_offset + 4])
            current_offset += 4

            shape = []
            for _ in range(shape_dims):
                dim = codec.unpack_uint32(data[current_offset : current_offset + 4])
                shape.append(dim)
                current_offset += 4
            shape = tuple(shape)

        array_info = cls(name=name, length=length, dtype=dtype, shape=shape)
        bytes_consumed = current_offset - offset

        return array_info, bytes_consumed

    def calculate_size(self) -> int:
        """Calculate the serialized size of this ArrayInfo in bytes."""
        # name_len (4) + name length + length (8) + dtype (4) + has_shape (1)
        size = 4 + len(self.name.encode("utf-8")) + 8 + 4 + 1

        if self.shape is not None:
            # shape_dims (4) + shape array (4 * dimensions)
            size += 4 + (4 * len(self.shape))

        return size

    def __str__(self) -> str:
        """Return a human-readable description of the array info.

        Returns:
            str: Summary including name, length, dtype, and optional shape.
        """
        shape_str = f", shape={self.shape}" if self.shape else ""
        return f"ArrayInfo(name='{self.name}', length={self.length}, dtype={self.dtype.name}{shape_str})"

    def __repr__(self) -> str:
        """Return a developer-focused representation of the array info.

        Returns:
            str: Representation mirroring ``__str__`` for succinct debugging.
        """
        return self.__str__()


class FeatureIndexInfo:
    """Information about a feature index in the SCDL archive.

    Feature indices provide fast lookups for specific features in the data.
    As specified in the schema, each FeatureIndex may optionally store a header.
    """

    def __init__(
        self,
        name: str,
        length: int,
        dtype: ArrayDType,
        index_files: Optional[List[str]] = None,
        shape: Optional[Tuple[int, ...]] = None,
    ):
        """Initialize feature index information.

        Args:
            name: Name of the feature index
            length: Number of entries in the index
            dtype: Data type of index entries
            index_files: List of paths to feature index files
            shape: Optional shape for multidimensional indices
        """
        self.name = name
        self.length = length
        self.dtype = dtype
        self.index_files = index_files or []
        self.shape = shape

    def serialize(self, codec: BinaryHeaderCodec) -> bytes:
        """Serialize this FeatureIndexInfo to binary format.

        Args:
            codec: Binary codec for serialization

        Returns:
            Binary representation following SCDL schema

        Raises:
            HeaderSerializationError: If validation fails
        """
        # Validate before serialization
        self._validate()

        data = b""

        # name_len + name
        data += codec.pack_string(self.name)

        # length (uint64)
        data += codec.pack_uint64(self.length)

        # dtype (uint32 enum value)
        data += codec.pack_uint32(int(self.dtype))

        # index_files_count + index_files
        data += codec.pack_uint32(len(self.index_files))
        for file_path in self.index_files:
            data += codec.pack_string(file_path)

        # has_shape + optional shape data
        if self.shape is not None:
            data += codec.pack_uint8(1)  # has_shape = true
            data += codec.pack_uint32(len(self.shape))  # shape_dims
            for dim in self.shape:
                data += codec.pack_uint32(dim)  # shape array
        else:
            data += codec.pack_uint8(0)  # has_shape = false

        return data

    @classmethod
    def deserialize(cls, codec: BinaryHeaderCodec, data: bytes, offset: int = 0) -> Tuple["FeatureIndexInfo", int]:
        """Deserialize FeatureIndexInfo from binary data.

        Args:
            codec: Binary codec for deserialization
            data: Binary data containing serialized FeatureIndexInfo
            offset: Starting offset in data

        Returns:
            Tuple of (FeatureIndexInfo instance, bytes consumed)

        Raises:
            HeaderSerializationError: If data is invalid
        """
        current_offset = offset

        # Read name
        name, name_bytes = codec.unpack_string(data[current_offset:])
        current_offset += name_bytes

        # Read length
        length = codec.unpack_uint64(data[current_offset : current_offset + 8])
        current_offset += 8

        # Read dtype
        dtype_value = codec.unpack_uint32(data[current_offset : current_offset + 4])
        current_offset += 4

        try:
            dtype = ArrayDType(dtype_value)
        except ValueError:
            raise HeaderSerializationError(f"Invalid ArrayDType value in FeatureIndex: {dtype_value}")

        # Read index files
        files_count = codec.unpack_uint32(data[current_offset : current_offset + 4])
        current_offset += 4

        index_files = []
        for _ in range(files_count):
            file_path, file_bytes = codec.unpack_string(data[current_offset:])
            index_files.append(file_path)
            current_offset += file_bytes

        # Read optional shape
        has_shape = codec.unpack_uint8(data[current_offset : current_offset + 1])
        current_offset += 1

        shape = None
        if has_shape:
            shape_dims = codec.unpack_uint32(data[current_offset : current_offset + 4])
            current_offset += 4

            shape = []
            for _ in range(shape_dims):
                dim = codec.unpack_uint32(data[current_offset : current_offset + 4])
                shape.append(dim)
                current_offset += 4
            shape = tuple(shape)

        feature_index = cls(name=name, length=length, dtype=dtype, index_files=index_files, shape=shape)
        bytes_consumed = current_offset - offset

        return feature_index, bytes_consumed

    def _validate(self) -> None:
        """Validate FeatureIndexInfo according to SCDL schema requirements.

        Raises:
            HeaderSerializationError: If validation fails
        """
        # Schema requirement: All string lengths must be > 0
        if not self.name or len(self.name.strip()) == 0:
            raise HeaderSerializationError("FeatureIndex name cannot be empty (schema requirement)")

        # Validate index files
        for i, file_path in enumerate(self.index_files):
            if not file_path or len(file_path.strip()) == 0:
                raise HeaderSerializationError(f"FeatureIndex file path {i} cannot be empty")

        # Additional reasonable validations
        if self.length < 0:
            raise HeaderSerializationError(f"FeatureIndex length cannot be negative: {self.length}")

        if self.shape is not None:
            if len(self.shape) == 0:
                raise HeaderSerializationError("FeatureIndex shape cannot be empty when specified")
            for i, dim in enumerate(self.shape):
                if dim <= 0:
                    raise HeaderSerializationError(f"FeatureIndex shape dimension {i} must be positive: {dim}")

        # Validate UTF-8 encoding
        try:
            self.name.encode("utf-8")
            for file_path in self.index_files:
                file_path.encode("utf-8")
        except UnicodeEncodeError as e:
            raise HeaderSerializationError(f"FeatureIndex contains invalid UTF-8: {e}")

    def calculate_size(self) -> int:
        """Calculate the serialized size of this FeatureIndexInfo in bytes."""
        # name_len (4) + name length + length (8) + dtype (4) + files_count (4)
        size = 4 + len(self.name.encode("utf-8")) + 8 + 4 + 4

        # Add size for each file path
        for file_path in self.index_files:
            size += 4 + len(file_path.encode("utf-8"))  # len + content

        # has_shape (1)
        size += 1

        if self.shape is not None:
            # shape_dims (4) + shape array (4 * dimensions)
            size += 4 + (4 * len(self.shape))

        return size

    def __str__(self) -> str:
        """Return a human-readable description of the feature index info.

        Returns:
            str: Summary including name, length, dtype, file count, and optional shape.
        """
        shape_str = f", shape={self.shape}" if self.shape else ""
        files_str = f", files={len(self.index_files)}"
        return f"FeatureIndexInfo(name='{self.name}', length={self.length}, dtype={self.dtype.name}{files_str}{shape_str})"

    def __repr__(self) -> str:
        """Return a developer-focused representation of the feature index info.

        Returns:
            str: Representation mirroring ``__str__`` for succinct debugging.
        """
        return self.__str__()


class ChunkedInfo:
    """Chunking metadata for CHUNKED_MEMMAP backend."""

    def __init__(self, chunk_size: int, num_chunks: int, total_rows: int):
        """Initialize chunked info."""
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        self.total_rows = total_rows

    def get_chunk_for_row(self, global_idx: int) -> Tuple[int, int]:
        """Return (chunk_id, local_idx) for a global row index."""
        if global_idx < 0 or global_idx >= self.total_rows:
            raise IndexError(f"Row index {global_idx} out of range [0, {self.total_rows})")
        return global_idx // self.chunk_size, global_idx % self.chunk_size

    def serialize(self, codec: BinaryHeaderCodec) -> bytes:
        """Serialize to binary format."""
        return (
            codec.pack_uint32(self.chunk_size)
            + codec.pack_uint32(self.num_chunks)
            + codec.pack_uint64(self.total_rows)
        )

    @classmethod
    def deserialize(cls, codec: BinaryHeaderCodec, data: bytes, offset: int = 0) -> Tuple["ChunkedInfo", int]:
        """Deserialize from binary data. Returns (ChunkedInfo, bytes_consumed)."""
        chunk_size = codec.unpack_uint32(data[offset : offset + 4])
        num_chunks = codec.unpack_uint32(data[offset + 4 : offset + 8])
        total_rows = codec.unpack_uint64(data[offset + 8 : offset + 16])
        return cls(chunk_size=chunk_size, num_chunks=num_chunks, total_rows=total_rows), 16


class SCDLHeader:
    """Header for a SCDL archive following the official schema specification.

    Contains metadata about the archive including version, backend, and array information.
    The header is stored in binary format and is not human-readable by design.
    """

    # Core header size is fixed at 16 bytes
    CORE_HEADER_SIZE = 16

    def __init__(
        self,
        version: Optional[SCDLVersion] = None,
        backend: Backend = Backend.MEMMAP_V0,
        arrays: Optional[List[ArrayInfo]] = None,
        feature_indices: Optional[List[FeatureIndexInfo]] = None,
        chunked_info: Optional["ChunkedInfo"] = None,
    ):
        """Initialize SCDL header.

        Args:
            version: SCDL schema version (defaults to current version)
            backend: Storage backend type
            arrays: List of arrays in the archive
            feature_indices: Optional list of feature indices in the archive
            chunked_info: Optional chunking metadata for CHUNKED_MEMMAP backend
        """
        self.version = version or CurrentSCDLVersion()
        self.endianness = Endianness.NETWORK  # Always network byte order per spec
        self.backend = backend
        self.arrays = arrays or []
        self.feature_indices = feature_indices or []
        self.chunked_info = chunked_info

        # Create codec with network byte order
        self._codec = BinaryHeaderCodec(self.endianness)

    def add_array(self, array_info: ArrayInfo) -> None:
        """Add an array to the header."""
        self.arrays.append(array_info)

    def get_array(self, name: str) -> Optional[ArrayInfo]:
        """Get array info by name."""
        for array in self.arrays:
            if array.name == name:
                return array
        return None

    def remove_array(self, name: str) -> bool:
        """Remove array by name. Returns True if found and removed."""
        for i, array in enumerate(self.arrays):
            if array.name == name:
                del self.arrays[i]
                return True
        return False

    def add_feature_index(self, feature_index: FeatureIndexInfo) -> None:
        """Add a feature index to the header."""
        self.feature_indices.append(feature_index)

    def get_feature_index(self, name: str) -> Optional[FeatureIndexInfo]:
        """Get feature index info by name."""
        for feature_index in self.feature_indices:
            if feature_index.name == name:
                return feature_index
        return None

    def remove_feature_index(self, name: str) -> bool:
        """Remove feature index by name. Returns True if found and removed."""
        for i, feature_index in enumerate(self.feature_indices):
            if feature_index.name == name:
                del self.feature_indices[i]
                return True
        return False

    def serialize(self) -> bytes:
        """Serialize the header to binary format following SCDL schema.

        Returns:
            Binary representation of the complete header

        Raises:
            HeaderSerializationError: If serialization fails
        """
        try:
            # Validate header before serialization
            self.validate()

            data = b""

            # Core Header (16 bytes fixed)
            # Magic number (4 bytes)
            data += SCDL_MAGIC_NUMBER

            # Version (3 bytes: major, minor, point)
            data += self._codec.pack_uint8(self.version.major)
            data += self._codec.pack_uint8(self.version.minor)
            data += self._codec.pack_uint8(self.version.point)

            # Endianness (1 byte) - always NETWORK per spec
            data += self._codec.pack_uint8(1)  # NETWORK = 1

            # Backend (4 bytes)
            data += self._codec.pack_uint32(int(self.backend))

            # Array count (4 bytes) - schema requires this matches actual descriptors
            array_count = len(self.arrays)
            data += self._codec.pack_uint32(array_count)

            # Array descriptors (variable size)
            for array in self.arrays:
                data += array.serialize(self._codec)

            # Feature indices (optional extension after arrays)
            # feature_index_count (4 bytes)
            data += self._codec.pack_uint32(len(self.feature_indices))

            # Feature index descriptors (variable size)
            for feature_index in self.feature_indices:
                data += feature_index.serialize(self._codec)

            # Chunked info (optional, for CHUNKED_MEMMAP backend)
            if self.chunked_info is not None:
                data += self._codec.pack_uint8(1)  # has_chunked_info = true
                data += self.chunked_info.serialize(self._codec)
            else:
                data += self._codec.pack_uint8(0)  # has_chunked_info = false

            return data

        except Exception as e:
            raise HeaderSerializationError(f"Failed to serialize SCDL header: {e}")

    @classmethod
    def deserialize(cls, data: bytes) -> "SCDLHeader":
        """Deserialize header from binary data.

        Args:
            data: Binary data containing SCDL header

        Returns:
            SCDLHeader instance

        Raises:
            HeaderSerializationError: If deserialization fails or data is invalid
        """
        if len(data) < cls.CORE_HEADER_SIZE:
            raise HeaderSerializationError(
                f"Header data too short: {len(data)} bytes < {cls.CORE_HEADER_SIZE} bytes minimum"
            )

        # Use network byte order for reading
        codec = BinaryHeaderCodec(Endianness.NETWORK)
        offset = 0

        try:
            # Validate magic number
            magic = data[offset : offset + 4]
            if magic != SCDL_MAGIC_NUMBER:
                raise HeaderSerializationError(f"Invalid magic number: {magic} != {SCDL_MAGIC_NUMBER}")
            offset += 4

            # Read version
            version_major = codec.unpack_uint8(data[offset : offset + 1])
            offset += 1
            version_minor = codec.unpack_uint8(data[offset : offset + 1])
            offset += 1
            version_point = codec.unpack_uint8(data[offset : offset + 1])
            offset += 1

            version = SCDLVersion()
            version.major = version_major
            version.minor = version_minor
            version.point = version_point

            # Read and validate endianness
            endianness_value = codec.unpack_uint8(data[offset : offset + 1])
            offset += 1
            if endianness_value != 1:  # Must be NETWORK
                raise HeaderSerializationError(f"Invalid endianness: {endianness_value} (must be 1 for NETWORK)")

            # Read backend
            backend_value = codec.unpack_uint32(data[offset : offset + 4])
            offset += 4
            try:
                backend = Backend(backend_value)
            except ValueError:
                raise HeaderSerializationError(f"Invalid backend value: {backend_value}")

            # Read array count
            array_count = codec.unpack_uint32(data[offset : offset + 4])
            offset += 4

            # Read array descriptors
            arrays = []
            for i in range(array_count):
                if offset >= len(data):
                    raise HeaderSerializationError(f"Unexpected end of data while reading array {i}")

                array_info, bytes_consumed = ArrayInfo.deserialize(codec, data, offset)
                arrays.append(array_info)
                offset += bytes_consumed

            # Read feature indices (optional, for backwards compatibility)
            feature_indices = []
            if offset < len(data):
                # Check if we have enough data for feature index count
                if offset + 4 <= len(data):
                    feature_index_count = codec.unpack_uint32(data[offset : offset + 4])
                    offset += 4

                    # Read feature index descriptors
                    for i in range(feature_index_count):
                        if offset >= len(data):
                            raise HeaderSerializationError(f"Unexpected end of data while reading feature index {i}")

                        feature_index, bytes_consumed = FeatureIndexInfo.deserialize(codec, data, offset)
                        feature_indices.append(feature_index)
                        offset += bytes_consumed

            # Read chunked info (optional, for backwards compatibility)
            chunked_info = None
            if offset < len(data):
                has_chunked_info = codec.unpack_uint8(data[offset : offset + 1])
                offset += 1
                if has_chunked_info:
                    chunked_info, bytes_consumed = ChunkedInfo.deserialize(codec, data, offset)
                    offset += bytes_consumed

            header = cls(
                version=version,
                backend=backend,
                arrays=arrays,
                feature_indices=feature_indices,
                chunked_info=chunked_info,
            )
            return header

        except HeaderSerializationError:
            raise
        except Exception as e:
            raise HeaderSerializationError(f"Failed to deserialize SCDL header: {e}")

    def save(self, file_path: str) -> None:
        """Save the header to a binary file.

        Args:
            file_path: Path to save the header file

        Raises:
            HeaderSerializationError: If saving fails
        """
        try:
            with open(file_path, "wb") as f:
                f.write(self.serialize())
        except Exception as e:
            raise HeaderSerializationError(f"Failed to save header to {file_path}: {e}")

    @classmethod
    def load(cls, file_path: str) -> "SCDLHeader":
        """Load header from a binary file.

        Args:
            file_path: Path to the header file

        Returns:
            SCDLHeader instance

        Raises:
            HeaderSerializationError: If loading fails
        """
        try:
            with open(file_path, "rb") as f:
                data = f.read()
            return cls.deserialize(data)
        except FileNotFoundError:
            raise HeaderSerializationError(f"Header file not found: {file_path}")
        except Exception as e:
            raise HeaderSerializationError(f"Failed to load header from {file_path}: {e}")

    def calculate_total_size(self) -> int:
        """Calculate the total serialized size of the header in bytes."""
        total_size = self.CORE_HEADER_SIZE

        # Array descriptors
        for array in self.arrays:
            total_size += array.calculate_size()

        # Feature index count (4 bytes) + feature index descriptors
        total_size += 4
        for feature_index in self.feature_indices:
            total_size += feature_index.calculate_size()

        return total_size

    def validate(self) -> None:
        """Validate the header for consistency and correctness.

        Raises:
            HeaderSerializationError: If validation fails
        """
        # Check version compatibility
        current_version = CurrentSCDLVersion()
        if self.version.major > current_version.major:
            raise HeaderSerializationError(f"Unsupported version: {self.version} > {current_version}")

        # Check array names are unique
        names = [array.name for array in self.arrays]
        if len(names) != len(set(names)):
            raise HeaderSerializationError("Duplicate array names found")

        # Check array names are valid
        for array in self.arrays:
            if not array.name or not array.name.strip():
                raise HeaderSerializationError("Empty array name found")
            if len(array.name.encode("utf-8")) > 1024:  # Reasonable limit
                raise HeaderSerializationError(f"Array name too long: {array.name}")

        # Check feature index names are unique
        feature_names = [fi.name for fi in self.feature_indices]
        if len(feature_names) != len(set(feature_names)):
            raise HeaderSerializationError("Duplicate feature index names found")

        # Check feature index names are valid
        for feature_index in self.feature_indices:
            if not feature_index.name or not feature_index.name.strip():
                raise HeaderSerializationError("Empty feature index name found")
            if len(feature_index.name.encode("utf-8")) > 1024:  # Reasonable limit
                raise HeaderSerializationError(f"Feature index name too long: {feature_index.name}")

        # Check for name conflicts between arrays and feature indices
        all_names = names + feature_names
        if len(all_names) != len(set(all_names)):
            raise HeaderSerializationError("Name conflicts between arrays and feature indices")

    def __str__(self) -> str:
        """Return a human-readable string representation of the header."""
        return (
            f"SCDLHeader(version={self.version}, backend={self.backend.name}, "
            f"arrays={len(self.arrays)}, feature_indices={len(self.feature_indices)})"
        )

    def __repr__(self) -> str:
        """Return a developer-focused representation of the header.

        Returns:
            str: Representation mirroring ``__str__`` for succinct debugging.
        """
        return self.__str__()

    def to_json(self) -> str:
        """Return a JSON string representation of the header.

        Note: This is for debugging/inspection only, not for serialization.
        """

        def default(o):
            if hasattr(o, "name"):
                return o.name
            if hasattr(o, "__dict__"):
                return o.__dict__
            return str(o)

        data = {
            "version": {"major": self.version.major, "minor": self.version.minor, "point": self.version.point},
            "endianness": self.endianness.name,
            "backend": self.backend.name,
            "arrays": [
                {"name": array.name, "length": array.length, "dtype": array.dtype.name, "shape": array.shape}
                for array in self.arrays
            ],
            "feature_indices": [
                {
                    "name": fi.name,
                    "length": fi.length,
                    "dtype": fi.dtype.name,
                    "index_files": fi.index_files,
                    "shape": fi.shape,
                }
                for fi in self.feature_indices
            ],
        }

        return json.dumps(data, indent=2, default=default)

    def to_yaml(self) -> str:
        """Return a YAML string representation of the header.

        Note: This is for debugging/inspection only, not for serialization.
        """
        try:
            import yaml
        except ImportError:
            raise RuntimeError("PyYAML is required for YAML serialization")

        data = {
            "version": f"{self.version.major}.{self.version.minor}.{self.version.point}",
            "endianness": self.endianness.name,
            "backend": self.backend.name,
            "arrays": [
                {
                    "name": array.name,
                    "length": array.length,
                    "dtype": array.dtype.name,
                    "shape": list(array.shape) if array.shape else None,
                }
                for array in self.arrays
            ],
            "feature_indices": [
                {
                    "name": fi.name,
                    "length": fi.length,
                    "dtype": fi.dtype.name,
                    "index_files": fi.index_files,
                    "shape": list(fi.shape) if fi.shape else None,
                }
                for fi in self.feature_indices
            ],
        }

        return yaml.dump(data, default_flow_style=False)


# Utility functions for header operations


def create_header_from_arrays(
    array_files: List[str], backend: Backend = Backend.MEMMAP_V0, version: Optional[SCDLVersion] = None
) -> SCDLHeader:
    """Create a SCDL header by scanning array files.

    Args:
        array_files: List of array file paths to include
        backend: Storage backend to use
        version: Schema version (defaults to current)

    Returns:
        SCDLHeader with arrays automatically detected

    Note:
        This function creates placeholder ArrayInfo objects.
        Real implementations should inspect files to determine actual properties.
    """
    header = SCDLHeader(version=version, backend=backend)

    for file_path in array_files:
        path = Path(file_path)
        array_info = ArrayInfo(
            name=path.name,
            length=0,  # Would be determined by inspecting file
            dtype=ArrayDType.FLOAT32_ARRAY,  # Would be determined by inspecting file
            shape=None,  # Would be determined by inspecting file
        )
        header.add_array(array_info)

    return header


def validate_header_compatibility(header1: SCDLHeader, header2: SCDLHeader) -> bool:
    """Check if two headers are compatible for operations like merging.

    Args:
        header1: First header
        header2: Second header

    Returns:
        True if headers are compatible
    """
    # Check version compatibility (same major version)
    if header1.version.major != header2.version.major:
        return False

    # Check backend compatibility
    if header1.backend != header2.backend:
        return False

    # Check for conflicting array names
    names1 = {array.name for array in header1.arrays}
    names2 = {array.name for array in header2.arrays}

    if names1.intersection(names2):
        return False

    # Check for conflicting feature index names
    fi_names1 = {fi.name for fi in header1.feature_indices}
    fi_names2 = {fi.name for fi in header2.feature_indices}

    if fi_names1.intersection(fi_names2):
        return False

    # Check for conflicts between arrays and feature indices across headers
    all_names1 = names1.union(fi_names1)
    all_names2 = names2.union(fi_names2)

    if all_names1.intersection(all_names2):
        return False

    return True


def merge_headers(header1: SCDLHeader, header2: SCDLHeader) -> SCDLHeader:
    """Merge two compatible headers into a single header.

    Args:
        header1: First header
        header2: Second header

    Returns:
        Merged header

    Raises:
        HeaderSerializationError: If headers are incompatible
    """
    if not validate_header_compatibility(header1, header2):
        raise HeaderSerializationError("Headers are not compatible for merging")

    # Use the newer version
    if header1.version.minor >= header2.version.minor:
        version = header1.version
    else:
        version = header2.version

    merged_header = SCDLHeader(
        version=version,
        backend=header1.backend,
        arrays=header1.arrays + header2.arrays,
        feature_indices=header1.feature_indices + header2.feature_indices,
    )

    return merged_header


class HeaderReader:
    """Optimized reader for SCDL headers with caching and validation.

    Provides efficient access to header information without full deserialization
    when only specific fields are needed.
    """

    def __init__(self, file_path: str):
        """Initialize with header file path."""
        self.file_path = file_path
        self._cached_header = None
        self._core_header_cached = False
        self._magic = None
        self._version = None
        self._backend = None
        self._array_count = None

    def validate_magic(self) -> bool:
        """Quickly validate magic number without full deserialization."""
        if self._magic is None:
            with open(self.file_path, "rb") as f:
                self._magic = f.read(4)
        return self._magic == SCDL_MAGIC_NUMBER

    def get_version(self) -> SCDLVersion:
        """Get version information quickly."""
        self._ensure_core_header()
        return self._version

    def get_backend(self) -> Backend:
        """Get backend information quickly."""
        self._ensure_core_header()
        return self._backend

    def get_array_count(self) -> int:
        """Get array count quickly."""
        self._ensure_core_header()
        return self._array_count

    def get_full_header(self) -> SCDLHeader:
        """Get complete header (cached after first access)."""
        if self._cached_header is None:
            self._cached_header = SCDLHeader.load(self.file_path)
        return self._cached_header

    def _ensure_core_header(self):
        """Read core header fields if not cached."""
        if self._core_header_cached:
            return

        codec = BinaryHeaderCodec(Endianness.NETWORK)
        with open(self.file_path, "rb") as f:
            core_data = f.read(SCDLHeader.CORE_HEADER_SIZE)

        if len(core_data) < SCDLHeader.CORE_HEADER_SIZE:
            raise HeaderSerializationError("Invalid header file")

        offset = 0

        # Magic number
        self._magic = core_data[offset : offset + 4]
        offset += 4

        # Version
        version = SCDLVersion()
        version.major = codec.unpack_uint8(core_data[offset : offset + 1])
        offset += 1
        version.minor = codec.unpack_uint8(core_data[offset : offset + 1])
        offset += 1
        version.point = codec.unpack_uint8(core_data[offset : offset + 1])
        offset += 1
        self._version = version

        # Skip endianness
        offset += 1

        # Backend
        backend_value = codec.unpack_uint32(core_data[offset : offset + 4])
        self._backend = Backend(backend_value)
        offset += 4

        # Array count
        self._array_count = codec.unpack_uint32(core_data[offset : offset + 4])

        self._core_header_cached = True
