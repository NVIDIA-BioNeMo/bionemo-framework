# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Tokenizer construction bound to one immutable JSON byte snapshot."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from bionemo.evo2.vllm.artifact_io import JsonSnapshot, read_json_snapshot


def _snapshot_identity(snapshot: JsonSnapshot) -> tuple[int, int, int, str]:
    return (
        snapshot.device,
        snapshot.inode,
        snapshot.size_bytes,
        snapshot.sha256,
    )


@dataclass(frozen=True)
class SnapshotBoundTokenizer:
    """A tokenizer loaded from captured bytes with a rescan-verifiable source."""

    path: Path
    source_sha256: str
    source_size_bytes: int
    source_device: int
    source_inode: int
    loaded_object_sha256: str
    _tokenizer: Any = field(repr=False, compare=False)

    @classmethod
    def from_path(cls, path: str | Path) -> SnapshotBoundTokenizer:
        """Construct from captured JSON bytes and reject source drift during load."""
        from tokenizers import Tokenizer

        snapshot = read_json_snapshot(path, label="tokenizer JSON")
        tokenizer = Tokenizer.from_str(snapshot.payload.decode("utf-8"))
        loaded_object_sha256 = hashlib.sha256(tokenizer.to_str().encode("utf-8")).hexdigest()
        post_load = read_json_snapshot(path, label="post-load tokenizer JSON")
        if _snapshot_identity(post_load) != _snapshot_identity(snapshot):
            raise RuntimeError("tokenizer source changed during tokenizer load")
        return cls(
            path=snapshot.path,
            source_sha256=snapshot.sha256,
            source_size_bytes=snapshot.size_bytes,
            source_device=snapshot.device,
            source_inode=snapshot.inode,
            loaded_object_sha256=loaded_object_sha256,
            _tokenizer=tokenizer,
        )

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode text with the tokenizer built from the retained snapshot."""
        return tuple(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode token IDs with the tokenizer built from the retained snapshot."""
        return self._tokenizer.decode(list(token_ids), skip_special_tokens=False)

    def __call__(self, token_ids: Sequence[int]) -> str:
        """Decode token IDs so the bound tokenizer can serve as a decoder callback."""
        return self.decode(token_ids)

    def verify_source(self) -> dict[str, Any]:
        """Rescan the source and require its exact original inode and digest."""
        current = read_json_snapshot(self.path, label="tokenizer JSON rescan")
        expected = (
            self.source_device,
            self.source_inode,
            self.source_size_bytes,
            self.source_sha256,
        )
        if _snapshot_identity(current) != expected:
            raise RuntimeError("tokenizer source no longer matches the loaded byte snapshot")
        return self.provenance() | {"post_load_source_rescan_passed": True}

    def provenance(self) -> dict[str, Any]:
        """Return exact source and loaded-object identity."""
        return {
            "schema_version": 1,
            "path": str(self.path),
            "source_sha256": self.source_sha256,
            "source_size_bytes": self.source_size_bytes,
            "source_device": self.source_device,
            "source_inode": self.source_inode,
            "loaded_object_sha256": self.loaded_object_sha256,
            "constructed_from_captured_bytes": True,
        }


__all__ = ["SnapshotBoundTokenizer"]
