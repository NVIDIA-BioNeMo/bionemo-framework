# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import logging
from typing import Mapping, Optional

from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only


class RankedLogger(logging.LoggerAdapter):
    """A multi-GPU-friendly python command line logger."""

    """A multi-GPU-friendly command-line logger.

    Prefixes messages with the process rank to aid debugging in distributed
    training. Can optionally restrict logs to rank zero only.
    """

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Initialize a rank-aware logger adapter.
        Logs on all processes with their rank prefixed in the message.

        Args:
            name: Logger name. Defaults to module ``__name__``.
            rank_zero_only: If True, only rank 0 emits logs.
            extra: Optional context mapping for `logging.LoggerAdapter`.

        """  # noqa: D205
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(self, level: int, msg: str, rank: Optional[int] = None, *args, **kwargs) -> None:
        """Delegate a log call to the underlying logger with rank prefixing.

        If `rank` is provided, logs only on that rank. Otherwise, logs on all
                ranks (subject to `rank_zero_only`).

        Args:
            level: Logging level (e.g., `logging.INFO`).
            msg: Message to log.
            rank: Optional rank to restrict logging to.
            args: Positional arguments for the underlying logger.
            kwargs: Keyword arguments for the underlying logger.

        Raises:
            RuntimeError: If `rank_zero_only.rank` is not set.

        """
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            current_rank = getattr(rank_zero_only, "rank", None)
            if current_rank is None:
                raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
            msg = rank_prefixed_message(msg, current_rank)
            if self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(level, msg, *args, **kwargs)
            else:
                if rank is None:
                    self.logger.log(level, msg, *args, **kwargs)
                elif current_rank == rank:
                    self.logger.log(level, msg, *args, **kwargs)
