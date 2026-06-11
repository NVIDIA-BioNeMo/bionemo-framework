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
from unittest.mock import MagicMock, patch

import pytest
from lightning_utilities.core.rank_zero import rank_zero_only

from src.utils.pylogger import RankedLogger


@pytest.fixture
def mock_logger():
    with patch("logging.getLogger") as mock_get_logger:
        logger = MagicMock(spec=logging.Logger)
        logger.isEnabledFor.return_value = True
        mock_get_logger.return_value = logger
        yield logger


def set_rank(rank):
    rank_zero_only.rank = rank


class TestRankedLogger:
    @pytest.mark.parametrize("rank", [0, 1])
    def test_log_all_ranks(self, mock_logger, rank):
        set_rank(rank)
        logger = RankedLogger(name="test_logger", rank_zero_only=False)

        msg = "test message"
        logger.log(logging.INFO, msg)

        mock_logger.log.assert_called_once()
        args, _ = mock_logger.log.call_args
        assert args[0] == logging.INFO
        assert f"[rank: {rank}] {msg}" in args[1]

    @pytest.mark.parametrize("rank", [0, 1])
    def test_log_rank_zero_only_flag(self, mock_logger, rank):
        set_rank(rank)
        logger = RankedLogger(name="test_logger", rank_zero_only=True)

        msg = "test message"
        logger.log(logging.INFO, msg)

        if rank == 0:
            mock_logger.log.assert_called_once()
            args, _ = mock_logger.log.call_args
            assert args[0] == logging.INFO
            assert f"[rank: {rank}] {msg}" in args[1]
        else:
            mock_logger.log.assert_not_called()

    @pytest.mark.parametrize("current_rank, target_rank", [(0, 0), (0, 1), (1, 0), (1, 1)])
    def test_log_specific_rank_kwarg(self, mock_logger, current_rank, target_rank):
        set_rank(current_rank)
        logger = RankedLogger(name="test_logger", rank_zero_only=False)

        msg = "test message"
        logger.log(logging.INFO, msg, rank=target_rank)

        if current_rank == target_rank:
            mock_logger.log.assert_called_once()
            args, _ = mock_logger.log.call_args
            assert args[0] == logging.INFO
            assert f"[rank: {current_rank}] {msg}" in args[1]
        else:
            mock_logger.log.assert_not_called()

    def test_rank_not_set(self, mock_logger):
        if hasattr(rank_zero_only, "rank"):
            del rank_zero_only.rank

        logger = RankedLogger(name="test_logger")
        with pytest.raises(RuntimeError, match="The `rank_zero_only.rank` needs to be set before use"):
            logger.log(logging.INFO, "test")

    def test_log_level_disabled(self, mock_logger):
        set_rank(0)
        # a bit of a hack to get the underlying logger to control this
        mock_logger.isEnabledFor.return_value = False

        logger = RankedLogger(name="test_logger")
        logger.log(logging.DEBUG, "a message")

        mock_logger.log.assert_not_called()
