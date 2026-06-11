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

import warnings


def permute(index: int, length: int, seed: int) -> int:
    """Index into a permuted array with constant space and time complexity.

    This function permutes an index `i` into a range `[0, l)` using a hash function. See
    https://afnan.io/posts/2019-04-05-explaining-the-hashed-permutation/ for more details and
    "Correlated Multi-Jittered Sampling" by Andrew Kensler for the original algorithm.

    Args:
        index: The index to permute.
        length: The range of the permuted index.
        seed: The permutation seed.

    Returns:
        The permuted index in range(0, length).
    """
    if length <= 1:
        raise ValueError("The length of the permuted range must be greater than 1.")

    if index not in range(length):
        raise ValueError("The index to permute must be in the range [0, l).")

    if seed < 0:
        raise ValueError("The permutation seed must be greater than or equal to 0.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        w = length - 1
        w |= w >> 1
        w |= w >> 2
        w |= w >> 4
        w |= w >> 8
        w |= w >> 16

        while True:
            index ^= seed
            index *= 0xE170893D
            index ^= seed >> 16
            index ^= (index & w) >> 4
            index ^= seed >> 8
            index *= 0x0929EB3F
            index ^= seed >> 23
            index ^= (index & w) >> 1
            index *= 1 | seed >> 27
            index *= 0x6935FA69
            index ^= (index & w) >> 11
            index *= 0x74DCB303
            index ^= (index & w) >> 2
            index *= 0x9E501CC3
            index ^= (index & w) >> 2
            index *= 0xC860A3DF
            index &= w
            if index < length:
                break

    return (index + seed) % length
