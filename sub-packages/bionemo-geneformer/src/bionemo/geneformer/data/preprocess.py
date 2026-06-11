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


from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from bionemo.llm.utils.remote import RemoteResource


__all__: Sequence[str] = ("ResourcePreprocessor",)


@dataclass
class ResourcePreprocessor(ABC):
    """Interface defining a ResourcePreprocessor. Implementors promise to provide both a complete RemoteResource and a freeform
    preprocess method. This interface can be used to generically define a workflow from a config file.

        remote -> prepare -> prepared data.
    """  # noqa: D205

    root_directory: Optional[str] = field(default_factory=RemoteResource.get_env_tmpdir)
    dest_directory: str = "data"

    def get_checksums(self) -> List[str]:  # noqa: D102
        return [resource.checksum for resource in self.get_remote_resources()]

    def get_urls(self) -> List[str]:  # noqa: D102
        return [resource.url for resource in self.get_remote_resources()]

    @abstractmethod
    def get_remote_resources(self) -> List[RemoteResource]:
        """Gets the remote resources associated with this preparor."""
        raise NotImplementedError()

    @abstractmethod
    def prepare(self) -> List:
        """Returns a list of prepared filenames."""
        raise NotImplementedError()
