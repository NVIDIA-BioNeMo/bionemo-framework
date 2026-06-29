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


import functools
import itertools
import re
from collections import Counter
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Any, Literal, Sequence

import pydantic
import yaml


__all__: Sequence[str] = (
    "Resource",
    "get_all_resources",
)


def _validate_ngc_resource(value: str) -> str:
    if not re.match(r"^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+){0,2}(:[A-Za-z0-9_.-]+)?$", value):
        raise ValueError("Pattern should be in format [org/[team/]]name[:version]")
    return value


class Resource(pydantic.BaseModel):
    """Class that represents a remote resource for downloading and caching test data."""

    model_config = pydantic.ConfigDict(use_attribute_docstrings=True)

    tag: Annotated[str, pydantic.StringConstraints(pattern=r"^[^/]*/[^/]*$")]  # Only slash between filename and tag.
    """A unique identifier for the resource. The file(s) will be accessible via load("filename/tag")."""

    ngc: Annotated[str, pydantic.AfterValidator(_validate_ngc_resource)] | None = None
    """The NGC URL for the resource.

    Should be in format [org/[team/]]name[:version]. If None, the resource is not available on NGC.
    """

    ngc_registry: Literal["model", "resource"] | None = None
    """The NGC resource type (model or resource) for the data. Must be provided if ngc is not None."""

    pbss: Annotated[pydantic.AnyUrl, pydantic.UrlConstraints(allowed_schemes=["s3"])]
    """The PBSS (NVIDIA-internal) URL of the resource."""

    sha256: str | None
    """The SHA256 checksum of the resource. If None, the SHA will not be checked on download (not recommended)."""

    owner: str
    """The owner or primary point of contact for the resource, in the format "Name <email>"."""

    description: str | None = None
    """A description of the file(s)."""

    unpack: Literal[False, None] = None
    """Whether the resource should be unpacked after download. If None, will defer to the file extension."""

    decompress: Literal[False, None] = None
    """Whether the resource should be decompressed after download. If None, will defer to the file extension."""

    @pydantic.model_validator(mode="after")
    def _validate_ngc_registry(self):
        if self.ngc and not self.ngc_registry:
            raise ValueError(f"ngc_registry must be provided if ngc is not None: {self.tag}")
        return self


@functools.cache
def get_all_resources(resource_path: Path | None = None) -> dict[str, Resource]:
    """Return a dictionary of all resources."""
    if not resource_path:
        resource_path = Path(files("bionemo.common.data").joinpath("resources"))  # type: ignore

    resources_files = itertools.chain(resource_path.glob("*.yaml"), resource_path.glob("*.yml"))

    all_resources = [resource for file in resources_files for resource in _parse_resource_file(file)]

    resource_list = pydantic.TypeAdapter(list[Resource]).validate_python(all_resources)
    resource_dict = {resource.tag: resource for resource in resource_list}

    if len(resource_dict) != len(resource_list):
        # Show the # of and which ones are duplicated so that a user can begin debugging and resolve the issue.
        tag_counts = Counter([resource.tag for resource in resource_list])
        raise ValueError(f"Duplicate resource tags found!: {[tag for tag, count in tag_counts.items() if count > 1]}")

    return resource_dict


def _parse_resource_file(file) -> list[dict[str, Any]]:
    with file.open("r") as f:
        resources = yaml.safe_load(f)
        for resource in resources:
            resource["tag"] = f"{file.stem}/{resource['tag']}"
        return resources
