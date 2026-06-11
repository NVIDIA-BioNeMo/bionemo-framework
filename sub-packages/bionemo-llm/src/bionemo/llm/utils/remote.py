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


import os
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Optional, Sequence
from urllib import request

import requests
from nemo.utils import logging

from bionemo.core import BIONEMO_CACHE_DIR


__all__: Sequence[str] = (
    "FTPRemoteResource",
    "RemoteResource",
)


@dataclass
class RemoteResource:
    """Responsible for downloading remote files, along with optional processing of downloaded files for downstream usecases.

    Each object is invoked through either its constructor (setting up the destination and checksum), or through a pre-configured class method.
    `download_resource()` contains the core functionality, which is to download the file at `url` to the fully qualified filename. Class methods
    can be used to further configure this process.

    Receive:
        a file, its checksum, a destination directory, and a root directory

        Our dataclass then provides some useful things:
            - fully qualified destination folder (property)
            - fully qualified destination file (property)
            - check_exists()
            - download_resource()

        Form the fully qualified destination folder.
        Create a fully qualified path for the file

        (all lives in the download routine)
        Check that the fq destination folder exists, otherwise create it
        Download the file.
        Checksum the download.
        Done.

        Postprocessing should be their own method with their own configuration.

    Example usage:
        >>> # The following will download and preprocess the prepackaged resources.
        >>> GRCh38Ensembl99ResourcePreparer().prepare()
        >>> Hg38chromResourcePreparer().prepare()
        >>> GRCh38p13_ResourcePreparer().prepare()


    Attributes:
        dest_directory: The directory to place the desired file upon completing the download. Should have the form {dest_directory}/{dest_filename}
        dest_filename: The desired name for the file upon completing the download.
        checksum: checksum associated with the file located at url. If set to None, check_exists only checks for the existance of `{dest_directory}/{dest_filename}`
        url: URL of the file to download
        root_directory: the bottom-level directory, the fully qualified path is formed by joining root_directory, dest_directory, and dest_filename.
    """

    checksum: Optional[str]
    dest_filename: str
    dest_directory: str
    root_directory: str | os.PathLike = BIONEMO_CACHE_DIR
    url: Optional[str] = None

    @property
    def fully_qualified_dest_folder(self):  # noqa: D102
        return Path(self.root_directory) / self.dest_directory

    @property
    def fully_qualified_dest_filename(self):
        """Returns the fully qualified destination path of the file.

        Example:
            /tmp/my_folder/file.tar.gz
        """
        return os.path.join(self.fully_qualified_dest_folder, self.dest_filename)

    def exists_or_create_destination_directory(self, exist_ok=True):
        """Checks that the `fully_qualified_destination_directory` exists, if it does not, the directory is created (or fails).

        exists_ok: Triest to create `fully_qualified_dest_folder` if it doesnt already exist.
        """
        os.makedirs(self.fully_qualified_dest_folder, exist_ok=exist_ok)

    @staticmethod
    def get_env_tmpdir():
        """Convenience method that exposes the environment TMPDIR variable."""
        return os.environ.get("TMPDIR", "/tmp")

    def download_resource(self, overwrite=False) -> str:
        """Downloads the resource to its specified fully_qualified_dest name.

        Returns: the fully qualified destination filename.
        """
        self.exists_or_create_destination_directory()

        if not self.check_exists() or overwrite:
            logging.info(f"Downloading resource: {self.url}")
            with requests.get(self.url, stream=True) as r, open(self.fully_qualified_dest_filename, "wb") as fd:
                r.raise_for_status()
                for bytes in r:
                    fd.write(bytes)
        else:
            logging.info(f"Resource already exists, skipping download: {self.url}")

        self.check_exists()
        return self.fully_qualified_dest_filename

    def check_exists(self):
        """Returns true if `fully_qualified_dest_filename` exists and the checksum matches `self.checksum`"""  # noqa: D415
        if os.path.exists(self.fully_qualified_dest_filename):
            with open(self.fully_qualified_dest_filename, "rb") as fd:
                data = fd.read()
                result = md5(data).hexdigest()
            if self.checksum is None:
                logging.info("No checksum provided, filename exists. Assuming it is complete.")
                matches = True
            else:
                matches = result == self.checksum
            return matches

        return False


class FTPRemoteResource(RemoteResource):  # noqa: D101
    def download_resource(self, overwrite=False) -> str:
        """Downloads the resource to its specified fully_qualified_dest name.

        Returns: the fully qualified destination filename.
        """
        self.exists_or_create_destination_directory()

        if not self.check_exists() or overwrite:
            request.urlretrieve(self.url, self.fully_qualified_dest_filename)

        self.check_exists()
        return self.fully_qualified_dest_filename
