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
from typing import Any, Dict, List

from nemo.lightning import io


class WillHaveGetSetHparam(ABC):
    """An ABC that states that a particular class _will_ have our mutatable IO Mixin variant added to it.

    This is a placeholder until a similar piece of functionality is added in NeMo.


    Raises:
        NotImplementedError: You must implement set_hparam, get_hparam, and get_hparams
    """

    @abstractmethod
    def set_hparam(self, attribute: str, value: Any, also_change_value: bool = True) -> None:
        """Mutates the saved hyper-parameter for the io mixed class.

        If you would like to only change the saved hyper-param
            for example in the case of loading a dataclass where the same variables are mutated to other non-savable
            entities by deterministic rules after init, then use `also_change_value=False` to only update the
            hyper-parameter.

        Args:
            attribute: The element name to modify within the saved init settings for self
            value: New parameter for the saved init settings
            also_change_value: If you also want to mutate the attribute of this same name in self to be the desired
                value, set this to True, otherwise if the init arg and self arg are expected to be divergent, then
                do not set this and modify the self attribute separately in the normal pythonic way.

        Returns:
            None.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_hparam(self, attribute: str) -> Any:
        """Looks up the saved hyper-parameter for the io mixed class.

        Args:
            attribute: The element name to look up within the saved init settings for self
        Returns:
            Value
        Raises:
            KeyError if the attribute does not exist in the saved init settings.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_hparams(self) -> Dict[str, Any]:
        """Returns the hyper-parameters of init in a dictionary format.

        Returns:
            Dict[str, Any]: A dictionary of the init hyper-parameters on this object.
        """
        raise NotImplementedError()


class IOMixinWithGettersSetters(WillHaveGetSetHparam, io.IOMixin):
    """An implementation of WillHaveGetSetHparam which makes use of the io.IOMixin.__io__ added to your classes.

    This enables you to mutate the hyper-parameters of your classes which will later be saved in configs.
    """

    def set_hparam(self, attribute: str, value: Any, also_change_value: bool = True) -> None:
        """Mutates the saved hyper-parameter for the io mixed class.

        If you would like to only change the saved hyper-param
            for example in the case of loading a dataclass where the same variables are mutated to other non-savable
            entities by deterministic rules after init, then use `also_change_value=False` to only update the
            hyper-parameter.

        Args:
            attribute: The element name to modify within the saved init settings for self
            value: New parameter for the saved init settings
            also_change_value: If you also want to mutate the attribute of this same name in self to be the desired
                value, set this to True, otherwise if the init arg and self arg are expected to be divergent, then
                do not set this and modify the self attribute separately in the normal pythonic way.

        Returns:
            None.
        """
        # Change the attribute of self and also change the io tracker so it gets updated in the config
        if also_change_value:
            setattr(self, attribute, value)
        setattr(self.__io__, attribute, value)

    def get_hparam(self, attribute: str) -> Any:
        """Looks up the saved hyper-parameter for the io mixed class.

        Args:
            attribute: The element name to look up within the saved init settings for self
        Returns:
            Value
        Raises:
            KeyError if the attribute does not exist in the saved init settings.
        """
        if attribute not in dir(self.__io__):
            raise KeyError(
                f"Attribute '{attribute}' not found in hyper-parameters. Options: {sorted(self.get_hparams().keys())}"
            )
        return getattr(self.__io__, attribute)

    def get_non_default_hparams(self) -> List[str]:
        """Returns a list of hyper-parameters that have been changed from their default values.

        Returns:
            List[str]: A list of hyper-parameters that have been changed from their default values.
        """
        return [k for k in self.__io__.__dict__["__argument_history__"].keys() if k != "__fn_or_cls__"]

    def get_hparams(self) -> Dict[str, Any]:
        """Returns the hyper-parameters of init in a dictionary format.

        Returns:
            Dict[str, Any]: A dictionary of the init hyper-parameters on this object.
        """
        return {k: getattr(self.__io__, k) for k in self.get_non_default_hparams()}
