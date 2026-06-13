from abc import abstractmethod
from typing import Any

from molmo_spaces.policy.base_policy import InferencePolicy


class StatefulPolicy(InferencePolicy):
    @abstractmethod
    def get_state(self) -> Any:
        ...

    @abstractmethod
    def set_state(self, state: Any):
        ...
