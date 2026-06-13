"""Local stub of openpi.policies.rlbench_policy.

The upstream openpi/training/config.py was modified to `import` this module and
reference `RLBenchInputs` / `RLBenchOutputs`, but the real file was never added.
This stub provides minimal class shells so module import of
`openpi.training.config` succeeds. The classes here are only constructed when
the RLBench-specific data factory is used (config.py:410-413); for any other
data-factory (e.g. MlSpacesDatasetConfigFactory used by molmobot training),
these are never instantiated.

If the real rlbench_policy.py is delivered later, just replace this file.
"""

import dataclasses

from openpi import transforms
from openpi.models import model as _model


@dataclasses.dataclass(frozen=True)
class RLBenchInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        raise NotImplementedError(
            "RLBenchInputs is a stub. If you see this, the real "
            "openpi.policies.rlbench_policy module needs to be added."
        )


@dataclasses.dataclass(frozen=True)
class RLBenchOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        raise NotImplementedError(
            "RLBenchOutputs is a stub. If you see this, the real "
            "openpi.policies.rlbench_policy module needs to be added."
        )
