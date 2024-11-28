
from typing import Optional, Sequence

from mmengine.runner import Runner

from mmdet3d.structures import Det3DDataSample
from mmdet3d.engine.hooks import Det3DVisualizationHook
from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class OmniDet3DVisualizationHook(Det3DVisualizationHook):
    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[Det3DDataSample]) -> None:
        for data_sample in outputs:
            print('#######')


        super().after_test_iter(runner, batch_idx, data_batch, outputs)