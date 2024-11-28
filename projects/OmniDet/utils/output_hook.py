from typing import Optional, Sequence

from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class OutputHook(Hook):
    def __init__(self,
                 save_dir: Optional[str] = None,
    ) -> None:
        self.save_dir = save_dir
    
    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[dict]) -> None:
        # print('#######')
        super().after_val_iter(runner, batch_idx, data_batch, outputs)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[dict]) -> None:
        # print('#######')
        super().after_test_iter(runner, batch_idx, data_batch, outputs)
