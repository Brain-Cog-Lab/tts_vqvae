from typing import Any
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback


class StopBatch(Callback):

    def __init__(self, stop_batch=200_000) -> None:
        super().__init__()
        self.global_step = 0
        self.stop_batch = stop_batch

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                           outputs, batch: Any, batch_idx: int) -> None:
        self.global_step += 1
        if self.global_step >= self.stop_batch:
            trainer.should_stop = True
