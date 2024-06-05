from typing import Any
import pytorch_lightning as pl
from .base import ImageLoggerBase, ImageLoggerBaseDVS


class ImageLoggerEpochDVS(ImageLoggerBaseDVS):

    def __init__(self, img_dir, epoch_freq=1, state='vq') -> None:
        # state choice: [vq,transformer]
        super().__init__(img_dir=img_dir, freq=epoch_freq, state=state)

    def on_train_epoch_end(self, trainer: pl.Trainer,
                           pl_module: pl.LightningModule) -> None:
        freq_step = (trainer.current_epoch + 1)
        if self.check_freq(freq_step=freq_step):
            loader = trainer.train_dataloader
            batch = next(iter(loader))
            if self.state == 'vq':
                self.log_imgs_vq(batch=batch,
                                 trainer=trainer,
                                 pl_module=pl_module,
                                 flag='train')
            elif self.state == 'transformer':

                self.log_imgs_transformer(batch=batch,
                                          trainer=trainer,
                                          pl_module=pl_module,
                                          flag='train')
            else:
                raise RuntimeError("Invalid 'state' in ImageLogger2Epoch")

    def on_validation_end(self, trainer: pl.Trainer,
                          pl_module: pl.LightningModule) -> None:
        loader = trainer.val_dataloaders
        batch = next(iter(loader))

        self.log_imgs(batch=batch,
                      trainer=trainer,
                      pl_module=pl_module,
                      flag='val')


class ImageLoggerEpoch(ImageLoggerBase):

    def __init__(self, img_dir, epoch_freq=1, state='vq') -> None:
        # state choice: [vq,transformer]
        super().__init__(img_dir=img_dir, freq=epoch_freq, state=state)

    def on_train_epoch_end(self, trainer: pl.Trainer,
                           pl_module: pl.LightningModule) -> None:
        freq_step = (trainer.current_epoch + 1)
        if self.check_freq(freq_step=freq_step):
            loader = trainer.train_dataloader
            batch = next(iter(loader))
            if self.state == 'vq':
                self.log_imgs_vq(batch=batch,
                                 trainer=trainer,
                                 pl_module=pl_module,
                                 flag='train')
            elif self.state == 'transformer':

                self.log_imgs_transformer(batch=batch,
                                          trainer=trainer,
                                          pl_module=pl_module,
                                          flag='train')
            else:
                raise RuntimeError("Invalid 'state' in ImageLogger2Epoch")

    def on_validation_end(self, trainer: pl.Trainer,
                          pl_module: pl.LightningModule) -> None:
        loader = trainer.val_dataloaders
        batch = next(iter(loader))

        self.log_imgs(batch=batch,
                      trainer=trainer,
                      pl_module=pl_module,
                      flag='val')


class ImageLoggerBatch(ImageLoggerBase):

    def __init__(self, img_dir, batch_freq=100, state='vq') -> None:
        # state choice: [vq,transformer]
        super().__init__(img_dir=img_dir, freq=batch_freq, state=state)
        self.global_batch = 0

    def on_train_batch_end(self, trainer: pl.Trainer,
                           pl_module: pl.LightningModule, outputs, batch: Any,
                           batch_idx: int) -> None:
        self.global_batch += 1
        freq_step = self.global_batch
        if self.check_freq(freq_step=freq_step):
            if self.state == 'vq':
                self.log_imgs_vq(batch=batch,
                                 trainer=trainer,
                                 pl_module=pl_module,
                                 flag='train')
            elif self.state == 'transformer':

                self.log_imgs_transformer(batch=batch,
                                          trainer=trainer,
                                          pl_module=pl_module,
                                          flag='train')
            else:
                raise RuntimeError("Invalid 'state' in ImageLogger2Epoch")
