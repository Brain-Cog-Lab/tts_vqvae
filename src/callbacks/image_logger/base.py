import torch
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
import torchvision
import os
from pytorch_lightning.utilities import rank_zero_only
from PIL import Image
import numpy as np
from torchvision.utils import save_image
from spikingjelly.datasets import play_frame
from src.utils.dvs import save_frame_dvs


class ImageLoggerBaseDVS(Callback):

    def __init__(self, img_dir, freq=1, state='vq') -> None:
        super().__init__()
        self.freq = freq
        self.img_dir = img_dir
        self.train_dir, self.val_dir = self.mk_dirs()
        self.state = state

    def mk_dirs(self):
        train_dir = os.path.join(self.img_dir, 'train')
        val_dir = os.path.join(self.img_dir, 'val')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        return train_dir, val_dir

    def check_freq(self, freq_step):
        if freq_step % self.freq == 0:
            return True
        else:
            return False

    @rank_zero_only
    def log_imgs_vq(self,
                    batch,
                    trainer: pl.Trainer,
                    pl_module: pl.LightningModule,
                    flag='train'):

        if pl_module.training:
            pl_module.eval()
        if flag == 'train':
            root = self.train_dir
        else:
            root = self.val_dir

        path = os.path.join(
            root,
            str(trainer.current_epoch) + '-' + str(trainer.global_step))
        assert hasattr(pl_module,
                       'log_images'), "model did not implement log_images"
        with torch.no_grad():
            log = pl_module.log_images(batch)
        inputs = log['inputs']
        recs = log['reconstructions']  # (b,t,c,h,w)

        torch.clamp(inputs, min=-1, max=1)
        torch.clamp(recs, min=-1, max=1)
        inputs = (inputs + 1) * 0.5
        recs = (recs + 1) * 0.5
        save_frame_dvs(inputs[0], save_gif_to=path + '_input.gif')
        save_frame_dvs(recs[0], save_gif_to=path + '_rec.gif')
        # play_frame(inputs[0], save_gif_to=path + '_input.gif')
        # play_frame(recs[0], save_gif_to=path + '_rec.gif')
        '''input_pad = torch.zeros(inputs.shape[1], 1, inputs.shape[3],
                                inputs.shape[4]).to(inputs.device)
        recs_pad = torch.zeros(recs.shape[1], 1, recs.shape[3],
                               recs.shape[4]).to(recs.device)
        save_image(torch.cat([input_pad, inputs[0]], dim=1),
                   fp=path + '_input.png')
        save_image(torch.cat([recs_pad, recs[0]], dim=1), fp=path + '_rec.png')'''

        if pl_module.training:
            pl_module.train()

    @rank_zero_only
    def log_imgs_transformer(self,
                             batch,
                             trainer: pl.Trainer,
                             pl_module: pl.LightningModule,
                             flag='train'):

        if pl_module.training:
            pl_module.eval()
        if flag == 'train':
            root = self.train_dir
        else:
            root = self.val_dir

        path = os.path.join(
            root,
            str(trainer.current_epoch) + '-' + str(trainer.global_step))
        assert hasattr(pl_module,
                       'log_images'), "model did not implement log_images"
        with torch.no_grad():
            log = pl_module.log_images(batch)

        # inputs = log['inputs']
        recs = log['reconstructions']
        x_sample = log["samples_half"]
        x_sample_nopix = log["samples_nopix"]  # (b,t,c,h,w)
        # x_sample_det = log["samples_det"]

        recs = (recs + 1) * 0.5
        x_sample = (x_sample + 1) * 0.5
        x_sample_nopix = (x_sample_nopix + 1) * 0.5

        save_frame_dvs(recs[0], save_gif_to=path + '_rec.gif')
        save_frame_dvs(x_sample[0], save_gif_to=path + '_half.gif')
        save_frame_dvs(x_sample_nopix[0], save_gif_to=path + '_nopix.gif')

        if pl_module.training:
            pl_module.train()


class ImageLoggerBase(Callback):

    def __init__(self, img_dir, freq=1, state='vq') -> None:
        # state choice: [vq,transformer]
        super().__init__()
        self.freq = freq
        self.img_dir = img_dir
        self.train_dir, self.val_dir = self.mk_dirs()
        self.state = state

    def mk_dirs(self):
        train_dir = os.path.join(self.img_dir, 'train')
        val_dir = os.path.join(self.img_dir, 'val')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        return train_dir, val_dir

    def check_freq(self, freq_step):
        if freq_step % self.freq == 0:
            return True
        else:
            return False

    @rank_zero_only
    def log_imgs_vq(self,
                    batch,
                    trainer: pl.Trainer,
                    pl_module: pl.LightningModule,
                    flag='train'):

        if pl_module.training:
            pl_module.eval()
        if flag == 'train':
            root = self.train_dir
        else:
            root = self.val_dir

        path = os.path.join(
            root,
            str(trainer.current_epoch) + '-' + str(trainer.global_step))
        assert hasattr(pl_module,
                       'log_images'), "model did not implement log_images"
        with torch.no_grad():
            log = pl_module.log_images(batch)
        inputs = log['inputs']
        recs = log['reconstructions']
        nrow = min(inputs.shape[0], 4)

        torch.clamp(inputs, min=-1, max=1)
        torch.clamp(recs, min=-1, max=1)
        inputs = 0.5 * (inputs + 1)
        recs = 0.5 * (recs + 1)
        save_image(inputs[:nrow], fp=path + '_input.png', nrow=nrow)
        save_image(recs[:nrow], fp=path + '_rec.png', nrow=nrow)
        if pl_module.training:
            pl_module.train()

    @rank_zero_only
    def log_imgs_transformer(self,
                             batch,
                             trainer: pl.Trainer,
                             pl_module: pl.LightningModule,
                             flag='train'):

        if pl_module.training:
            pl_module.eval()
        if flag == 'train':
            root = self.train_dir
        else:
            root = self.val_dir

        path = os.path.join(
            root,
            str(trainer.current_epoch) + '-' + str(trainer.global_step))
        assert hasattr(pl_module,
                       'log_images'), "model did not implement log_images"
        with torch.no_grad():
            log = pl_module.log_images(batch)

        # inputs = log['inputs']
        recs = log['reconstructions']
        x_sample = log["samples_half"]
        x_sample_nopix = log["samples_nopix"]
        # x_sample_det = log["samples_det"]
        nrow = min(recs.shape[0], 4)

        # torch.clamp(inputs, min=-1, max=1)
        torch.clamp(recs, min=-1, max=1)
        torch.clamp(x_sample, min=-1, max=1)
        torch.clamp(x_sample_nopix, min=-1, max=1)
        # torch.clamp(x_sample_det, min=-1, max=1)

        # inputs = 0.5 * (inputs + 1)
        recs = 0.5 * (recs + 1)
        x_sample = 0.5 * (x_sample + 1)
        x_sample_nopix = 0.5 * (x_sample_nopix + 1)
        # x_sample_det = 0.5 * (x_sample_det + 1)

        # save_image(inputs[:nrow], fp=path + '_input.png', nrow=nrow)
        save_image(recs[:nrow], fp=path + '_rec.png', nrow=nrow)
        save_image(x_sample[:nrow], fp=path + '_half.png', nrow=nrow)
        save_image(x_sample_nopix[:nrow], fp=path + '_nopix.png', nrow=nrow)
        # save_image(x_sample_det[:nrow], fp=path + '_det.png', nrow=nrow)

        if pl_module.training:
            pl_module.train()


class ImageLogger(Callback):
    """
    original implementation
    """

    def __init__(self,
                 batch_frequency,
                 max_images,
                 clamp=True,
                 increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.WandbLogger: self._wandb,
            pl.loggers.CSVLogger: self._csv,
        }
        self.log_steps = [
            2**n for n in range(int(np.log2(self.batch_freq)) + 1)
        ]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        raise ValueError("No way wandb")
        grids = dict()
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grids[f"{split}/{k}"] = wandb.Image(grid)
        pl_module.logger.experiment.log(grids)

    @rank_zero_only
    def _csv(self, pl_module, images, batch_idx, split):
        pass
        '''for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)'''

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch,
                  batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)

            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx)
                and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch,
                                              split=split,
                                              pl_module=pl_module)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch,
                           batch_idx)

            logger_log_images = self.logger_log_images.get(
                logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch,
                           batch_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                batch_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")
