import pytorch_lightning as pl
import torch
from src.utils.snn_decoder import SnnDecoder
from src.utils.snn_encoder import SnnEncoder
from src.utils.auto_instance import instantiate_from_config
from src.modules.sencoder.base_encoder import EncoderSnn
from src.modules.sdecoder.base_decoder import DecoderSnn
from src.modules.svqvae.quantizer_snn import VectorQuantizerSnn
from torch.nn import functional as F


class SVQModel(pl.LightningModule):

    def __init__(
        self,
        ddconfig,
        lossconfig,
        n_embed,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        remap=None,
        sane_index_shape=False,  # tell vector quantizer to return indices as bhw
        time_step=3,
        snn_encoder='direct',
        snn_decoder='mean',
    ):
        super().__init__()

        # snn encoder and decoder
        self.time_step = time_step
        self.snn_encoder = SnnEncoder(method=snn_encoder, time_step=time_step)
        self.snn_decoder = SnnDecoder(method=snn_decoder)

        self.image_key = image_key
        self.encoder = EncoderSnn(**ddconfig)
        self.decoder = DecoderSnn(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizerSnn(n_embed,
                                           embed_dim,
                                           beta=0.25,
                                           remap=remap,
                                           sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim,
                                               ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize",
                                 torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.automatic_optimization = False

    def reset(self):
        for m in self.modules():
            if hasattr(m, 'n_reset'):
                m.n_reset()

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x_t):
        h_t = self.encoder(x_t)
        t, b = h_t.shape[0], h_t.shape[1]

        # flatten time step and batch
        h_t = self.quant_conv(torch.flatten(h_t, start_dim=0, end_dim=1))

        # recover time dim
        t_shape = [t, b] + list(h_t.shape[1:])
        h_t = h_t.reshape(t_shape)

        quant_t, emb_loss_t, info_t = self.quantize(h_t)
        return quant_t, emb_loss_t, info_t

    def decode(self, quant_t):

        t, b = quant_t.shape[0], quant_t.shape[1]

        # flatten time step and batch
        quant_t = self.post_quant_conv(
            torch.flatten(quant_t, start_dim=0, end_dim=1))

        # recover time dim
        t_shape = [t, b] + list(quant_t.shape[1:])

        quant_t = quant_t.reshape(t_shape)
        dec_t = self.decoder(quant_t)
        return dec_t

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input_t):
        quant_t, diff_t, _ = self.encode(input_t)
        dec_t = self.decode(quant_t)
        return dec_t, diff_t

    def get_input(self, batch, key):
        if isinstance(batch, dict):
            return batch[key]
        else:
            return batch[0]

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)

        # encoder x to x_t, has time dim
        x_t = self.snn_encoder(x)  # (t,b,c,h,w)

        xrec_t, qloss_t = self(x_t)

        # we need to eliminate t for backward
        xrec = self.snn_decoder(xrec_t)
        qloss = torch.mean(qloss_t, dim=0)

        opt_ae, opt_dis = self.optimizers()

        # autoencode
        aeloss, log_dict_ae = self.loss(qloss,
                                        x,
                                        xrec,
                                        0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="train")

        self.log("train/aeloss",
                 aeloss,
                 prog_bar=True,
                 logger=True,
                 on_step=True,
                 on_epoch=True)
        self.log_dict(log_dict_ae,
                      prog_bar=False,
                      logger=True,
                      on_step=True,
                      on_epoch=True)

        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        opt_ae.step()

        # discriminator
        discloss, log_dict_disc = self.loss(qloss,
                                            x,
                                            xrec,
                                            1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="train")
        self.log("train/discloss",
                 discloss,
                 prog_bar=True,
                 logger=True,
                 on_step=True,
                 on_epoch=True)
        self.log_dict(log_dict_disc,
                      prog_bar=False,
                      logger=True,
                      on_step=True,
                      on_epoch=True)

        opt_dis.zero_grad()
        self.manual_backward(discloss)
        opt_dis.step()

        # reset snn membrane potential
        self.reset()

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)

        # encoder x to x_t, has time dim
        x_t = self.snn_encoder(x)  # (t,b,c,h,w)

        xrec_t, qloss_t = self(x_t)

        # we need to eliminate t for backward
        xrec = self.snn_decoder(xrec_t)
        qloss = torch.mean(qloss_t, dim=0)

        aeloss, log_dict_ae = self.loss(qloss,
                                        x,
                                        xrec,
                                        0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val")

        discloss, log_dict_disc = self.loss(qloss,
                                            x,
                                            xrec,
                                            1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss",
                 rec_loss,
                 prog_bar=True,
                 logger=True,
                 on_step=True,
                 on_epoch=True,
                 sync_dist=True)
        self.log("val/aeloss",
                 aeloss,
                 prog_bar=True,
                 logger=True,
                 on_step=True,
                 on_epoch=True,
                 sync_dist=True)
        # self.log_dict(log_dict_ae)
        # self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quantize.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr,
                                  betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr,
                                    betas=(0.5, 0.9))
        return opt_ae, opt_disc

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        x_t = self.snn_encoder(x)
        xrec_t, _ = self(x_t)
        xrec = self.snn_decoder(xrec_t)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize",
                                 torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class SVQModelEpoch(pl.LightningModule):

    def __init__(
        self,
        ddconfig,
        lossconfig,
        n_embed,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        remap=None,
        sane_index_shape=False,  # tell vector quantizer to return indices as bhw
        time_step=3,
        snn_encoder='direct',
        snn_decoder='mean',
    ):
        """
        discriminator loss starts via epoch, rather global_step
        """
        super().__init__()

        # snn encoder and decoder
        self.time_step = time_step
        self.snn_encoder = SnnEncoder(method=snn_encoder, time_step=time_step)
        self.snn_decoder = SnnDecoder(method=snn_decoder)

        self.image_key = image_key
        self.encoder = EncoderSnn(**ddconfig)
        self.decoder = DecoderSnn(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizerSnn(n_embed,
                                           embed_dim,
                                           beta=0.25,
                                           remap=remap,
                                           sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim,
                                               ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize",
                                 torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.automatic_optimization = False

    def reset(self):
        for m in self.modules():
            if hasattr(m, 'n_reset'):
                m.n_reset()

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x_t):
        h_t = self.encoder(x_t)
        t, b = h_t.shape[0], h_t.shape[1]

        # flatten time step and batch
        h_t = self.quant_conv(torch.flatten(h_t, start_dim=0, end_dim=1))

        # recover time dim
        t_shape = [t, b] + list(h_t.shape[1:])
        h_t = h_t.reshape(t_shape)

        quant_t, emb_loss_t, info_t = self.quantize(h_t)
        return quant_t, emb_loss_t, info_t

    def decode(self, quant_t):

        t, b = quant_t.shape[0], quant_t.shape[1]

        # flatten time step and batch
        quant_t = self.post_quant_conv(
            torch.flatten(quant_t, start_dim=0, end_dim=1))

        # recover time dim
        t_shape = [t, b] + list(quant_t.shape[1:])

        quant_t = quant_t.reshape(t_shape)
        dec_t = self.decoder(quant_t)
        return dec_t

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input_t):
        quant_t, diff_t, _ = self.encode(input_t)
        dec_t = self.decode(quant_t)
        return dec_t, diff_t

    def get_input(self, batch, key):
        if isinstance(batch, dict):
            return batch[key]
        else:
            return batch[0]

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)

        # encoder x to x_t, has time dim
        x_t = self.snn_encoder(x)  # (t,b,c,h,w)

        xrec_t, qloss_t = self(x_t)

        # we need to eliminate t for backward
        xrec = self.snn_decoder(xrec_t)
        qloss = torch.mean(qloss_t, dim=0)

        opt_ae, opt_dis = self.optimizers()

        # autoencode
        aeloss, log_dict_ae = self.loss(qloss,
                                        x,
                                        xrec,
                                        0,
                                        self.current_epoch,
                                        last_layer=self.get_last_layer(),
                                        split="train")

        self.log("train/aeloss",
                 aeloss,
                 prog_bar=True,
                 logger=True,
                 on_step=True,
                 on_epoch=True)
        self.log_dict(log_dict_ae,
                      prog_bar=False,
                      logger=True,
                      on_step=True,
                      on_epoch=True)

        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        opt_ae.step()

        # discriminator
        discloss, log_dict_disc = self.loss(qloss,
                                            x,
                                            xrec,
                                            1,
                                            self.current_epoch,
                                            last_layer=self.get_last_layer(),
                                            split="train")
        self.log("train/discloss",
                 discloss,
                 prog_bar=True,
                 logger=True,
                 on_step=True,
                 on_epoch=True)
        self.log_dict(log_dict_disc,
                      prog_bar=False,
                      logger=True,
                      on_step=True,
                      on_epoch=True)

        opt_dis.zero_grad()
        self.manual_backward(discloss)
        opt_dis.step()

        # reset snn membrane potential
        self.reset()

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)

        # encoder x to x_t, has time dim
        x_t = self.snn_encoder(x)  # (t,b,c,h,w)

        xrec_t, qloss_t = self(x_t)

        # we need to eliminate t for backward
        xrec = self.snn_decoder(xrec_t)
        qloss = torch.mean(qloss_t, dim=0)

        aeloss, log_dict_ae = self.loss(qloss,
                                        x,
                                        xrec,
                                        0,
                                        self.current_epoch,
                                        last_layer=self.get_last_layer(),
                                        split="val")

        discloss, log_dict_disc = self.loss(qloss,
                                            x,
                                            xrec,
                                            1,
                                            self.current_epoch,
                                            last_layer=self.get_last_layer(),
                                            split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss",
                 rec_loss,
                 prog_bar=True,
                 logger=True,
                 on_step=True,
                 on_epoch=True,
                 sync_dist=True)
        self.log("val/aeloss",
                 aeloss,
                 prog_bar=True,
                 logger=True,
                 on_step=True,
                 on_epoch=True,
                 sync_dist=True)
        # self.log_dict(log_dict_ae)
        # self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quantize.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr,
                                  betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr,
                                    betas=(0.5, 0.9))
        return opt_ae, opt_disc

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        x_t = self.snn_encoder(x)
        xrec_t, _ = self(x_t)
        xrec = self.snn_decoder(xrec_t)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize",
                                 torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class SVQModelEpochDVS(pl.LightningModule):

    def __init__(
        self,
        ddconfig,
        lossconfig,
        n_embed,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        remap=None,
        sane_index_shape=False,  # tell vector quantizer to return indices as bhw
        time_step=3,
        snn_encoder='direct',
        snn_decoder='mean',
    ):
        """
        discriminator loss starts via epoch, rather global_step
        """
        super().__init__()

        # snn encoder and decoder
        self.time_step = time_step
        self.snn_encoder = SnnEncoder(method=snn_encoder, time_step=time_step)
        self.snn_decoder = SnnDecoder(method=snn_decoder)

        self.image_key = image_key
        self.encoder = EncoderSnn(**ddconfig)
        self.decoder = DecoderSnn(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizerSnn(n_embed,
                                           embed_dim,
                                           beta=0.25,
                                           remap=remap,
                                           sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim,
                                               ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if monitor is not None:
            self.monitor = monitor

        self.automatic_optimization = False

    def reset(self):
        for m in self.modules():
            if hasattr(m, 'n_reset'):
                m.n_reset()

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x_t):
        h_t = self.encoder(x_t)
        t, b = h_t.shape[0], h_t.shape[1]

        # flatten time step and batch
        h_t = self.quant_conv(torch.flatten(h_t, start_dim=0, end_dim=1))

        # recover time dim
        t_shape = [t, b] + list(h_t.shape[1:])
        h_t = h_t.reshape(t_shape)

        quant_t, emb_loss_t, info_t = self.quantize(h_t)
        return quant_t, emb_loss_t, info_t

    def decode(self, quant_t):

        t, b = quant_t.shape[0], quant_t.shape[1]

        # flatten time step and batch
        quant_t = self.post_quant_conv(
            torch.flatten(quant_t, start_dim=0, end_dim=1))

        # recover time dim
        t_shape = [t, b] + list(quant_t.shape[1:])

        quant_t = quant_t.reshape(t_shape)
        dec_t = self.decoder(quant_t)
        return dec_t

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input_t):
        quant_t, diff_t, _ = self.encode(input_t)
        dec_t = self.decode(quant_t)
        return dec_t, diff_t

    def get_input(self, batch, key):
        if isinstance(batch, dict):
            return batch[key]
        else:
            return batch[0]

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)

        # encoder x to x_t, has time dim
        x_t = self.snn_encoder(x)  # (t,b,c,h,w)

        xrec_t, qloss_t = self(x_t)

        # xrec_t.shape = (t,b,c,h,w)
        qloss = torch.mean(qloss_t, dim=0)

        opt_ae, opt_dis = self.optimizers()

        # autoencode
        aeloss = []
        for t in range(x_t.shape[0]):
            aeloss_t, log_dict_ae_t = self.loss(
                qloss,
                x_t[t],
                xrec_t[t],
                0,
                self.current_epoch,
                last_layer=self.get_last_layer(),
                split="train")
            aeloss.append(aeloss_t)

        aeloss = sum(aeloss) / len(aeloss)

        self.log("train/aeloss",
                 aeloss,
                 prog_bar=True,
                 logger=True,
                 on_step=True,
                 on_epoch=True)

        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        opt_ae.step()

        discloss = []
        # discriminator
        for t in range(x_t.shape[0]):
            discloss_t, log_dict_disc_t = self.loss(
                qloss,
                x_t[t],
                xrec_t[t],
                1,
                self.current_epoch,
                last_layer=self.get_last_layer(),
                split="train")
            discloss.append(discloss_t)

        discloss = sum(discloss) / len(discloss)

        self.log("train/discloss",
                 discloss,
                 prog_bar=True,
                 logger=True,
                 on_step=True,
                 on_epoch=True)

        opt_dis.zero_grad()
        self.manual_backward(discloss)
        opt_dis.step()

        # reset snn membrane potential
        self.reset()

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)

        # encoder x to x_t, has time dim
        x_t = self.snn_encoder(x)  # (t,b,c,h,w)

        xrec_t, qloss_t = self(x_t)

        # xrec_t.shape = (t,b,c,h,w)
        qloss = torch.mean(qloss_t, dim=0)

        aeloss = []
        for t in range(x_t.shape[0]):
            aeloss_t, log_dict_ae_t = self.loss(
                qloss,
                x_t[t],
                xrec_t[t],
                0,
                self.current_epoch,
                last_layer=self.get_last_layer(),
                split="val")
            aeloss.append(aeloss_t)

        aeloss = torch.tensor(aeloss).to(x_t.device)
        aeloss = torch.mean(aeloss)

        discloss = []
        for t in range(x_t.shape[0]):
            discloss_t, log_dict_disc_t = self.loss(
                qloss,
                x_t[t],
                xrec_t[t],
                1,
                self.current_epoch,
                last_layer=self.get_last_layer(),
                split="val")
            discloss.append(discloss_t)
        discloss = torch.tensor(discloss).to(x_t.device)
        discloss = torch.mean(discloss)

        self.log("val/aeloss",
                 aeloss,
                 prog_bar=True,
                 logger=True,
                 on_step=True,
                 on_epoch=True,
                 sync_dist=True)
        # self.log_dict(log_dict_ae)
        # self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quantize.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr,
                                  betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr,
                                    betas=(0.5, 0.9))
        return opt_ae, opt_disc

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        x_t = self.snn_encoder(x)
        xrec_t, _ = self(x_t)
        xrec = self.snn_decoder(xrec_t)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log
