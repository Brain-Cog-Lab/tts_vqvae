import os, math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from src.utils.auto_instance import instantiate_from_config
from src.modules.util import SOSProvider
import random
import copy
from braincog.base.node import LIFNode


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class Net2NetTransformer(pl.LightningModule):

    def __init__(
        self,
        transformer_config,
        first_stage_config,
        cond_stage_config,
        permuter_config=None,
        ckpt_path=None,
        ignore_keys=[],
        first_stage_key="image",
        cond_stage_key="depth",
        downsample_cond_size=-1,
        pkeep=1.0,
        sos_token=0,
        unconditional=False,
    ):
        super().__init__()
        self.be_unconditional = unconditional
        self.sos_token = sos_token
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.init_first_stage_from_ckpt(first_stage_config)
        self.init_cond_stage_from_ckpt(cond_stage_config)
        if permuter_config is None:
            permuter_config = {
                "target": "src.modules.transformer.permuter.Identity"
            }
        self.permuter = instantiate_from_config(config=permuter_config)
        self.transformer = instantiate_from_config(config=transformer_config)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.downsample_cond_size = downsample_cond_size
        self.pkeep = pkeep

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

    def init_cond_stage_from_ckpt(self, config):
        if config == "__is_first_stage__":
            print("Using first stage also as cond stage.")
            self.cond_stage_model = self.first_stage_model
        elif config == "__is_unconditional__" or self.be_unconditional:
            print(
                f"Using no cond stage. Assuming the training is intended to be unconditional. "
                f"Prepending {self.sos_token} as a sos token.")
            self.be_unconditional = True
            self.cond_stage_key = self.first_stage_key
            self.cond_stage_model = SOSProvider(self.sos_token)
        else:
            model = instantiate_from_config(config)
            model = model.eval()
            model.train = disabled_train
            self.cond_stage_model = model

    def forward(self, x, c):
        # one step to produce the logits
        _, z_indices = self.encode_to_z_snn(x)
        _, c_indices = self.encode_to_c(c)

        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(
                self.pkeep *
                torch.ones(z_indices.shape, device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices,
                                           self.transformer.config.vocab_size)
            a_indices = mask * z_indices + (1 - mask) * r_indices
        else:
            a_indices = z_indices

        cz_indices = torch.cat((c_indices, a_indices), dim=1)

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices
        # make the prediction
        logits, _ = self.transformer(cz_indices[:, :-1])
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        logits = logits[:, c_indices.shape[1] - 1:]

        # print(logits)

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def sample(self,
               x,
               c,
               steps,
               temperature=1.0,
               sample=False,
               top_k=None,
               callback=lambda k: None):
        x = torch.cat((c, x), dim=1)
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        if self.pkeep <= 0.0:
            # one pass suffices since input is pure noise anyway
            assert len(x.shape) == 2
            noise_shape = (x.shape[0], steps - 1)
            # noise = torch.randint(self.transformer.config.vocab_size, noise_shape).to(x)
            noise = c.clone()[:, x.shape[1] - c.shape[1]:-1]
            x = torch.cat((x, noise), dim=1)
            logits, _ = self.transformer(x)
            # take all logits for now and scale by temp
            logits = logits / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                shape = probs.shape
                probs = probs.reshape(shape[0] * shape[1], shape[2])
                ix = torch.multinomial(probs, num_samples=1)
                probs = probs.reshape(shape[0], shape[1], shape[2])
                ix = ix.reshape(shape[0], shape[1])
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # cut off conditioning
            x = ix[:, c.shape[1] - 1:]
        else:
            for k in range(steps):
                callback(k)
                assert x.size(
                    1) <= block_size  # make sure model can see conditioning
                x_cond = x if x.size(
                    1
                ) <= block_size else x[:,
                                       -block_size:]  # crop context if needed
                logits, _ = self.transformer(x_cond)
                # pluck the logits at the final step and scale by temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop probabilities to only the top k options
                if top_k is not None:
                    logits = self.top_k_logits(logits, top_k)
                # apply softmax to convert to probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution or take the most likely
                if sample:
                    ix = torch.multinomial(probs, num_samples=1)
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                # append to the sequence and continue
                x = torch.cat((x, ix), dim=1)
            # cut off conditioning
            x = x[:, c.shape[1]:]
        return x

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        indices = self.permuter(indices)
        return quant_z, indices

    @torch.no_grad()
    def encode_to_z_snn(self, x):
        x_t = self.first_stage_model.snn_encoder(x)
        quant_z, _, info = self.first_stage_model.encode(x_t)
        indices = info[2].view(info[2].shape[0], -1)
        indices = self.permuter(indices)
        self.first_stage_model.reset()
        return quant_z, indices

    @torch.no_grad()
    def encode_to_c(self, c):
        if self.downsample_cond_size > -1:
            c = F.interpolate(c,
                              size=(self.downsample_cond_size,
                                    self.downsample_cond_size))
        quant_c, _, [_, _, indices] = self.cond_stage_model.encode(c)
        if len(indices.shape) > 2:
            indices = indices.view(c.shape[0], -1)
        return quant_c, indices

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        index = self.permuter(index, reverse=True)
        bhwc = (zshape[0], zshape[2], zshape[3], zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x

    @torch.no_grad()
    def decode_to_img_te(self, index, ztshape):
        # index.shape = (bs, hw)
        # ztshape = (t,b,c,h,w)
        index = self.permuter(index, reverse=True)
        h, w = ztshape[-2], ztshape[-1]
        quant_z_t = self.first_stage_model.quantize.get_codebook_entry_te(
            index, h=h, w=w)
        x_t = self.first_stage_model.decode(quant_z_t)
        x = self.first_stage_model.snn_decoder(x_t)
        self.first_stage_model.reset()
        return x

    @torch.no_grad()
    def decode_to_img_te_2(self, index, h, w):
        # index.shape = (bs, hw)
        # decode_to_img_te that need not use ztshape, only need for h & w of indices
        index = self.permuter(index, reverse=True)
        quant_z_t = self.first_stage_model.quantize.get_codebook_entry_te(
            index, h=h, w=w)
        x_t = self.first_stage_model.decode(quant_z_t)
        x = self.first_stage_model.snn_decoder(x_t)
        self.first_stage_model.reset()
        return x

    def decode_to_img_te_destruction(self, index, h, w, method='none'):
        """
        destroy temporal coding in temporal dim
        """
        index = self.permuter(index, reverse=True)
        quant_z_t = self.first_stage_model.quantize.get_codebook_entry_te(
            index, h=h, w=w)

        T = quant_z_t.shape[0]
        if method == "none":
            quant_z_t_destruction = quant_z_t
        elif method == 'reverse':
            indices = torch.arange(T - 1, -1, -1,
                                   dtype=torch.int64).to('cuda:0')
            # print(indices)
            quant_z_t_destruction = torch.index_select(input=quant_z_t,
                                                       dim=0,
                                                       index=indices)
        elif method == 'repeat':
            indices = torch.tensor([T - 1] * T).to('cuda:0')
            quant_z_t_destruction = torch.index_select(input=quant_z_t,
                                                       dim=0,
                                                       index=indices)
        elif method == "empty":
            empty = torch.zeros([1] + list(quant_z_t.shape[1:])).to('cuda:0')
            quant_z_t_destruction = torch.concat([empty, quant_z_t[1:]])

        x_t = self.first_stage_model.decode(quant_z_t_destruction)
        x = self.first_stage_model.snn_decoder(x_t)
        self.first_stage_model.reset()
        return x

    def decode_to_img_te_repeat(self, index, h, w, num=1):
        """
        repeat destroy temporal coding in temporal dim
        num is the number of time step to replace
        """
        assert num >= 1
        index = self.permuter(index, reverse=True)
        quant_z_t = self.first_stage_model.quantize.get_codebook_entry_te(
            index, h=h, w=w)  # (t,b,c,h,w)

        T = quant_z_t.shape[0]
        assert num <= T
        last_idx = T - 1
        indices = list(range(T - num)) + [last_idx] * num
        indices = torch.tensor(indices, dtype=torch.int64).to(quant_z_t.device)
        quant_z_t_destruction = torch.index_select(input=quant_z_t,
                                                   dim=0,
                                                   index=indices)

        x_t = self.first_stage_model.decode(quant_z_t_destruction)
        x = self.first_stage_model.snn_decoder(x_t)
        self.first_stage_model.reset()
        return x

    @torch.no_grad()
    def log_images(self,
                   batch,
                   temperature=None,
                   top_k=None,
                   callback=None,
                   lr_interface=False,
                   **kwargs):
        log = dict()

        N = 4
        if lr_interface:
            x, c = self.get_xc(batch, N, diffuse=False, upsample_factor=8)
        else:
            x, c = self.get_xc(batch, N)
        x = x.to(device=self.device)
        c = c.to(device=self.device)

        quant_z, z_indices = self.encode_to_z_snn(x)
        quant_c, c_indices = self.encode_to_c(c)

        # create a "half"" sample
        z_start_indices = z_indices[:, :z_indices.shape[1] // 2]
        index_sample = self.sample(
            z_start_indices,
            c_indices,
            steps=z_indices.shape[1] - z_start_indices.shape[1],
            temperature=temperature if temperature is not None else 1.0,
            sample=True,
            top_k=top_k if top_k is not None else 100,
            callback=callback if callback is not None else lambda k: None)
        x_sample = self.decode_to_img_te(index_sample, quant_z.shape)

        # sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(
            z_start_indices,
            c_indices,
            steps=z_indices.shape[1],
            temperature=temperature if temperature is not None else 1.0,
            sample=True,
            top_k=top_k if top_k is not None else 100,
            callback=callback if callback is not None else lambda k: None)
        x_sample_nopix = self.decode_to_img_te(index_sample, quant_z.shape)

        # det sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(
            z_start_indices,
            c_indices,
            steps=z_indices.shape[1],
            sample=False,
            callback=callback if callback is not None else lambda k: None)
        x_sample_det = self.decode_to_img_te(index_sample, quant_z.shape)

        # reconstruction
        x_rec = self.decode_to_img_te(z_indices, quant_z.shape)

        log["inputs"] = x
        log["reconstructions"] = x_rec

        if self.cond_stage_key in ["objects_bbox", "objects_center_points"]:
            figure_size = (x_rec.shape[2], x_rec.shape[3])
            dataset = kwargs["pl_module"].trainer.datamodule.datasets[
                "validation"]
            label_for_category_no = dataset.get_textual_label_for_category_no
            plotter = dataset.conditional_builders[self.cond_stage_key].plot
            log["conditioning"] = torch.zeros_like(log["reconstructions"])
            for i in range(quant_c.shape[0]):
                log["conditioning"][i] = plotter(quant_c[i],
                                                 label_for_category_no,
                                                 figure_size)
            log["conditioning_rec"] = log["conditioning"]
        elif self.cond_stage_key != "image":
            cond_rec = self.cond_stage_model.decode(quant_c)
            if self.cond_stage_key == "segmentation":
                # get image from segmentation mask
                num_classes = cond_rec.shape[1]

                c = torch.argmax(c, dim=1, keepdim=True)
                c = F.one_hot(c, num_classes=num_classes)
                c = c.squeeze(1).permute(0, 3, 1, 2).float()
                c = self.cond_stage_model.to_rgb(c)

                cond_rec = torch.argmax(cond_rec, dim=1, keepdim=True)
                cond_rec = F.one_hot(cond_rec, num_classes=num_classes)
                cond_rec = cond_rec.squeeze(1).permute(0, 3, 1, 2).float()
                cond_rec = self.cond_stage_model.to_rgb(cond_rec)
            log["conditioning_rec"] = cond_rec
            log["conditioning"] = c

        log["samples_half"] = x_sample
        log["samples_nopix"] = x_sample_nopix
        log["samples_det"] = x_sample_det
        return log

    def get_input(self, key, batch):
        x = batch[key]
        '''if len(x.shape) == 3:
            x = x[..., None]
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        if x.dtype == torch.double:
            x = x.float()'''
        return x

    def get_xc(self, batch, N=None):
        x = self.get_input(self.first_stage_key, batch)
        c = self.get_input(self.cond_stage_key, batch)
        if N is not None:
            x = x[:N]
            c = c[:N]
        return x, c

    def shared_step(self, batch, batch_idx):
        x, c = self.get_xc(batch)
        logits, target = self(x, c)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               target.reshape(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train/loss",
                 loss,
                 prog_bar=True,
                 logger=True,
                 on_step=True,
                 on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val/loss",
                 loss,
                 prog_bar=True,
                 logger=True,
                 on_step=True,
                 on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            if isinstance(m, LIFNode):
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                    no_decay.add(fpn)
            else:
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(
                            m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(
                            m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(
            inter_params
        ) == 0, "parameters %s made it into both decay/no_decay sets!" % (
            str(inter_params), )
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": 0.01
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0
            },
        ]
        optimizer = torch.optim.AdamW(optim_groups,
                                      lr=self.learning_rate,
                                      betas=(0.9, 0.95))
        return optimizer


class Net2NetTransformerNonTE(pl.LightningModule):
    """
    Net2Net transformer wrapper for no temporal embedding first stage model
    """

    def __init__(
        self,
        transformer_config,
        first_stage_config,
        cond_stage_config,
        permuter_config=None,
        ckpt_path=None,
        ignore_keys=[],
        first_stage_key="image",
        cond_stage_key="depth",
        downsample_cond_size=-1,
        pkeep=1.0,
        sos_token=0,
        unconditional=False,
    ):
        super().__init__()
        self.be_unconditional = unconditional
        self.sos_token = sos_token
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.init_first_stage_from_ckpt(first_stage_config)
        self.init_cond_stage_from_ckpt(cond_stage_config)
        if permuter_config is None:
            permuter_config = {
                "target": "src.modules.transformer.permuter.Identity"
            }
        self.permuter = instantiate_from_config(config=permuter_config)
        self.transformer = instantiate_from_config(config=transformer_config)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.downsample_cond_size = downsample_cond_size
        self.pkeep = pkeep

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

    def init_cond_stage_from_ckpt(self, config):
        if config == "__is_first_stage__":
            print("Using first stage also as cond stage.")
            self.cond_stage_model = self.first_stage_model
        elif config == "__is_unconditional__" or self.be_unconditional:
            print(
                f"Using no cond stage. Assuming the training is intended to be unconditional. "
                f"Prepending {self.sos_token} as a sos token.")
            self.be_unconditional = True
            self.cond_stage_key = self.first_stage_key
            self.cond_stage_model = SOSProvider(self.sos_token)
        else:
            model = instantiate_from_config(config)
            model = model.eval()
            model.train = disabled_train
            self.cond_stage_model = model

    def forward(self, x, c):
        # one step to produce the logits
        _, z_indices_t = self.encode_to_z_snn_nonte(
            x)  # z_indices_t.shape = (t,b,...)
        _, c_indices = self.encode_to_c(c)
        c_indices_t = c_indices.unsqueeze(
            0)  # add time dim: (b,...) => (1,b,...)
        c_indices_t = c_indices_t.repeat(z_indices_t.shape[0], 1,
                                         1)  # (T,b,...)
        '''if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(
                self.pkeep *
                torch.ones(z_indices.shape, device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices,
                                           self.transformer.config.vocab_size)
            a_indices = mask * z_indices + (1 - mask) * r_indices
        else:
            a_indices = z_indices'''

        a_indices_t = z_indices_t

        cz_indices_t = torch.cat((c_indices_t, a_indices_t), dim=2)

        # select random time step for training gpt
        T = cz_indices_t.shape[0]
        t_select = random.randint(0, T - 1)
        cz_indices_select = cz_indices_t[t_select]

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices_t[t_select]
        # make the prediction
        logits, _ = self.transformer(cz_indices_select[:, :-1])
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        logits = logits[:, c_indices.shape[1] - 1:]

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def sample(self,
               x,
               c,
               steps,
               temperature=1.0,
               sample=False,
               top_k=None,
               callback=lambda k: None):
        x = torch.cat((c, x), dim=1)
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        x_t = []
        if self.pkeep <= 0.0:
            # one pass suffices since input is pure noise anyway
            assert len(x.shape) == 2
            noise_shape = (x.shape[0], steps - 1)
            # noise = torch.randint(self.transformer.config.vocab_size, noise_shape).to(x)
            noise = c.clone()[:, x.shape[1] - c.shape[1]:-1]
            x = torch.cat((x, noise), dim=1)
            logits, _ = self.transformer(x)
            # take all logits for now and scale by temp
            logits = logits / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                shape = probs.shape
                probs = probs.reshape(shape[0] * shape[1], shape[2])
                ix = torch.multinomial(probs, num_samples=1)
                probs = probs.reshape(shape[0], shape[1], shape[2])
                ix = ix.reshape(shape[0], shape[1])
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # cut off conditioning
            x = ix[:, c.shape[1] - 1:]
        else:
            for t in range(self.first_stage_model.time_step):
                x_temp = copy.deepcopy(x)
                for k in range(steps):
                    callback(k)
                    assert x_temp.size(
                        1
                    ) <= block_size  # make sure model can see conditioning
                    x_cond = x_temp if x_temp.size(
                        1
                    ) <= block_size else x_temp[:,
                                                -block_size:]  # crop context if needed
                    logits, _ = self.transformer(x_cond)
                    # pluck the logits at the final step and scale by temperature
                    logits = logits[:, -1, :] / temperature
                    # optionally crop probabilities to only the top k options
                    if top_k is not None:
                        logits = self.top_k_logits(logits, top_k)
                    # apply softmax to convert to probabilities
                    probs = F.softmax(logits, dim=-1)
                    # sample from the distribution or take the most likely
                    if sample:
                        ix = torch.multinomial(probs, num_samples=1)
                    else:
                        _, ix = torch.topk(probs, k=1, dim=-1)
                    # append to the sequence and continue
                    x_temp = torch.cat((x_temp, ix), dim=1)
                # cut off conditioning
                x_temp = x_temp[:, c.shape[1]:]
                x_t.append(x_temp)

        x_t = torch.stack(x_t, dim=0)
        return x_t

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        indices = self.permuter(indices)
        return quant_z, indices

    @torch.no_grad()
    def encode_to_z_snn_nonte(self, x):
        """
        no temporal embedding (te) version.
        """
        x_t = self.first_stage_model.snn_encoder(x)
        quant_z_t, _, info_t = self.first_stage_model.encode(x_t)
        bs = quant_z_t.shape[1]
        indices_t = [self.permuter(info[2].view(bs, -1)) for info in info_t]
        indices_t = torch.stack(indices_t, dim=0)
        # indices = info[2].view(info[2].shape[0], -1)
        # indices = self.permuter(indices)
        self.first_stage_model.reset()
        return quant_z_t, indices_t

    @torch.no_grad()
    def encode_to_c(self, c):
        if self.downsample_cond_size > -1:
            c = F.interpolate(c,
                              size=(self.downsample_cond_size,
                                    self.downsample_cond_size))
        quant_c, _, [_, _, indices] = self.cond_stage_model.encode(c)
        if len(indices.shape) > 2:
            indices = indices.view(c.shape[0], -1)
        return quant_c, indices

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        index = self.permuter(index, reverse=True)
        bhwc = (zshape[0], zshape[2], zshape[3], zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x

    '''@torch.no_grad()
    def decode_to_img_te(self, index, ztshape):
        # index.shape = (bs, hw)
        # ztshape = (t,b,c,h,w)
        index = self.permuter(index, reverse=True)
        h, w = ztshape[-2], ztshape[-1]
        quant_z_t = self.first_stage_model.quantize.get_codebook_entry_te(
            index, h=h, w=w)
        x_t = self.first_stage_model.decode(quant_z_t)
        x = self.first_stage_model.snn_decoder(x_t)
        self.first_stage_model.reset()
        return x

    @torch.no_grad()
    def decode_to_img_te_2(self, index, h, w):
        # index.shape = (bs, hw)
        # decode_to_img_te that need not use ztshape, only need for h & w of indices
        index = self.permuter(index, reverse=True)
        quant_z_t = self.first_stage_model.quantize.get_codebook_entry_te(
            index, h=h, w=w)
        x_t = self.first_stage_model.decode(quant_z_t)
        x = self.first_stage_model.snn_decoder(x_t)
        self.first_stage_model.reset()
        return x'''

    def decode_to_img_nonte(self, index_t, h, w):
        index_t = self.permuter(index_t, reverse=True)
        quant_z_t = self.first_stage_model.quantize.get_codebook_entry_nonte(
            index_t, h=h, w=w)
        x_t = self.first_stage_model.decode(quant_z_t)
        x = self.first_stage_model.snn_decoder(x_t)
        self.first_stage_model.reset()
        return x

    @torch.no_grad()
    def log_images(self,
                   batch,
                   temperature=None,
                   top_k=None,
                   callback=None,
                   lr_interface=False,
                   **kwargs):
        log = dict()

        N = 4
        if lr_interface:
            x, c = self.get_xc(batch, N, diffuse=False, upsample_factor=8)
        else:
            x, c = self.get_xc(batch, N)
        x = x.to(device=self.device)
        c = c.to(device=self.device)

        quant_z_t, z_indices_t = self.encode_to_z_snn_nonte(x)
        z_indices = z_indices_t[0]
        quant_c, c_indices = self.encode_to_c(c)

        # create a "half"" sample
        z_start_indices = z_indices[:, :z_indices.shape[1] // 2]
        index_sample_t = self.sample(
            z_start_indices,
            c_indices,
            steps=z_indices.shape[1] - z_start_indices.shape[1],
            temperature=temperature if temperature is not None else 1.0,
            sample=True,
            top_k=top_k if top_k is not None else 100,
            callback=callback if callback is not None else lambda k: None)
        x_sample = self.decode_to_img_nonte(index_sample_t,
                                            h=quant_z_t.shape[-2],
                                            w=quant_z_t.shape[-1])

        # sample
        z_start_indices = z_indices[:, :0]
        index_sample_t = self.sample(
            z_start_indices,
            c_indices,
            steps=z_indices.shape[1],
            temperature=temperature if temperature is not None else 1.0,
            sample=True,
            top_k=top_k if top_k is not None else 100,
            callback=callback if callback is not None else lambda k: None)
        x_sample_nopix = self.decode_to_img_nonte(index_sample_t,
                                                  h=quant_z_t.shape[-2],
                                                  w=quant_z_t.shape[-1])

        # det sample
        z_start_indices = z_indices[:, :0]
        index_sample_t = self.sample(
            z_start_indices,
            c_indices,
            steps=z_indices.shape[1],
            sample=False,
            callback=callback if callback is not None else lambda k: None)
        x_sample_det = self.decode_to_img_nonte(index_sample_t,
                                                h=quant_z_t.shape[-2],
                                                w=quant_z_t.shape[-1])

        # reconstruction
        x_rec = self.decode_to_img_nonte(z_indices_t,
                                         h=quant_z_t.shape[-2],
                                         w=quant_z_t.shape[-1])

        log["inputs"] = x
        log["reconstructions"] = x_rec

        if self.cond_stage_key in ["objects_bbox", "objects_center_points"]:
            figure_size = (x_rec.shape[2], x_rec.shape[3])
            dataset = kwargs["pl_module"].trainer.datamodule.datasets[
                "validation"]
            label_for_category_no = dataset.get_textual_label_for_category_no
            plotter = dataset.conditional_builders[self.cond_stage_key].plot
            log["conditioning"] = torch.zeros_like(log["reconstructions"])
            for i in range(quant_c.shape[0]):
                log["conditioning"][i] = plotter(quant_c[i],
                                                 label_for_category_no,
                                                 figure_size)
            log["conditioning_rec"] = log["conditioning"]
        elif self.cond_stage_key != "image":
            cond_rec = self.cond_stage_model.decode(quant_c)
            if self.cond_stage_key == "segmentation":
                # get image from segmentation mask
                num_classes = cond_rec.shape[1]

                c = torch.argmax(c, dim=1, keepdim=True)
                c = F.one_hot(c, num_classes=num_classes)
                c = c.squeeze(1).permute(0, 3, 1, 2).float()
                c = self.cond_stage_model.to_rgb(c)

                cond_rec = torch.argmax(cond_rec, dim=1, keepdim=True)
                cond_rec = F.one_hot(cond_rec, num_classes=num_classes)
                cond_rec = cond_rec.squeeze(1).permute(0, 3, 1, 2).float()
                cond_rec = self.cond_stage_model.to_rgb(cond_rec)
            log["conditioning_rec"] = cond_rec
            log["conditioning"] = c

        log["samples_half"] = x_sample
        log["samples_nopix"] = x_sample_nopix
        log["samples_det"] = x_sample_det
        return log

    def get_input(self, key, batch):
        x = batch[key]
        '''if len(x.shape) == 3:
            x = x[..., None]
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        if x.dtype == torch.double:
            x = x.float()'''
        return x

    def get_xc(self, batch, N=None):
        x = self.get_input(self.first_stage_key, batch)
        c = self.get_input(self.cond_stage_key, batch)
        if N is not None:
            x = x[:N]
            c = c[:N]
        return x, c

    def shared_step(self, batch, batch_idx):
        x, c = self.get_xc(batch)
        logits, target = self(x, c)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               target.reshape(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train/loss",
                 loss,
                 prog_bar=True,
                 logger=True,
                 on_step=True,
                 on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val/loss",
                 loss,
                 prog_bar=True,
                 logger=True,
                 on_step=True,
                 on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(
                        m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(
                        m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(
            inter_params
        ) == 0, "parameters %s made it into both decay/no_decay sets!" % (
            str(inter_params), )
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": 0.01
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0
            },
        ]
        optimizer = torch.optim.AdamW(optim_groups,
                                      lr=self.learning_rate,
                                      betas=(0.9, 0.95))
        return optimizer
