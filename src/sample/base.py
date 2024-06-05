import torch
from einops import repeat
import time
from omegaconf import OmegaConf
from src.utils.auto_instance import instantiate_from_config
from src.modules.transformer.mingpt import sample_with_past
from tqdm import tqdm
import os
from torchvision.utils import save_image
from src.utils.dvs import save_frame_dvs


class BaseSampler:

    def __init__(self,
                 gen_config_path,
                 gen_ckpt_path,
                 batch_size,
                 code_h,
                 code_w,
                 verbose_time=False):
        gen_config = OmegaConf.load(gen_config_path)
        self.model = instantiate_from_config(gen_config.model)
        if gen_ckpt_path is not None:
            ckpt = torch.load(gen_ckpt_path, map_location='cpu')
            self.model.load_state_dict(ckpt['state_dict'])
        self.model = self.model.to("cuda:0")
        self.model.eval()
        self.batch_size = batch_size
        self.code_h = code_h
        self.code_w = code_w
        self.verbose_time = verbose_time
        self.steps = self.code_w * self.code_h

    @staticmethod
    @torch.no_grad()
    def sample_unconditional_te_static_batch(model,
                                             batch_size,
                                             steps=256,
                                             temperature=None,
                                             top_k=None,
                                             top_p=None,
                                             callback=None,
                                             h=16,
                                             w=16,
                                             verbose_time=False):
        log = dict()
        assert model.be_unconditional, 'Expecting an unconditional model.'
        c_indices = repeat(torch.tensor([model.sos_token]),
                           '1 -> b 1',
                           b=batch_size).to(model.device)  # sos token
        t1 = time.time()
        index_sample = sample_with_past(c_indices,
                                        model.transformer,
                                        steps=steps,
                                        sample_logits=True,
                                        top_k=top_k,
                                        callback=callback,
                                        temperature=temperature,
                                        top_p=top_p)
        if verbose_time:
            sampling_time = time.time() - t1
            print(f"Full sampling takes about {sampling_time:.2f} seconds.")
        x_sample = model.decode_to_img_te_2(index_sample, h=h, w=w)
        log["samples"] = x_sample
        return log

    def sample_unconditional_te(self, num_samples, temperature=1.0):
        batches = [
            self.batch_size for _ in range(num_samples // self.batch_size)
        ] + [num_samples % self.batch_size]
        samples = []
        for bs in batches:
            if bs == 0: break
            logs = self.sample_unconditional_te_static_batch(
                model=self.model,
                batch_size=bs,
                steps=self.steps,
                temperature=temperature,
                h=self.code_h,
                w=self.code_w,
                verbose_time=self.verbose_time)
            imgs = logs["samples"]
            torch.clamp(imgs, min=-1, max=1)
            # imgs = 0.5 * (imgs + 1)
            samples.append(imgs)

        return torch.cat(samples, dim=0)  # (-1,1)

    def save_sample_unconditional_te(self,
                                     num_samples,
                                     save_dir,
                                     temperature=1.0):
        batches = [
            self.batch_size for _ in range(num_samples // self.batch_size)
        ] + [num_samples % self.batch_size]
        idx = 0
        for bs in batches:
            if bs == 0: break
            logs = self.sample_unconditional_te_static_batch(
                model=self.model,
                batch_size=bs,
                steps=self.steps,
                temperature=temperature,
                h=self.code_h,
                w=self.code_w,
                verbose_time=self.verbose_time)
            imgs = logs["samples"]
            torch.clamp(imgs, min=-1, max=1)
            imgs = 0.5 * (imgs + 1)
            for img in imgs:
                img_path = os.path.join(save_dir, f"sample_{idx}.png")
                save_image(img, img_path)
                print(f"save sample {idx}")
                idx += 1

    def save_sample_unconditional_te_dvs(self,
                                         num_samples,
                                         save_dir,
                                         temperature=1.0,
                                         time_unfold=False,
                                         unfold_dir=None):
        batches = [
            self.batch_size for _ in range(num_samples // self.batch_size)
        ] + [num_samples % self.batch_size]
        idx = 0
        if time_unfold:
            assert unfold_dir is not None, "no unfold dir"
        for bs in batches:
            if bs == 0: break
            logs = self.sample_unconditional_te_static_batch(
                model=self.model,
                batch_size=bs,
                steps=self.steps,
                temperature=temperature,
                h=self.code_h,
                w=self.code_w,
                verbose_time=self.verbose_time)
            imgs = logs["samples"]  # (b,t,2,h,w)
            img_pad = torch.zeros((imgs.shape[1], 1, imgs.shape[3],
                                   imgs.shape[4])).to(imgs.device)
            imgs = 0.5 * (imgs + 1)
            for img in imgs:
                gif_path = os.path.join(save_dir, f"sample_{idx}.gif")
                save_frame_dvs(img, save_gif_to=gif_path)
                print(f"save sample {idx}")

                if time_unfold:
                    img_path = os.path.join(unfold_dir, f"sample_{idx}.png")
                    save_image(torch.cat([img_pad, img], dim=1),
                               fp=img_path,
                               padding=5,
                               pad_value=0.9)
                    print(f"save unfold sample {idx}")

                idx += 1

    @staticmethod
    @torch.no_grad()
    def sample_unconditional_nonte_static_batch(model,
                                                batch_size,
                                                steps=256,
                                                temperature=None,
                                                top_k=None,
                                                top_p=None,
                                                callback=None,
                                                h=16,
                                                w=16,
                                                verbose_time=False):
        log = dict()
        assert model.be_unconditional, 'Expecting an unconditional model.'
        time_step = model.first_stage_model.time_step
        c_indices = repeat(torch.tensor([model.sos_token]),
                           '1 -> b 1',
                           b=batch_size).to(model.device)  # sos token
        t1 = time.time()
        index_sample_t = []
        for t in range(time_step):
            index_sample = sample_with_past(c_indices,
                                            model.transformer,
                                            steps=steps,
                                            sample_logits=True,
                                            top_k=top_k,
                                            callback=callback,
                                            temperature=temperature,
                                            top_p=top_p)
            index_sample_t.append(index_sample)
        if verbose_time:
            sampling_time = time.time() - t1
            print(f"Full sampling takes about {sampling_time:.2f} seconds.")
        index_sample_t = torch.stack(index_sample_t, dim=0)
        # print(index_sample_t.shape)
        x_sample = model.decode_to_img_nonte(index_sample_t, h=h, w=w)
        log["samples"] = x_sample
        return log

    def sample_unconditional_nonte(self, num_samples, temperature=1.0):
        batches = [
            self.batch_size for _ in range(num_samples // self.batch_size)
        ] + [num_samples % self.batch_size]
        samples = []
        for bs in batches:
            if bs == 0: break
            logs = self.sample_unconditional_nonte_static_batch(
                model=self.model,
                batch_size=bs,
                steps=self.steps,
                temperature=temperature,
                h=self.code_h,
                w=self.code_w,
                verbose_time=self.verbose_time)
            imgs = logs["samples"]
            torch.clamp(imgs, min=-1, max=1)
            # imgs = 0.5 * (imgs + 1)
            samples.append(imgs)

        return torch.cat(samples, dim=0)  # (-1,1)

    def save_sample_unconditional_nonte(self,
                                        num_samples,
                                        save_dir,
                                        temperature=1.0):
        batches = [
            self.batch_size for _ in range(num_samples // self.batch_size)
        ] + [num_samples % self.batch_size]
        idx = 0
        for bs in batches:
            if bs == 0: break
            logs = self.sample_unconditional_nonte_static_batch(
                model=self.model,
                batch_size=bs,
                steps=self.steps,
                temperature=temperature,
                h=self.code_h,
                w=self.code_w,
                verbose_time=self.verbose_time)
            imgs = logs["samples"]
            torch.clamp(imgs, min=-1, max=1)
            imgs = 0.5 * (imgs + 1)
            for img in imgs:
                img_path = os.path.join(save_dir, f"sample_{idx}.png")
                save_image(img, img_path)
                print(f"save sample {idx}")
                idx += 1

    def save_sample_unconditional_nonte_dvs(self,
                                            num_samples,
                                            save_dir,
                                            temperature=1.0,
                                            time_unfold=False,
                                            unfold_dir=None):
        batches = [
            self.batch_size for _ in range(num_samples // self.batch_size)
        ] + [num_samples % self.batch_size]
        idx = 0
        for bs in batches:
            if bs == 0: break
            logs = self.sample_unconditional_nonte_static_batch(
                model=self.model,
                batch_size=bs,
                steps=self.steps,
                temperature=temperature,
                h=self.code_h,
                w=self.code_w,
                verbose_time=self.verbose_time)
            imgs = logs["samples"]
            img_pad = torch.zeros(imgs.shape[1], 1, imgs.shape[3],
                                  imgs.shape[4]).to(imgs.device)
            imgs = 0.5 * (imgs + 1)
            for img in imgs:
                gif_path = os.path.join(save_dir, f"sample_{idx}.gif")
                save_frame_dvs(img, save_gif_to=gif_path)
                print(f"save sample {idx}")

                if time_unfold:
                    img_path = os.path.join(unfold_dir, f"sample_{idx}.png")
                    save_image(torch.cat([img_pad, img], dim=1),
                               fp=img_path,
                               padding=5,
                               pad_value=0.9)
                    print(f"save unfold sample {idx}")

                idx += 1

    @staticmethod
    @torch.no_grad()
    def sample_unconditional_te_static_batch_destruction(
            model,
            batch_size,
            steps=256,
            temperature=None,
            top_k=None,
            top_p=None,
            callback=None,
            h=16,
            w=16,
            verbose_time=False):
        log = dict()
        assert model.be_unconditional, 'Expecting an unconditional model.'
        c_indices = repeat(torch.tensor([model.sos_token]),
                           '1 -> b 1',
                           b=batch_size).to(model.device)  # sos token
        t1 = time.time()
        index_sample = sample_with_past(c_indices,
                                        model.transformer,
                                        steps=steps,
                                        sample_logits=True,
                                        top_k=top_k,
                                        callback=callback,
                                        temperature=temperature,
                                        top_p=top_p)
        if verbose_time:
            sampling_time = time.time() - t1
            print(f"Full sampling takes about {sampling_time:.2f} seconds.")
        x_sample = model.decode_to_img_te_destruction(index_sample,
                                                      h=h,
                                                      w=w,
                                                      method='none')
        x_sample_reverse = model.decode_to_img_te_destruction(index_sample,
                                                              h=h,
                                                              w=w,
                                                              method='reverse')
        x_sample_repeat = model.decode_to_img_te_destruction(index_sample,
                                                             h=h,
                                                             w=w,
                                                             method='repeat')
        x_sample_empty = model.decode_to_img_te_destruction(index_sample,
                                                            h=h,
                                                            w=w,
                                                            method='empty')
        log["none"] = x_sample
        log['reverse'] = x_sample_reverse
        log['repeat'] = x_sample_repeat
        log['empty'] = x_sample_empty
        return log

    def save_sample_unconditional_te_destruction(self,
                                                 num_samples,
                                                 save_dirs: dict,
                                                 temperature=1.0):
        batches = [
            self.batch_size for _ in range(num_samples // self.batch_size)
        ] + [num_samples % self.batch_size]

        idx = 0
        none_dir = save_dirs['none']
        reverse_dir = save_dirs['reverse']
        repeat_dir = save_dirs['repeat']
        empty_dir = save_dirs['empty']
        for bs in batches:
            if bs == 0: break
            logs = self.sample_unconditional_te_static_batch_destruction(
                model=self.model,
                batch_size=bs,
                steps=self.steps,
                temperature=temperature,
                h=self.code_h,
                w=self.code_w,
                verbose_time=self.verbose_time)

            sample_none = logs["none"]
            torch.clamp(sample_none, min=-1, max=1)
            sample_none = 0.5 * (sample_none + 1)

            sample_reverse = logs['reverse']
            torch.clamp(sample_reverse, min=-1, max=1)
            sample_reverse = 0.5 * (sample_reverse + 1)

            sample_repeat = logs['repeat']
            torch.clamp(sample_repeat, min=-1, max=1)
            sample_repeat = 0.5 * (sample_repeat + 1)

            sample_empty = logs['empty']
            torch.clamp(sample_empty, min=-1, max=1)
            sample_empty = 0.5 * (sample_empty + 1)

            for i in range(bs):
                none_path = os.path.join(none_dir, f"sample_{idx}.png")
                reverse_path = os.path.join(reverse_dir, f"sample_{idx}.png")
                repeat_path = os.path.join(repeat_dir, f"sample_{idx}.png")
                empty_path = os.path.join(empty_dir, f"sample_{idx}.png")

                save_image(sample_none[i], none_path)
                save_image(sample_reverse[i], reverse_path)
                save_image(sample_repeat[i], repeat_path)
                save_image(sample_empty[i], empty_path)

                print(f"Saved samples {idx}")

                idx += 1

    @staticmethod
    @torch.no_grad()
    def sample_unconditional_te_static_batch_repeat(model,
                                                    batch_size,
                                                    steps=256,
                                                    temperature=None,
                                                    top_k=None,
                                                    top_p=None,
                                                    callback=None,
                                                    h=16,
                                                    w=16,
                                                    verbose_time=False,
                                                    T=6):
        log = dict()
        assert model.be_unconditional, 'Expecting an unconditional model.'
        c_indices = repeat(torch.tensor([model.sos_token]),
                           '1 -> b 1',
                           b=batch_size).to(model.device)  # sos token
        t1 = time.time()
        index_sample = sample_with_past(c_indices,
                                        model.transformer,
                                        steps=steps,
                                        sample_logits=True,
                                        top_k=top_k,
                                        callback=callback,
                                        temperature=temperature,
                                        top_p=top_p)
        if verbose_time:
            sampling_time = time.time() - t1
            print(f"Full sampling takes about {sampling_time:.2f} seconds.")

        for t in range(T):
            x_t = model.decode_to_img_te_repeat(index_sample,
                                                h=h,
                                                w=w,
                                                num=t + 1)
            log[f"repeat_{t + 1}"] = x_t

        return log

    def save_sample_unconditional_te_repeat(self,
                                            num_samples,
                                            save_dirs,
                                            temperature=1.0,
                                            T=6):
        batches = [
            self.batch_size for _ in range(num_samples // self.batch_size)
        ] + [num_samples % self.batch_size]

        idx = 0
        for bs in batches:
            if bs == 0: break
            logs = self.sample_unconditional_te_static_batch_repeat(
                model=self.model,
                batch_size=bs,
                steps=self.steps,
                temperature=temperature,
                h=self.code_h,
                w=self.code_w,
                verbose_time=self.verbose_time,
                T=T)

            idx_temp = 0
            for t in range(T):
                idx_temp = idx
                x_t = logs[f"repeat_{t + 1}"]
                torch.clamp(x_t, min=-1, max=1)
                x_t = 0.5 * (x_t + 1)
                dir = save_dirs[f"repeat_{t + 1}"]
                for i in range(bs):
                    path = os.path.join(dir, f"sample_{idx_temp}.png")
                    save_image(x_t[i], path)
                    print(f"saved {path}")
                    idx_temp += 1

            idx = idx_temp
