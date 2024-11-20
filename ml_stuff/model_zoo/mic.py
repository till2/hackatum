import timm
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import warnings

import kornia
import numpy as np
from einops import repeat, rearrange
from torch import nn, Tensor

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def strong_transform(param, data):
    data = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data)
    data = gaussian_blur(blur=param['blur'], data=data)
    return data


def denorm(img, mean, std):
    return img.mul(std).add(mean)


def renorm(img, mean, std):
    return img.sub(mean).div(std)


def color_jitter(color_jitter, mean, std, data, s=.25, p=.2):
    # s is the strength of colorjitter
    if color_jitter > p:
        mean = torch.as_tensor(mean, device=data.device)
        mean = repeat(mean, 'C -> B C 1 1', B=data.shape[0], C=3)
        std = torch.as_tensor(std, device=data.device)
        std = repeat(std, 'C -> B C 1 1', B=data.shape[0], C=3)
        if isinstance(s, dict):
            seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
        else:
            seq = nn.Sequential(
                kornia.augmentation.ColorJitter(
                    brightness=s, contrast=s, saturation=s, hue=s))
        data = denorm(data, mean, std)
        data = seq(data)
        data = renorm(data, mean, std)
    return data


def gaussian_blur(blur, data):
    if blur > 0.5:
        sigma = np.random.uniform(0.15, 1.15)
        kernel_size_y = int(
            np.floor(
                np.ceil(0.1 * data.shape[2]) - 0.5 +
                np.ceil(0.1 * data.shape[2]) % 2))
        kernel_size_x = int(
            np.floor(
                np.ceil(0.1 * data.shape[3]) - 0.5 +
                np.ceil(0.1 * data.shape[3]) % 2))
        kernel_size = (kernel_size_y, kernel_size_x)
        seq = nn.Sequential(
            kornia.filters.GaussianBlur2d(
                kernel_size=kernel_size, sigma=(sigma, sigma)))
        data = seq(data)
    return data

class Masking(nn.Module):
    def __init__(self, block_size, ratio, color_jitter_s, color_jitter_p, blur, mean, std):
        super(Masking, self).__init__()

        self.block_size = block_size
        self.ratio = ratio

        self.augmentation_params = None
        if (color_jitter_p > 0 and color_jitter_s > 0) or blur:
            print('[Masking] Use color augmentation.')
            self.augmentation_params = {
                'color_jitter': random.uniform(0, 1),
                'color_jitter_s': color_jitter_s,
                'color_jitter_p': color_jitter_p,
                'blur': random.uniform(0, 1) if blur else 0,
                'mean': mean,
                'std': std
            }

    @torch.no_grad()
    def forward(self, img: Tensor):
        img = img.clone()
        B, _, H, W = img.shape

        if self.augmentation_params is not None:
            img = strong_transform(self.augmentation_params, data=img.clone())

        mshape = B, 1, round(H / self.block_size), round(W / self.block_size)
        input_mask = torch.rand(mshape, device=img.device)
        input_mask = (input_mask > self.ratio).float()
        input_mask = resize(input_mask, size=(H, W))
        masked_img = img * input_mask

        return masked_img


class MIC(nn.Module):

    def __init__(self, masking_conf, backbone="resnext", img_channels=3, img_size=448, pseudo_threshold=0.8, alpha=0.99):
        super().__init__()

        backbones = ["resnext", "vit"]
        self.pseudo_threshold = pseudo_threshold
        self.alpha = alpha
        self.img_size = img_size

        assert backbone in backbones, f"backbone must be one of: {backbones}."
        if backbone == "resnext":

            self.student = timm.create_model("resnext101_64x4d", pretrained=True)

            self.student.conv1 = nn.Conv2d(
                in_channels=img_channels,
                out_channels=64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
            self.student.fc = nn.Linear(self.student.fc.in_features, img_size * 2)

            self.teacher = copy.deepcopy(self.student)

        elif backbone == "vit":
            raise NotImplementedError("CS")
        self.masking = Masking(**masking_conf)


    def get_pseudo_label_and_weight(self, pseudo_prob):
        pseudo_prob = pseudo_prob.detach()
        pseudo_label = (pseudo_prob > 0.5).float()
        if self.pseudo_threshold is not None:
            ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long()
            ps_size = np.size(np.array(pseudo_label.cpu()))
            pseudo_weight = torch.sum(ps_large_p).item() / ps_size
            pseudo_weight = pseudo_weight * torch.ones(
                pseudo_prob.shape, device=pseudo_prob.device)
        else:
            pseudo_weight = torch.ones(pseudo_prob.shape, device=pseudo_prob.device)
        return pseudo_label, pseudo_weight


    def forward_teacher(self, x):
        pseudo_prob = F.sigmoid(self.teacher(x))

        pseudo_label, pseudo_weight = self.get_pseudo_label_and_weight(pseudo_prob)
        del pseudo_prob

        # TODO: UNDERSTAND/ADD Region of interest filtering 

        return pseudo_label, pseudo_weight


    def forward_student(self, x):
        return self.student(x)


    def get_loss_supervised(self, x, label, loss_fn, pred_with_teacher=False, ):
        if pred_with_teacher:
            y = self.teacher(x)
        else:
            y = self.forward_student(x)

        start_logits, end_logits = rearrange(y, "b (c w) -> c b w", w=self.img_size, c=2)
        start_targets, end_targets = label
        return loss_fn(start_logits, end_logits, start_targets, end_targets).mean()

    def loss_mic(self, x, loss_fn):
        x_masked = self.masking(x)
        y =  self.student(x_masked)
        pseudo_label, pseudo_weight = self.forward_teacher(x)

        start_logits, end_logits = rearrange(y, "b (c w) -> c b w", w=self.img_size, c=2)
        pseudo_start_targets, pseudo_end_targets = rearrange(pseudo_label, "b (c w) -> c b w", w=self.img_size, c=2)

        return (pseudo_weight * loss_fn(start_logits, end_logits, pseudo_start_targets, pseudo_end_targets)).mean()

    def loss_adaptation(self, x, loss_fn):
        y =  self.student(x)
        pseudo_label, pseudo_weight = self.forward_teacher(x)
        
        start_logits, end_logits = rearrange(y, "b (c w) -> c b w", w=self.img_size, c=2)
        pseudo_start_targets, pseudo_end_targets = rearrange(pseudo_label, "b (c w) -> c b w", w=self.img_size, c=2)

        return (pseudo_weight * loss_fn(start_logits, end_logits, pseudo_start_targets, pseudo_end_targets)).mean()


    def _init_ema_weights(self):
        for param in self.teacher.parameters():
            param.detach_()
        mp = list(self.student.parameters())
        mcp = list(self.teacher.parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.teacher.parameters(),
                                    self.student.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def update_weights(self, iter):
        # Init/update ema model
        if iter == 0:
            self._init_ema_weights()

        if iter > 0:
            self._update_ema(iter)
