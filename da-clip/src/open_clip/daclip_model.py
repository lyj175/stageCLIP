from typing import Optional

import logging
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import copy


from .transformer import (
    ControlTransformer
)
from .model import CLIP, CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower


class DaCLIP(nn.Module):
    def __init__(self, clip_model: CLIP):
        super().__init__()
        self.clip = clip_model
        self.visual = clip_model.visual
        self.visual_control = copy.deepcopy(clip_model.visual)
        self.visual_control.transformer = ControlTransformer(self.visual_control.transformer)
        self.logit_scale = copy.deepcopy(clip_model.logit_scale)

    def initial_controller(self):
        for (kv, param_v), (kc, param_c) in zip(self.clip.visual.named_parameters(), self.visual_control.named_parameters()):
            if 'transformer' not in kv:
                param_c.data.copy_(param_v.data)

        for param_v, param_c in zip(self.clip.visual.transformer.parameters(), self.visual_control.transformer.parameters()):
            param_c.data.copy_(param_v.data)

        self.logit_scale.data.copy_(self.clip.logit_scale.data)
        
    def lock_clip(self):
        for param in self.clip.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.clip.visual.set_grad_checkpointing(enable)
        self.clip.transformer.grad_checkpointing = enable
        self.visual_control.set_grad_checkpointing(enable)

    def encode_image(self, image, control=False, normalize: bool = False):
        # if control:
        #     degra_features, hiddens = self.visual_control(image, output_hiddens=True)
        #     image_features = self.clip.visual(image, control=hiddens)
        #
        #     image_features = F.normalize(image_features, dim=-1) if normalize else image_features
        #     degra_features = F.normalize(degra_features, dim=-1) if normalize else degra_features
        #     return image_features, degra_features
        # else:
        #     return self.clip.encode_image(image, normalize)

        if image is None:
            return None, None

        if control:
            # degra_features, hiddens = self.visual_control(image, output_hiddens=True)
            # image_features = self.clip.visual(image, control=hiddens)
            #
            # image_features = F.normalize(image_features, dim=-1) if normalize else image_features
            # degra_features = F.normalize(degra_features, dim=-1) if normalize else degra_features
            # return image_features, degra_features

            image_features_samples = []
            degra_features_samples = []
            for i in range(image.shape[0]):
                degra_features, hiddens = self.visual_control(image[i], output_hiddens=True)
                image_features = self.clip.visual(image[i], control=hiddens)

                image_features = F.normalize(image_features, dim=-1) if normalize else image_features
                degra_features = F.normalize(degra_features, dim=-1) if normalize else degra_features
                image_features_samples.append(image_features)
                degra_features_samples.append(degra_features)
            return torch.stack(image_features_samples,dim=0), torch.stack(degra_features_samples,dim=0)
        else:
            return self.clip.encode_image(image, normalize)

    def encode_text(self, text, normalize: bool = False):
        return self.clip.encode_text(text, normalize)

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
            adjust_region = None
    ):
        # (caption, degradation) = text.chunk(2, dim=-1) if text is not None else (None, None)
        ad_r = []
        if adjust_region is not None and adjust_region[0] is not None:
            target = adjust_region[0].unsqueeze(1)
            target = self.encode_text(target, normalize=True)
            far_from = adjust_region[1].unsqueeze(1)
            far_from = self.encode_text(far_from, normalize=True)
            ad_r.append(target)
            ad_r.append(far_from)

        caption = text
        image_features, image_degra_features = self.encode_image(image, control=True, normalize=True) if image is not None else None
        text_features = self.encode_text(caption, normalize=True) if text is not None else None

        # text_degra_features = self.encode_text(degradation, normalize=True) if degradation is not None else None

        #TODO clip冻结visual/controller调参Visual/
        # return {
        #     "image_features": image_features,#clip冻结visual
        #     "text_features": text_features,#caption text
        #     "image_degra_features": image_degra_features,#controller调参Visual
        #     "text_degra_features": text_degra_features,#ded type text
        #     "logit_scale": self.logit_scale.exp()
        # }
        return {
            "image_features": image_features,  # clip冻结visual controlled
            "text_features": text_features,  # caption text
            "image_degra_features": image_degra_features,  # controller调参Visual
            "logit_scale": self.logit_scale.exp(),
            "adjust_region": ad_r,
        }


