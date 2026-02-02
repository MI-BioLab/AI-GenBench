"""
Adapted from openclip.py and resnet_mod.py
https://github.com/grip-unina/ClipBased-SyntheticImageDetection/blob/main/networks/openclipnet.py

Original license of the above files:
Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Tuple, TYPE_CHECKING
import torch
import torch.nn as nn

from algorithms.model_factory_registry import ModelFactoryRegistry

if TYPE_CHECKING:
    from open_clip import CLIP

dict_pretrain = {
    "clipL14openai": ("ViT-L-14", "openai"),
    "clipL14laion400m": ("ViT-L-14", "laion400m_e32"),
    "clipL14laion2B": ("ViT-L-14", "laion2b_s32b_b82k"),
    "clipL14datacomp": (
        "ViT-L-14",
        "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
        "open_clip_pytorch_model.bin",
    ),
    "clipL14commonpool": (
        "ViT-L-14",
        "laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K",
        "open_clip_pytorch_model.bin",
    ),
    "clipaL14datacomp": ("ViT-L-14-CLIPA", "datacomp1b"),
    "cocaL14laion2B": ("coca_ViT-L-14", "laion2b_s13b_b90k"),
    "clipg14laion2B": ("ViT-g-14", "laion2b_s34b_b88k"),
    "eva2L14merged2b": ("EVA02-L-14", "merged2b_s4b_b131k"),
    "clipB16laion2B": ("ViT-B-16", "laion2b_s34b_b88k"),
}


class OpenClipLinear(nn.Module):
    def __init__(
        self,
        num_classes=1,
        pretrain="clipL14commonpool",
        normalize=True,
        next_to_last=False,
    ):
        import open_clip

        super().__init__()

        if len(dict_pretrain[pretrain]) == 2:
            backbone = open_clip.create_model(
                dict_pretrain[pretrain][0], pretrained=dict_pretrain[pretrain][1]
            )
        else:
            from huggingface_hub import hf_hub_download

            backbone = open_clip.create_model(
                dict_pretrain[pretrain][0],
                pretrained=hf_hub_download(*dict_pretrain[pretrain][1:]),
            )

        if next_to_last:
            self.num_features = backbone.visual.proj.shape[0]
            backbone.visual.proj = None
        else:
            self.num_features = backbone.visual.output_dim

        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False
        self._bb: Tuple["CLIP"] = (backbone,)
        self.normalize = normalize

        self.fc = ChannelLinear(
            self.num_features, num_classes
        )  # Should be compatible with the rest of the framework
        torch.nn.init.normal_(self.fc.weight.data, 0.0, 0.02)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.backbone.to(*args, **kwargs)
        return self

    def forward_features(self, x) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone.encode_image(x, normalize=self.normalize)
        return features

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def forward(self, x: torch.Tensor, return_feature=False) -> torch.Tensor:
        features = self.forward_features(x)
        if return_feature:
            return features
        return self.forward_head(features)

    @property
    def backbone(self) -> "CLIP":
        return self._bb[0]


class OpenClipTune(nn.Module):
    def __init__(
        self,
        num_classes=1,
        pretrain="clipL14commonpool",
        normalize=True,
        next_to_last=False,
    ):
        import open_clip

        super().__init__()

        if len(dict_pretrain[pretrain]) == 2:
            backbone = open_clip.create_model(
                dict_pretrain[pretrain][0], pretrained=dict_pretrain[pretrain][1]
            )
        else:
            from huggingface_hub import hf_hub_download

            backbone = open_clip.create_model(
                dict_pretrain[pretrain][0],
                pretrained=hf_hub_download(*dict_pretrain[pretrain][1:]),
            )

        if next_to_last:
            self.num_features = backbone.visual.proj.shape[0]
            backbone.visual.proj = None
        else:
            self.num_features = backbone.visual.output_dim

        self.backbone: "CLIP" = backbone.visual
        self.normalize = normalize

        self.fc = ChannelLinear(
            self.num_features, num_classes
        )  # Should be compatible with the rest of the framework
        torch.nn.init.normal_(self.fc.weight.data, 0.0, 0.02)

    def forward_features(self, x) -> torch.Tensor:
        # features = self.visual(image)
        # return F.normalize(features, dim=-1) if normalize else features

        features = self.backbone(x)
        features = (
            torch.nn.functional.normalize(features, dim=-1)
            if self.normalize
            else features
        )

        return features

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def forward(self, x: torch.Tensor, return_feature=False) -> torch.Tensor:
        features = self.forward_features(x)
        if return_feature:
            return features
        return self.forward_head(features)


class ChannelLinear(nn.Linear):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, pool=None
    ) -> None:
        super().__init__(in_features, out_features, bias)
        self.compute_axis = 1
        self.pool = nn.Identity() if pool is None else pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        axis_ref = len(x.shape) - 1
        x = torch.transpose(x, self.compute_axis, axis_ref)
        out_shape = list(x.shape)
        out_shape[-1] = self.out_features
        x = x.reshape(-1, x.shape[-1])
        x = x.matmul(self.weight.t())
        if self.bias is not None:
            x = x + self.bias[None, :]
        x = torch.transpose(x.view(out_shape), axis_ref, self.compute_axis)
        x = self.pool(x)
        return x


def make_openclip_probe_model(model_name: str, pretrained: bool = True, **kwargs):
    num_classes = kwargs.pop("num_classes", 1)
    normalize = kwargs.pop("normalize", True)
    next_to_last = kwargs.pop("next_to_last", False)
    is_tune = model_name.endswith("_tune")

    if is_tune:
        model_name = model_name.removesuffix("_tune")
    else:
        assert model_name.endswith("_probe")
        model_name = model_name.removesuffix("_probe")

    try:
        if is_tune:
            return OpenClipTune(
                pretrain=model_name,
                num_classes=num_classes,
                normalize=normalize,
                next_to_last=next_to_last,
            )
        else:
            return OpenClipLinear(
                pretrain=model_name,
                num_classes=num_classes,
                normalize=normalize,
                next_to_last=next_to_last,
            )
    except Exception:
        return None


ModelFactoryRegistry().register_model_factory(
    "openclip_probe_model", make_openclip_probe_model
)


__all__ = ["OpenClipLinear", "make_openclip_probe_model"]


if __name__ == "__main__":
    model = make_openclip_probe_model("clipL14commonpool_tune")
    print(model)
    model = make_openclip_probe_model("clipL14commonpool_probe")
    print(model)
