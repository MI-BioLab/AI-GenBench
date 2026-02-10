from typing import Tuple
import torch
from torch import Tensor
from torch.nn import Linear, Module

from algorithms.model_factory_registry import ModelFactoryRegistry


def make_openai_clip_image_model(model_name: str, pretrained: bool = True, **kwargs):
    num_classes = kwargs.pop("num_classes", 1)
    is_tune = model_name.endswith("_tune")
    if is_tune:
        model_name = model_name.removesuffix("_tune")
    else:
        assert model_name.endswith("_probe")
        model_name = model_name.removesuffix("_probe")

    if is_tune:
        return OpenaiCLIPImageModelTune(model_name, num_classes=num_classes)
    else:
        return OpenaiCLIPImageModelProbe(model_name, num_classes=num_classes)


class OpenaiCLIPImageModelProbe(torch.nn.Module):
    def __init__(self, name, num_classes=1, shape=(3, 224, 224)):
        super().__init__()
        import clip

        backbone, _ = clip.load(
            name, device="cpu"
        )  # self.preprecess will not be used during training, which is handled in Dataset class
        self.reference_dtype = backbone.dtype
        backbone = backbone.visual
        backbone.eval()
        backbone.requires_grad_(False)

        self._bb: Tuple[Module] = (backbone,)

        # Get model intermediate features size
        with torch.no_grad():
            dummy_input = torch.zeros((1, *shape))
            features: Tensor = self.backbone(dummy_input.type(self.reference_dtype))
            self.intermediate_size: int = features.numel()

        self.fc = Linear(self.intermediate_size, num_classes)

    def forward(self, x: Tensor, return_feature=False) -> Tensor:
        features = self.forward_features(x)
        if return_feature:
            return features
        return self.forward_head(features)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.backbone.to(*args, **kwargs)
        return self

    def forward_features(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            features = self.backbone(x.type(self.reference_dtype))
        return features

    def forward_head(self, x: Tensor) -> Tensor:
        return self.fc(x)

    @property
    def backbone(self) -> Module:
        return self._bb[0]


class OpenaiCLIPImageModelTune(torch.nn.Module):
    def __init__(self, name, num_classes=1, shape=(3, 224, 224)):
        super().__init__()
        import clip

        backbone, _ = clip.load(
            name, device="cpu"
        )  # self.preprecess will not be used during training, which is handled in Dataset class
        self.reference_dtype = backbone.dtype
        backbone = backbone.visual
        backbone.eval()
        backbone.requires_grad_(False)

        self.backbone: Module = backbone

        # Get model intermediate features size
        with torch.no_grad():
            dummy_input = torch.zeros((1, *shape))
            features: Tensor = self.backbone(dummy_input.type(self.reference_dtype))
            self.intermediate_size: int = features.numel()
        backbone.requires_grad_(True)

        self.fc = Linear(self.intermediate_size, num_classes)

    def forward(self, x: Tensor, return_feature=False) -> Tensor:
        features = self.forward_features(x)
        if return_feature:
            return features
        return self.forward_head(features)

    def forward_features(self, x: Tensor) -> Tensor:
        features = self.backbone(x.type(self.reference_dtype))
        return features

    def forward_head(self, x: Tensor) -> Tensor:
        return self.fc(x)

import clip
for arch in clip.available_models():
    ModelFactoryRegistry().register_model_factory(
        f"{arch}_tune", make_openai_clip_image_model
    )
    ModelFactoryRegistry().register_model_factory(
        f"{arch}_probe", make_openai_clip_image_model
    )

__all__ = [
    "make_openai_clip_image_model",
    "OpenaiCLIPImageModelProbe",
    "OpenaiCLIPImageModelTune",
]


if __name__ == "__main__":
    model = make_openai_clip_image_model("RN50_tune", num_classes=1)
    print(model)
    model = make_openai_clip_image_model("ViT-L/14_probe", num_classes=1)
    print(model)
