import os
from typing import Optional, Tuple
from huggingface_hub import hf_hub_download
import torch
from torch import Tensor
from torch.nn import Linear, Module
import torch.hub

from algorithms.model_factory_registry import ModelFactoryRegistry


DINOV3_DEFAULT_REPOSITORY_PATH = 'dinov3'

def make_dinov3_model(model_name: str, pretrained: bool = True, checkpoint_path: Optional[str] = None, **kwargs):
    num_classes = kwargs.pop("num_classes", 1)
    is_tune = model_name.endswith("_tune")
    if is_tune:
        model_name = model_name.removesuffix("_tune")
    else:
        assert model_name.endswith("_probe")
        model_name = model_name.removesuffix("_probe")

    if pretrained and checkpoint_path is None:
        checkpoint_path = os.environ.get("DINOV3_CHECKPOINT_PATH", None)
        if checkpoint_path is None:
            print("DINOV3_CHECKPOINT_PATH env variable not set, downloading checkpoint from HF...")
            raise ValueError("checkpoint_path must be provided when pretrained is True")

    if is_tune:
        return DINOv3ModelTune(model_name, num_classes=num_classes, weights_path=checkpoint_path)
    else:
        return DINOv3ModelProbe(model_name, num_classes=num_classes, weights_path=checkpoint_path)


class DINOv3ModelProbe(torch.nn.Module):
    def __init__(self, model_name: str, weights_path: Optional[str], num_classes: int = 1, shape=(3, 256, 256)):
        super().__init__()

        # Get local rank from env
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )

        # Load DINOv3
        pretrained = weights_path is not None
        backbone: Module = torch.hub.load(_get_dinov3_repository_path(), model_name, source='local', pretrained=pretrained, weights=weights_path)
        backbone.to(device)
        backbone.eval()
        backbone.requires_grad_(False)

        self._bb: Tuple[Module] = (backbone,)

        with torch.no_grad():
            dummy_input = torch.zeros((1, *shape)).to(device)
            features: Tensor = self.backbone(dummy_input)
            self.intermediate_size: int = features.shape[-1]

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
            features = self.backbone(x)
        return features

    def forward_head(self, x: Tensor) -> Tensor:
        return self.fc(x)

    @property
    def backbone(self) -> Module:
        return self._bb[0]


class DINOv3ModelTune(torch.nn.Module):
    def __init__(self, model_name: str, weights_path: Optional[str], num_classes: int = 1, shape=(3, 256, 256)):
        super().__init__()

        # Get local rank from env
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )

        # Load DINOv3
        pretrained = weights_path is not None
        backbone: Module = torch.hub.load(_get_dinov3_repository_path(), model_name, source='local', pretrained=pretrained, weights=weights_path)
        backbone.to(device)
        backbone.eval()
        backbone.requires_grad_(False)

        self.backbone: Module = backbone

        with torch.no_grad():
            dummy_input = torch.zeros((1, *shape)).to(device)
            features: Tensor = self.backbone(dummy_input)
            self.intermediate_size: int = features.shape[-1]

        self.fc = Linear(self.intermediate_size, num_classes)

    def forward(self, x: Tensor, return_feature=False) -> Tensor:
        features = self.forward_features(x)
        if return_feature:
            return features
        return self.forward_head(features)

    def forward_features(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        return features

    def forward_head(self, x: Tensor) -> Tensor:
        return self.fc(x)


def _get_dinov3_repository_path() -> str:
    repo_path = DINOV3_DEFAULT_REPOSITORY_PATH
    if os.environ.get("DINOV3_REPOSITORY_PATH", None) is not None:
        repo_path = os.environ["DINOV3_REPOSITORY_PATH"]
    return repo_path


def _download_hf_checkpoint(repo_id: str, filename: str = "model.safetensors") -> str:
    """Download checkpoint from Hugging Face and return local path."""
    try:
        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )
        return checkpoint_path
    except Exception as e:
        print(f"Failed to download checkpoint from {repo_id}: {e}")
        raise e

ModelFactoryRegistry().register_model_factory("dinov3", make_dinov3_model)

__all__ = [
    "make_dinov3_model",
    "DINOv3ModelProbe",
    "DINOv3ModelTune",
]


if __name__ == "__main__":
    LOCAL_CHECKPOINT_PATH = "..."
    model = make_dinov3_model("dinov3_vitl16_tune", checkpoint_path=LOCAL_CHECKPOINT_PATH)
    print(model)
    model = make_dinov3_model("dinov3_vitl16_probe", checkpoint_path=LOCAL_CHECKPOINT_PATH)
    print(model)
