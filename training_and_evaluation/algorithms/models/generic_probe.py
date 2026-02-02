from typing import Tuple
from torch import Tensor
import torch
from torch.nn import Module, Linear


class GenericModelProbe(Module):
    """
    A generic class that can be used to create a probe model for any backbone.

    In probe (or linear probe, or last-layer tuning) the backbone is frozen and
    only the last layer/head is trained. To achieve this, the backbone is set to eval mode
    and requires_grad for its weights are set to False. In addition, the backbone is hidden
    from the list of modules (and parameters) to prevent it from being trained by accident.
    This also has the positive side effect of making the model checkpoint smaller.
    """

    def __init__(
        self,
        backbone: Module,
        embedding_size: int,
        num_classes: int = 1,
        new_head=None,
    ):
        """
        Args:
            backbone (Module): The backbone model to be used.
            embedding_size (int): The size of the embedding produced by the backbone.
            num_classes (int): The number of classes for the final classification layer.
            new_head (Module, optional): The head to replace the default one.
                If None, a new Linear layer will be created with the specified embedding size
                and number of classes.
        """
        super().__init__()

        self.embedding_size = embedding_size

        # Load DINOv2
        backbone.eval()
        backbone.requires_grad_(False)

        self._bb: Tuple[Module] = (backbone,)
        if new_head is None:
            new_head = Linear(embedding_size, num_classes)

        self.fc = new_head

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


__all__ = ["GenericModelProbe"]
