from typing import (
    Callable,
    Optional,
    Tuple,
    Union,
)

from algorithms.augmentation_pipelines.mild_train_augmentations import (
    make_mild_train_aug,
)
from algorithms.base_model import BaseDeepfakeDetectionModel


class MildTrainAugmentationsModel(BaseDeepfakeDetectionModel):

    def train_augmentation(
        self,
    ) -> Union[Callable, Tuple[Callable, Optional[Callable]]]:
        return make_mild_train_aug(
            self.model_input_size,
            self.training_cropping_strategy,
        )


__all__ = ["MildTrainAugmentationsModel"]
