from pathlib import Path
from typing import (
    Callable,
    Literal,
    Optional,
    Tuple,
    Union,
)
from lightning.pytorch.cli import OptimizerCallable

from algorithms.augmentation_pipelines.soft_train_augmentations import (
    make_soft_train_aug,
)
from algorithms.mix_resize_and_crop_model import MixResizeAndCropModel


class MixResizeAndCropSoftAugModel(MixResizeAndCropModel):
    def __init__(
        self,
        model_name: str,
        optimizer: OptimizerCallable,
        scheduler: str,
        model_input_size: Union[int, Tuple[int, int]],
        classification_threshold: float = 0.5,
        base_weights: Optional[Path] = None,
        logging_initial_step: Optional[int] = 0,
        training_cropping_strategy: Literal[
            "resize", "random_crop", "center_crop", "as_is"
        ] = "resize",
        evaluation_cropping_strategy: Literal["crop", "multicrop"] = "multicrop",
        training_task: Literal["binary", "multiclass"] = "binary",
        evaluation_type: Literal["binary", "multiclass"] = "binary",
        crops_scoring_weight: float = 0.5,
    ):
        super().__init__(
            model_name=model_name,
            optimizer=optimizer,
            scheduler=scheduler,
            model_input_size=model_input_size,
            classification_threshold=classification_threshold,
            base_weights=base_weights,
            logging_initial_step=logging_initial_step,
            training_cropping_strategy=training_cropping_strategy,
            evaluation_cropping_strategy=evaluation_cropping_strategy,
            training_task=training_task,
            evaluation_type=evaluation_type,
            crops_scoring_weight=crops_scoring_weight,
        )

        self.save_hyperparameters()

    def train_augmentation(
        self,
    ) -> Union[Callable, Tuple[Callable, Optional[Callable]]]:
        return make_soft_train_aug(
            self.model_input_size,
            self.training_cropping_strategy,
        )


__all__ = ["MixResizeAndCropSoftAugModel"]
