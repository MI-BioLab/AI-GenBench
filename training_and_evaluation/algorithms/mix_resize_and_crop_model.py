from pathlib import Path
from typing import (
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from torch import Tensor
from lightning.pytorch.cli import OptimizerCallable
from PIL.Image import Image

from algorithms.base_model import BaseDeepfakeDetectionModel


class MixResizeAndCropModel(BaseDeepfakeDetectionModel):
    """
    A version of the base model in which the model is evaluated
    by using both the resize version of the image and some crops.

    The importance of the scores obtained on the resized image versus the
    crops is controlled by the `crops_scoring_weight` parameter.
    """

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
        """
        Args:
            model_name: the name of the model to be used. Must be a valid model name
                that can be loaded by the model factory.
            optimizer: the optimizer to be used. Must be a callable that receives
                the model parameters and returns an optimizer (usually configured from
                yaml and instantiated by Lightning).
            scheduler: the learning rate scheduler to be used. Must be a string
                representing the scheduler name. You need to implement the scheduler
                configuration in the configure_optimizers() method.
            model_input_size: the input size of the model.
            classification_threshold: the threshold to be used for classification.
            base_weights: the path to the base weights to be loaded. If None, the model
                will be initialized with random weights or the weights defined from the model factory.
            logging_initial_step: the initial step to be used for logging. This is set
                to 0 by default and is shifted when moving to successive windows (automatically
                managed by LightningCLISlidingWindow class).
            training_cropping_strategy: the cropping strategy to be used during training.
            evaluation_cropping_strategy: the cropping strategy to be used during evaluation.
            training_task: the type of training task to be performed.
                Must be either "binary" or "multiclass". This is used to determine the loss function
                to be used during training and evaluation. When using multiclass classification,
                each generator is treated as a separate class, and the model is trained to output
                a probability for each class.
            evaluation_type: the type of problem to be solved. Must be either "binary" or "multiclass".
                Problem type can be multiclass only if training_task is also multiclass.
                If problem type is binary and training_task is multiclass, the model will
                be trained to output a score for each class, the loss function will be multi-class,
                but the output will then be converted to a binary score by calling the
                `multiclass_to_binary_prediciton()` method, which can be overridden to implement
                a custom logic.
            crops_scoring_weight: the weight to be applied to the scores obtained from the crops.
                Must be a value between 0 and 1. Resize scores will be weighted by (1 - crops_scoring_weight),
        """
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
        )

        self.crops_scoring_weight = crops_scoring_weight

        self.save_hyperparameters()

    def scores_fusion(self, scores: Tensor) -> Tensor:
        """
        A method to fuse the scores obtained from multiple crops of an image.

        In this version, the first crop is always the resized image, while subsequent crops
        are the crops obtained from the image. The scores are fused by applying a weight to the
        scores obtained from the crops and a complementary weight to the score obtained from the resized image.
        """

        if len(scores.shape) == 1 or scores.shape[1] == 1:
            # Scores are 1D or single-channel (binary scores)
            resize_score = scores[0]
            crops_scores = scores[1:]
            return (
                resize_score * (1 - self.crops_scoring_weight)
                + crops_scores.mean() * self.crops_scoring_weight
            )
        else:
            # Multiclass scores, of shape (n_crops, n_classes)
            resize_scores = scores[0]
            crops_scores = scores[1:]

            return (
                resize_scores * (1 - self.crops_scoring_weight)
                + crops_scores.mean(dim=0) * self.crops_scoring_weight
            )

    def make_eval_crops(
        self, image: Union[Tensor, Image]
    ) -> Union[Tensor, Image, Sequence[Tensor], Sequence[Image]]:
        """
        A customized version of the make_eval_crops method that always returns the resized image
        as the first crop, followed by the crops obtained from the image.
        """
        resized = self._resize_transform(image)
        crops = super().make_eval_crops(image)

        if isinstance(crops, Sequence):
            # If crops is a sequence, prepend the resized image
            return [resized] + list(crops)
        else:
            # If crops is a single crop, return a list with the resized image and the crop
            return [resized, crops]


__all__ = ["MixResizeAndCropModel"]
