from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
import torch
from torch import Tensor
from torch.nn import Module
from lightning.pytorch.cli import OptimizerCallable
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion
from torchmetrics.classification import (
    BinaryRecall,
    BinaryPrecision,
    BinaryAveragePrecision,
    BinaryAUROC,
    BinaryF1Score,
    BinarySpecificity,
    MulticlassAccuracy,
    MulticlassRecall,
    MulticlassPrecision,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassF1Score,
    MulticlassSpecificity,
)

from algorithms.abstract_model import AbstractBaseDeepfakeDetectionModel

from torch.optim.lr_scheduler import OneCycleLR
from PIL.Image import Image

from torchvision.transforms.v2 import Resize, CenterCrop, FiveCrop
import torchvision.transforms.v2.functional as F

from algorithms.augmentation_pipelines.baseline_augmentations import (
    make_baseline_predict_aug,
    make_baseline_test_aug,
    make_baseline_train_aug,
    make_baseline_val_aug,
)
from lightning_data_modules.deepfake_detection_datamodule import (
    DeepfakeDetectionDatamodule,
)
from training_metrics.balanced_binary_accuracy import BalancedBinaryAccuracy
from training_metrics.best_binary_threshold import BestBinaryThresholdF1


class BaseDeepfakeDetectionModel(AbstractBaseDeepfakeDetectionModel):
    """
    The base class for deepfake detection models.

    This class can be used as a starting point for implementing your own
    deepfake detection model.

    You can use this class:
    - as a template, by copy-pasting its code and modifying it to suit your needs
    - by inheriting from it and overriding its methods

    The methods already implemented here are a default implementation
    that can be used as is. Most of the internal bookkeping is already
    done in the parent class, so you can focus on the core logic of your model.
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
        evaluation_cropping_strategy: Literal[
            "resize", "crop", "multicrop", "as_is"
        ] = "resize",
        training_task: Literal["binary", "multiclass"] = "binary",
        evaluation_type: Literal["binary", "multiclass"] = "binary",
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
        """
        super().__init__(
            model_name=model_name,
            optimizer=optimizer,
            scheduler=scheduler,
            model_input_size=model_input_size,
            classification_threshold=classification_threshold,
            base_weights=base_weights,
            logging_initial_step=logging_initial_step,
        )

        self.training_cropping_strategy: Literal[
            "resize", "random_crop", "center_crop", "as_is"
        ] = training_cropping_strategy
        self.evaluation_cropping_strategy: Literal[
            "resize", "crop", "multicrop", "as_is"
        ] = evaluation_cropping_strategy

        # Note: those are not used if training/evaluation_cropping_strategy are "as_is"
        # You should use "as_is" if your model is image-size agnostic!
        self._resize_transform: Callable = Resize(self.model_input_size)
        self._center_crop_transform: Callable = CenterCrop(self.model_input_size)
        self._multicrop_transform: Callable = FiveCrop(self.model_input_size)

        self.training_task: Literal["binary", "multiclass"] = training_task
        self.evaluation_type: Literal["binary", "multiclass"] = evaluation_type

        if self.evaluation_type == "multiclass" and self.training_task == "binary":
            raise ValueError(
                "Evaluation type cannot be 'multiclass' if training_task is 'binary'."
            )

        self._multiclass_output_mask: Optional[Tensor] = None

        self.save_hyperparameters()

    def make_model(self) -> Module:
        """
        Create the model instance and stores it in self.model.

        This method differs from the make_model() method in the parent class
        in that it also sets the number of classes for the model if the
        classification task is multiclass.
        """
        if self.training_task == "multiclass":
            datamodule: DeepfakeDetectionDatamodule = self.trainer.datamodule
            num_generators = datamodule.num_generators
            self.model_args["num_classes"] = num_generators

        return super().make_model()

    def adapt_model_for_training(self):
        """
        Adapts the model for training.

        This method is called:
        - during the setup of the model, after the model
            has been created and the base weights have been loaded, or
        - after the validation/test, before returning to training.

        Defaults to setting the model in train mode and setting
        requires_grad to True for all the model parameters.

        Can be overridden to implement custom logic.
        """
        return super().adapt_model_for_training()

    def adapt_model_for_evaluation(
        self, stage_name: Literal["validation", "test", "predict"]
    ):
        """
        Adapts the model for evaluation.

        This method is called before starting the validation/test/predict stage.
        The name of the stage being started is passed as an argument.

        Defaults to setting the model in eval mode (note: setting requires_grad is
        usually not needed as Lightning runs the evaluation step using torch.no_grad/inference_mode)

        Can be overridden to implement custom logic.
        """
        return super().adapt_model_for_evaluation(stage_name)

    def forward(self, x: Tensor) -> Tensor:
        """
        Implements the forward pass of the model.

        Used by all stages (fit, validate, test, predict).
        """
        x = self.model(x)
        return x

    def training_step(
        self, batch: Tuple[Tensor, ...], batch_idx: int, dataloader_idx=0
    ):
        """
        A default, yet customizable, training step for deepfake detection models.

        This method is called during the training loop and should return the loss
        for the current batch.

        The default implementation computes the loss using binary cross entropy
        and logs the training loss and metrics.

        For most uses, it is already good enough to use as is.
        """
        x: Tensor
        y: Tensor
        generator_ids: Tensor
        identifiers: Tensor
        out: Tensor

        x, y, generator_ids, identifiers = batch
        out = self(x).squeeze(1)

        if self.training_task == "binary":
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out, y.float())
            score = out.detach().sigmoid().flatten()
        else:
            loss = torch.nn.functional.cross_entropy(out, generator_ids.long())
            score = out.detach().softmax(dim=1)

        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)

        if self.training_task == "multiclass":
            # Metrics expect multiclass ground truth labels
            y = generator_ids

        self._update_metrics(score, y, generator_ids, identifiers)

        return loss

    def validation_step(
        self, batch: Tuple[Tensor, ...], batch_idx: int, dataloader_idx=0
    ):
        """
        The validation step for deepfake detection models.

        The return values should be a tuple with the following values:
        - the output for each image
        - the ground truth labels for each image
        - the generator ids for each image
        - the identifiers for each image
        - the per-example loss

        The default YAML configurations contain a DeepFakePredictionsDump callback
        which will save these values in the log directory for further analysis.

        The default implementation of this method can be found `evaluation_step()`, which
        is shared by "val" and "test".
        """
        return self.evaluation_step(batch, batch_idx, dataloader_idx, stage="val")

    def test_step(self, batch: Tuple[Tensor, ...], batch_idx: int, dataloader_idx=0):
        """
        The test step for deepfake detection models.

        The return values should be a tuple with the following values:
        - the output for each image
        - the ground truth labels for each image
        - the generator ids for each image
        - the identifiers for each image
        - the per-example loss

        The default YAML configurations contain a DeepFakePredictionsDump callback
        which will save these values in the log directory for further analysis.

        The default implementation of this method can be found `evaluation_step()`, which
        is shared by "val" and "test".

        This method is only called when calling the main script using the "test" method.
        """
        return self.evaluation_step(batch, batch_idx, dataloader_idx, stage="test")

    @property
    def multiclass_output_mask(self) -> Tensor:
        """
        A mask to select only the generators that have been encountered so far
        during training (including the "real" generator). This is used to remove
        uninitialized classes in multiclass outputs.
        """
        if self._multiclass_output_mask is None:
            datamodule: DeepfakeDetectionDatamodule = self.trainer.datamodule

            n_generators_in_benchmark: int = datamodule.num_generators
            generators_so_far: Set[int] = datamodule.generators_so_far[1]

            # Create a mask to select only the generators that have been encountered so far
            self._multiclass_output_mask = torch.zeros(
                (n_generators_in_benchmark,), dtype=torch.bool, device=self.device
            )
            self._multiclass_output_mask[list(generators_so_far)] = (
                True  # Includes the "real" generator
            )

        return self._multiclass_output_mask

    @multiclass_output_mask.setter
    def multiclass_output_mask(self, mask: Optional[Tensor]):
        if mask is None:
            self._multiclass_output_mask = None
        else:
            self._multiclass_output_mask = mask.to(device=self.device)

    def evaluation_step(
        self,
        batch: Tuple[Tensor, ...],
        batch_idx: int,
        dataloader_idx=0,
        stage: str = "val",
    ):
        """
        A generic evaluation step for deepfake detection models.

        This method is called during the validation and test loops and should return the output
        for each example and the per-example loss.

        The default implementation computes the loss using binary cross entropy
        and logs the validation/test loss and metrics.

        This already includes the logic to handle multicrop evaluation, where multiple crops
        are passed to the model and the scores are fused using the scores_fusion() method.
        This also takes care to handle the maximum batch size for the model in a way that prevents
        running out of memory when using multiple crops.

        Note: while train/validation/test_step() are Lightning methods, while evaluation_step()
        is a custom method which is used to unify the validation and test steps.
        You can implement separate validation and test steps by overriding validation_step()
        and test_step() instead.
        """
        x: Tensor
        y: Tensor
        generator_ids: Tensor
        identifiers: Tensor
        crop_to_image_index: Tensor
        out: Tensor

        x, y, generator_ids, identifiers, crop_to_image_index = batch
        unique_images = torch.unique(crop_to_image_index)
        max_batch_size = len(unique_images)

        if max_batch_size == len(x):
            # Not using multicrop evaluation
            out = self(x).squeeze(1)  # Forward pass
        else:
            # Multicrop evaluation
            # We need to split the batch into smaller batches to avoid running out of memory
            # when using multiple crops
            outs = []
            for i in range(0, len(x), max_batch_size):
                out = self(x[i : i + max_batch_size]).squeeze(1)  # Forward pass
                outs.append(out)
            out = torch.cat(outs, dim=0)

        out = out.detach()

        # Store original multiclass predictions before any conversion
        original_multiclass_predictions = (
            out.clone() if self.training_task == "multiclass" else None
        )

        # Convert multiclass to binary if needed
        if self.training_task == "multiclass" and self.evaluation_type == "binary":
            # Also remove outputs for uninitialized classes using self.multiclass_output_mask
            out = self.multiclass_to_binary_prediction(
                out[:, self.multiclass_output_mask]
            )

        if self.training_task == "binary" or self.evaluation_type == "binary":
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                out, y.float(), reduction="none"
            )
            score = out.sigmoid().flatten()
        else:
            loss = torch.nn.functional.cross_entropy(
                out, generator_ids.long(), reduction="none"
            )
            score = out.softmax(dim=1)

        # This will log the loss averaged across all crops
        # That is, it doesn't account for the fact that images can have different
        # number of crops
        self.log(
            f"{stage}_loss", loss.mean(), on_step=False, on_epoch=True, sync_dist=True
        )

        if self.evaluation_type == "multiclass":
            # Metrics expect multiclass ground truth labels
            y = generator_ids

        n_images = unique_images.shape[0]
        fusion_scores = torch.empty(
            (n_images, *score.shape[1:]), dtype=score.dtype, device=score.device
        )
        fusion_y = torch.empty((n_images, *y.shape[1:]), dtype=y.dtype, device=y.device)
        fusion_generator_ids = torch.empty(
            (n_images, *generator_ids.shape[1:]),
            dtype=generator_ids.dtype,
            device=generator_ids.device,
        )
        fusion_identifiers = torch.empty(
            (n_images, *identifiers.shape[1:]),
            dtype=identifiers.dtype,
            device=identifiers.device,
        )
        fusion_losses = torch.empty((n_images,), dtype=loss.dtype, device=loss.device)

        # Store crop-level predictions
        crop_scores = score.clone()
        crop_y = y.clone()
        crop_generator_ids = generator_ids.clone()
        crop_identifiers = identifiers.clone()
        crop_losses = loss.clone()
        crop_to_image_mapping = crop_to_image_index.clone()

        # Store original multiclass predictions for crops if available
        crop_multiclass_predictions = (
            original_multiclass_predictions.clone()
            if original_multiclass_predictions is not None
            else None
        )

        # Prepare fused multiclass predictions if available
        fusion_multiclass_predictions = None
        if original_multiclass_predictions is not None:
            fusion_multiclass_predictions = torch.empty(
                (n_images, *original_multiclass_predictions.shape[1:]),
                dtype=original_multiclass_predictions.dtype,
                device=original_multiclass_predictions.device,
            )

        for i, unique_image in enumerate(unique_images.tolist()):
            mask = crop_to_image_index == unique_image
            fusion_scores[i] = self.scores_fusion(score[mask])
            fusion_y[i] = y[mask][0]
            fusion_generator_ids[i] = generator_ids[mask][0]
            fusion_identifiers[i] = identifiers[mask][0]
            fusion_losses[i] = loss[mask].mean()

            # Fuse multiclass predictions if available
            if original_multiclass_predictions is not None:
                fusion_multiclass_predictions[i] = self.scores_fusion(
                    original_multiclass_predictions[mask]
                )

        self._update_metrics(
            fusion_scores, fusion_y, fusion_generator_ids, fusion_identifiers
        )

        # Return results as a dictionary
        results = {
            # Fused results (one per image)
            "fused_scores": fusion_scores,
            "fused_labels": fusion_y,
            "fused_generator_ids": fusion_generator_ids,
            "fused_identifiers": fusion_identifiers,
            "fused_losses": fusion_losses,
            # Crop-level results (one per crop)
            "crop_scores": crop_scores,
            "crop_labels": crop_y,
            "crop_generator_ids": crop_generator_ids,
            "crop_identifiers": crop_identifiers,
            "crop_losses": crop_losses,
            "crop_to_image_index": crop_to_image_mapping,
            # General information
            "identifier": fusion_identifiers,
        }

        # Add multiclass predictions if available
        if original_multiclass_predictions is not None:
            results["fused_multiclass_predictions"] = fusion_multiclass_predictions
            results["crop_multiclass_predictions"] = crop_multiclass_predictions

        return results

    def predict_step(self, batch: Tuple, batch_idx: int, dataloader_idx=0):
        """
        The predict step for deepfake detection models.

        Note: "predict" is usually only used to obtain predictions on unlabeled data.
        These predictions are dumped in the logging directory
        (for instance, to upload them to a competition portal)
        and are not used for evaluation purposes.

        This method is only called when calling the main script using the "predict" method.
        """
        x: Tensor
        out: Tensor

        x = batch[0]
        out = self(x)

        return out

    def multiclass_to_binary_prediction(
        self,
        batch_multiclass_scores: Tensor,
    ) -> Tensor:
        """
        Converts multiclass scores to binary scores.

        This method is used when the classification task is multiclass,
        but the evaluation type is binary. It sums or otherwise aggregates the probabilities
        of all classes except the first one (which is assumed to be the "real" class)
        to obtain a single score for each example.

        By default, it sums the probabilities of all classes except the first one.

        Args:
            batch_multiclass_scores: The multiclass scores output by the model.
                This is a tensor of shape (batch_size, num_classes).
        Returns:
            A tensor of shape (batch_size,) containing the binary scores.
        """
        softmax_scores = batch_multiclass_scores.softmax(dim=1)

        # Compute fake probability by summing probabilities for classes with index >=1
        # (we only want to consider classes encountered during training, thus the mask)
        binary_scores = softmax_scores[:, 1:].sum(dim=1).flatten()

        # It may happen, due to numerical instability, that the sum may get > 1.0 (like 1.0000001)
        # We need to clip the values to [0, 1], otherswise torchmetrics will just go
        # (alas silently) crazy (it will apply an additional sigmoid if scores are even just a bit above 1.0).
        return binary_scores.clamp(0.0, 1.0)

    def scores_fusion(self, scores: Tensor) -> Tensor:
        """
        A method to fuse the scores obtained from multiple crops of an image.

        This method is called when using a multicrop evaluation strategy.

        The default implementation returns the mean of the scores.

        You can override this method to implement a different fusion strategy.
        """
        if len(scores.shape) == 1 or scores.shape[1] == 1:
            # If scores are 1D or single-channel (binary scores) just return the mean
            return scores.mean()
        else:
            # Multiclass scores, of shape (n_crops, n_classes): return the mean across crops
            return scores.mean(dim=0)

    def configure_optimizers(self):
        """
        Lightning method to configure the optimizer and the learning rate scheduler.

        You can use this method as a template to configure your own optimizer and scheduler.

        Consider looking at the Lightning documentation for more details.
        """
        self.trainer.fit_loop.setup_data()
        optimizer = self.optimizer_factory(
            [x for x in self.model.parameters() if x.requires_grad]
        )

        scheduler_arguments: Dict[str, Any] = dict(
            optimizer=optimizer,
        )

        # Only add "monitor" if your scheduler needs to
        # monitor a metric, such as when using ReduceLROnPlateau
        scheduler_def: Dict[str, Any] = {
            "interval": "step",  # Valid values: "step", "epoch"
            # "monitor": "train_loss_step",
        }

        # Very handy, estimated_stepping_batches already considers gradient accumulation
        total_steps = int(self.trainer.estimated_stepping_batches)

        # Implement your custom sheduler here!
        if self.scheduler_name == "OneCycleLR":
            scheduler_arguments["total_steps"] = total_steps
            scheduler_arguments["max_lr"] = optimizer.param_groups[0]["lr"]

            scheduler = OneCycleLR(**scheduler_arguments)

            scheduler_def["interval"] = "step"
        else:
            raise ValueError(f"Invalid scheduler name: {self.scheduler_name}")

        scheduler_def["scheduler"] = scheduler

        # Must return a Ligthning-compatible definition!
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_def,
        }

    def lr_scheduler_step(
        self, scheduler: LRSchedulerTypeUnion, metric: Optional[Any]
    ) -> None:
        """
        An optional Lightning method to step the learning rate scheduler.

        You may want to customize this if you are using a custom learning rate scheduler
        that requires more than just `step()` or `step(metric)`.
        """
        return super().lr_scheduler_step(scheduler, metric)

    def register_metrics(self):
        """
        Register the metrics to be used during training and evaluation.

        Note: it completely overrides the metrics registered in the parent class.
        """

        if self.training_task == "multiclass":
            # multiclass task
            fit_accuracy = lambda num_classes: MulticlassAccuracy(
                num_classes=num_classes
            )
            fit_recall = lambda num_classes: MulticlassRecall(num_classes=num_classes)
            aggregated_fit_metrics = lambda num_classes: [
                MulticlassPrecision(num_classes=num_classes),
                MulticlassSpecificity(num_classes=num_classes),
                MulticlassAveragePrecision(num_classes=num_classes),
                MulticlassAUROC(num_classes=num_classes),
                MulticlassF1Score(num_classes=num_classes),
            ]
        else:
            # binary classification task
            fit_accuracy = lambda num_classes: BalancedBinaryAccuracy(
                threshold=self.classification_threshold
            )
            fit_recall = lambda num_classes: BinaryRecall(
                threshold=self.classification_threshold
            )
            aggregated_fit_metrics = lambda num_classes: [
                BinaryPrecision(threshold=self.classification_threshold),
                BinarySpecificity(threshold=self.classification_threshold),
                BinaryAveragePrecision(),
                BinaryAUROC(),
                BinaryF1Score(threshold=self.classification_threshold),
            ]

        if self.evaluation_type == "multiclass":
            # multiclass evaluation
            eval_accuracy = lambda num_classes: MulticlassAccuracy(
                num_classes=num_classes
            )
            eval_recall = lambda num_classes: MulticlassRecall(num_classes=num_classes)
            aggregated_eval_metrics = lambda num_classes: [
                MulticlassPrecision(num_classes=num_classes),
                MulticlassSpecificity(num_classes=num_classes),
                MulticlassAveragePrecision(num_classes=num_classes),
                MulticlassAUROC(num_classes=num_classes),
                MulticlassF1Score(num_classes=num_classes),
            ]
        else:
            # binary evaluation
            eval_accuracy = lambda num_classes: BalancedBinaryAccuracy(
                threshold=self.classification_threshold
            )
            eval_recall = lambda num_classes: BinaryRecall(
                threshold=self.classification_threshold
            )
            aggregated_eval_metrics = lambda num_classes: [
                BinaryPrecision(threshold=self.classification_threshold),
                BinarySpecificity(threshold=self.classification_threshold),
                BinaryAveragePrecision(),
                BinaryAUROC(),
                BinaryF1Score(threshold=self.classification_threshold),
            ]

        self._managers_set_metrics_from_factories(
            stage="fit",  # Training only
            generator_id="*",  # All examples
            metric_factories=fit_accuracy,  # A single metric or a MetricCollection
            epoch_only=False,  # Log with both step and epoch granularity
            non_sliding_window_only=True,  # Only applies to "sliding window"-agnostic metric managers
        )

        self._managers_set_metrics_from_factories(
            stage=["test", "validate"],  # During evaluation only
            generator_id=None,  # A different plot for each generator
            metric_factories=eval_recall,
            epoch_only=True,  # Log only at epoch, not step. Common for eval-time metrics.
            non_sliding_window_only=True,  # Only applies to "sliding window"-agnostic metric managers
        )

        self._managers_set_metrics_from_factories(
            stage=["test", "validate"],  # During evaluation only
            generator_id="*",  # All examples
            metric_factories=eval_accuracy,
            epoch_only=True,  # Log only at epoch, not step. Common for eval-time metrics.
            # Applies to all metric managers
        )

        self._managers_set_metrics_from_factories(
            stage="fit",  # All stages
            generator_id="*",  # All examples
            metric_factories=[
                fit_recall,  # a.k.a. Sensitivity or True Positive Rate
                aggregated_fit_metrics,  # All other metrics
            ],
            epoch_only=True,  # Log only at epoch, not step
            # Applies to all metric managers
        )

        self._managers_set_metrics_from_factories(
            stage=["test", "validate"],  # All stages
            generator_id="*",  # All examples
            metric_factories=[
                eval_recall,  # a.k.a. Sensitivity or True Positive Rate
                aggregated_eval_metrics,  # All other metrics
            ],
            epoch_only=True,  # Log only at epoch, not step
            # Applies to all metric managers
        )

        if self.evaluation_type == "binary":
            self._managers_set_metrics(
                stage=["test", "validate"],  # During evaluation only
                generator_id="*",  # All examples
                metrics=BestBinaryThresholdF1(),
                epoch_only=True,  # Log only at epoch, not step. Common for eval-time metrics.
            )

    def train_augmentation(
        self,
    ) -> Union[Callable, Tuple[Callable, Optional[Callable]]]:
        """
        A function that returns the augmentation for the training set.

        Important note: it should return 1 or 2 Callable values: the first callable is the deterministic part of the augmentation,
        the second (optional) one is the non-deterministic part of the augmentation (random rotation, crop, etcetera).

        If you only need purely-deterministic augmentations, only return one callable.
        """
        return make_baseline_train_aug(
            self.model_input_size,
            self.training_cropping_strategy,
        )

    def val_augmentation(self) -> Tuple[Callable, Callable]:
        """
        A function that returns the augmentation for the validation set.

        The augmentation is divided in two parts: before and after cropping/resize.
        When running a multicrop evaluation (that is, if you implemented a make_val_crops()
        returning more than 1 image), the second part is called for each crop.
        """
        return make_baseline_val_aug()

    def test_augmentation(self) -> Tuple[Callable, Callable]:
        """
        A function that returns the augmentation for the test set.

        The augmentation is divided in two parts: before and after cropping/resize.
        When running a multicrop evaluation (that is, if you implemented a make_test_crops()
        returning more than 1 image), the second part is called for each crop.
        """
        return make_baseline_test_aug()

    def predict_augmentation(self) -> Tuple[Callable, Callable]:
        """
        A function that returns the augmentation for the prediction set.

        The augmentation is divided in two parts: before and after cropping/resize.
        When running a multicrop evaluation (that is, if you implemented a make_predict_crops()
        returning more than 1 image), the second part is called for each crop.
        """
        return make_baseline_predict_aug()

    def make_val_crops(
        self, image: Union[Tensor, Image]
    ) -> Union[Tensor, Image, Sequence[Tensor], Sequence[Image]]:
        """
        A function that returns one or multiple views/crops for each image.

        IMPORTANT: when returning multiple views/crops for each image, an appropriate predictions/scores
        fusion mechanism should be implemented in the validation_step() method of the model.

        Each returned crop will be augmented with the second callable returned by the the val_aug() function.

        Hint: this function can also be used to implement a plain resize.
        """
        return self.make_eval_crops(image)

    def make_test_crops(
        self, image: Union[Tensor, Image]
    ) -> Union[Tensor, Image, Sequence[Tensor], Sequence[Image]]:
        """
        A function that returns one or multiple crops for each image.

        IMPORTANT: when returning multiple views/crops for each image, an appropriate predictions/scores
        fusion mechanism should be implemented in the test_step() method of the model.

        Each returned crop will be augmented with the second callable returned by the the test_aug() function.

        Hint: this function can also be used to implement a plain resize.
        """
        return self.make_eval_crops(image)

    def make_predict_crops(
        self, image: Union[Tensor, Image]
    ) -> Union[Tensor, Image, Sequence[Tensor], Sequence[Image]]:
        """
        A function that returns one or multiple crops for each image.

        IMPORTANT: when returning multiple views/crops for each image, an appropriate predictions/scores
        fusion mechanism should be implemented in the predict_step() method of the model.

        Each returned crop will be augmented with the second callable returned by the the predict_aug() function.

        Hint: this function can also be used to implement a plain resize.
        """
        return self.make_eval_crops(image)

    def make_eval_crops(
        self, image: Union[Tensor, Image]
    ) -> Union[Tensor, Image, Sequence[Tensor], Sequence[Image]]:
        if self.evaluation_cropping_strategy == "as_is":
            return image
        elif self.evaluation_cropping_strategy == "resize":
            return self._resize_transform(image)
        elif self.evaluation_cropping_strategy == "crop":
            return self._center_crop_transform(image)
        elif self.evaluation_cropping_strategy == "multicrop":
            # Will default to center crop if image is smaller than self.model_input_size

            img_size = F.get_size(image)
            if (
                img_size[0] <= self.model_input_size[0]
                or img_size[1] <= self.model_input_size[1]
            ):
                return self._center_crop_transform(image)
            else:
                return self._multicrop_transform(image)
        else:
            raise ValueError(
                f"Invalid evaluation cropping strategy: {self.evaluation_cropping_strategy}"
            )

    def on_validation_epoch_end(self) -> None:
        self.multiclass_output_mask = None
        return super().on_validation_epoch_end()

    def on_test_epoch_end(self) -> None:
        self.multiclass_output_mask = None
        return super().on_test_epoch_end()


__all__ = ["BaseDeepfakeDetectionModel"]
