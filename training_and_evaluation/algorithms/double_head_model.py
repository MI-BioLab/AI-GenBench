from copy import deepcopy
from pathlib import Path
from typing import (
    Callable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING,
)
import torch
from torch import Tensor
from torch.nn import Module, Identity, Linear
from lightning.pytorch.cli import OptimizerCallable
from torchmetrics import Metric
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
from lightning.pytorch.callbacks import Callback

from training_metrics.metrics_manager import (
    SUPPORTED_METRICS,
    GeneratorKey,
    MetricsManager,
    StageT,
)

from algorithms.base_model import BaseDeepfakeDetectionModel
from training_metrics.balanced_binary_accuracy import BalancedBinaryAccuracy
from training_metrics.best_binary_threshold import BestBinaryThresholdF1
from training_metrics.metrics_manager_sliding_window import MetricsManagerSlidingWindow

if TYPE_CHECKING:
    from lightning_data_modules.deepfake_detection_datamodule import (
        DeepfakeDetectionDatamodule,
    )


class DoubleHeadModel(BaseDeepfakeDetectionModel):
    """
    A version of `BaseDeepfakeDetectionModel` that features both a binary and a multiclass
    classification heads.

    During training, two losses are computed: one for the binary classification task
    and one for the multiclass classification task. The model is trained to minimize both losses.

    The model supports either:
    - a binary classification head built atop of the multiclass classification head
    - completely separate binary and multiclass classification heads
    """

    def __init__(
        self,
        model_name: str,
        optimizer: OptimizerCallable,
        scheduler: str,
        model_input_size: Union[int, Tuple[int, int]],
        multi_head_approach: Literal["binary_atop", "separate"],
        binary_loss_weight: float = 0.5,
        multiclass_loss_weight: Optional[float] = None,
        classification_threshold: float = 0.5,
        base_weights: Optional[Path] = None,
        logging_initial_step: Optional[int] = 0,
        training_cropping_strategy: Literal[
            "resize", "random_crop", "center_crop", "as_is"
        ] = "resize",
        evaluation_cropping_strategy: Literal[
            "resize", "crop", "multicrop", "as_is"
        ] = "resize",
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
            multi_head_approach: the approach to be used for the multi-head model.
                Must be either "binary_atop" or "separate". If "binary_atop",
                the binary classification head is built atop of the multiclass classification head.
                If "separate", the binary classification head is completely separate from the multiclass
                classification head. In this case, the model will have two separate heads, one for binary
                classification and one for multiclass classification.
            binary_loss_weight: the weight to be used for the binary loss during training.
                This is used to balance the binary and multiclass losses during training.
                The default value is 0.5, meaning that both losses are equally weighted.
            multiclass_loss_weight: the weight to be used for the multiclass loss during training.
                If None, it is set to 1.0 - binary_loss_weight, meaning that
                the multiclass loss is weighted inversely to the binary loss.
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
            training_cropping_strategy=training_cropping_strategy,
            evaluation_cropping_strategy=evaluation_cropping_strategy,
            training_task="multiclass",
            evaluation_type="multiclass",
        )

        self.multi_head_approach: Literal["binary_atop", "separate"] = (
            multi_head_approach
        )
        self.binary_loss_weight: float = binary_loss_weight
        self.multiclass_loss_weight: float = (
            multiclass_loss_weight
            if multiclass_loss_weight is not None
            else 1.0 - binary_loss_weight
        )

        self.save_hyperparameters()

    def setup(self, stage: str):
        self.model = self.make_model()
        if self.base_weights is not None:
            self.load_previous_checkpoint(self.base_weights)

        self.adapt_model_for_training()
        self.register_metrics()

    def configure_callbacks(self):
        model_callbacks = super().configure_callbacks()
        if model_callbacks is None:
            model_callbacks = []
        elif isinstance(model_callbacks, Callback):
            model_callbacks = [model_callbacks]

        found_metric_manager = False
        new_callbacks = []
        for callback in self.trainer.callbacks:
            if isinstance(callback, MetricsManager):
                found_metric_manager = True
                binary_callback = callback
                multiclass_callback = deepcopy(callback)
                setattr(binary_callback, "_is_multiclass_manager", False)
                setattr(multiclass_callback, "_is_multiclass_manager", True)
                self.metric_managers.append(binary_callback)
                self.metric_managers.append(multiclass_callback)

                # How Lightning handles the callback override process is so absurd
                # that you need to fix the final list of callbacks manually after this method
                # (see the `configure_model()` method for the final fix).
                new_callbacks.append(multiclass_callback)
                new_callbacks.append(binary_callback)

        if not found_metric_manager:
            raise RuntimeError("MetricsManager callback(s) not found!")

        return new_callbacks + list(model_callbacks)

    def configure_model(self):
        # De-duplicate callbacks
        unique_callbacks = set(id(x) for x in self.trainer.callbacks)
        to_keep_callbacks = []
        for callback in self.trainer.callbacks:
            if id(callback) in unique_callbacks:
                unique_callbacks.remove(id(callback))
                to_keep_callbacks.append(callback)
        self.trainer.callbacks = self.trainer._callback_connector._reorder_callbacks(
            to_keep_callbacks
        )

    def make_model(self) -> Module:
        model = super().make_model()

        if self.multi_head_approach == "binary_atop":
            # Create a binary head atop of the multiclass head
            model = LayeredMultiHeadClassifier(model)
        else:
            # Create a separate binary head
            model = SeparateHeadsClassifier(model, modify_in_place=False)

        return model

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Implements the forward pass of the model.

        Used by all stages (fit, validate, test, predict).
        """
        multiclass_out, binary_out = self.model(x)
        return multiclass_out, binary_out

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
        multiclass_out: Tensor
        binary_out: Tensor

        x, y, generator_ids, identifiers = batch
        multiclass_out, binary_out = self(x)

        # Squeeze binary output to remove the last dimension
        binary_out = binary_out.squeeze(-1)

        binary_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            binary_out, y.float()
        )
        binary_score = binary_out.detach().sigmoid().flatten()

        multiclass_loss = torch.nn.functional.cross_entropy(
            multiclass_out, generator_ids.long()
        )
        multiclass_score = multiclass_out.detach().softmax(dim=1)

        loss = (binary_loss * self.binary_loss_weight) + (
            multiclass_loss * self.multiclass_loss_weight
        )

        self.log(
            "train_binary_loss",
            binary_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train_multiclass_loss",
            multiclass_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)

        self._update_metrics(
            binary_score, y, generator_ids, identifiers, multiclass_metrics=False
        )
        self._update_metrics(
            multiclass_score,
            generator_ids,
            generator_ids,
            identifiers,
            multiclass_metrics=True,
        )

        return loss

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
        multiclass_out: Tensor
        binary_out: Tensor

        x, y, generator_ids, identifiers, crop_to_image_index = batch
        unique_images = torch.unique(crop_to_image_index)
        max_batch_size = len(unique_images)

        if max_batch_size == len(x):
            # Not using multicrop evaluation
            multiclass_out, binary_out = self(x)  # Forward pass
        else:
            # Multicrop evaluation
            # We need to split the batch into smaller batches to avoid running out of memory
            # when using multiple crops
            multiclass_outs = []
            binary_outs = []
            for i in range(0, len(x), max_batch_size):
                outs = self(x[i : i + max_batch_size])  # Forward pass
                multiclass_outs.append(outs[0])
                binary_outs.append(outs[1])
            multiclass_out = torch.cat(multiclass_outs, dim=0)
            binary_out = torch.cat(binary_outs, dim=0)

        multiclass_out = multiclass_out.detach()
        binary_out = binary_out.detach()

        # Squeeze binary output to remove the last dimension
        binary_out = binary_out.squeeze(-1)

        # Convert multiclass to binary if needed
        binary_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            binary_out, y.float(), reduction="none"
        )
        binary_score = binary_out.sigmoid().flatten()
        multiclass_loss = torch.nn.functional.cross_entropy(
            multiclass_out, generator_ids.long(), reduction="none"
        )
        multiclass_score = multiclass_out.softmax(dim=1)

        loss = (binary_loss.mean() * self.binary_loss_weight) + (
            multiclass_loss.mean() * self.multiclass_loss_weight
        )

        # This will log the loss averaged across all crops
        # That is, it doesn't account for the fact that images can have different
        # number of crops
        self.log(
            f"{stage}_multiclass_loss",
            multiclass_loss.mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{stage}_binary_loss",
            binary_loss.mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        n_images = unique_images.shape[0]
        fusion_multiclass_scores = torch.empty(
            (n_images, *multiclass_score.shape[1:]),
            dtype=multiclass_score.dtype,
            device=multiclass_score.device,
        )
        fusion_binary_scores = torch.empty(
            (n_images, *binary_score.shape[1:]),
            dtype=binary_score.dtype,
            device=binary_score.device,
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
        fusion_multiclass_losses = torch.empty(
            (n_images,), dtype=multiclass_loss.dtype, device=multiclass_loss.device
        )
        fusion_binary_losses = torch.empty(
            (n_images,), dtype=binary_loss.dtype, device=binary_loss.device
        )

        # Store crop-level predictions
        crop_multiclass_scores = multiclass_score.clone()
        crop_binary_scores = binary_score.clone()
        crop_y = y.clone()
        crop_generator_ids = generator_ids.clone()
        crop_identifiers = identifiers.clone()
        crop_multiclass_losss = multiclass_loss.clone()
        crop_binary_losses = binary_loss.clone()
        crop_to_image_mapping = crop_to_image_index.clone()

        for i, unique_image in enumerate(unique_images.tolist()):
            mask = crop_to_image_index == unique_image
            fusion_binary_scores[i] = self.scores_fusion(binary_score[mask])
            fusion_multiclass_scores[i] = self.scores_fusion(multiclass_score[mask])
            fusion_y[i] = y[mask][0]
            fusion_generator_ids[i] = generator_ids[mask][0]
            fusion_identifiers[i] = identifiers[mask][0]
            fusion_multiclass_losses[i] = multiclass_loss[mask].mean()
            fusion_binary_losses[i] = binary_loss[mask].mean()

        self._update_metrics(
            fusion_binary_scores,
            fusion_y,
            fusion_generator_ids,
            fusion_identifiers,
            multiclass_metrics=False,
        )
        self._update_metrics(
            fusion_multiclass_scores,
            fusion_generator_ids,
            fusion_generator_ids,
            fusion_identifiers,
            multiclass_metrics=True,
        )

        # Return results as a dictionary
        results = {
            # Fused results (one per image)
            "fused_multiclass_scores": fusion_multiclass_scores,
            "fused_binary_scores": fusion_binary_scores,
            "fused_labels": fusion_y,
            "fused_generator_ids": fusion_generator_ids,
            "fused_identifiers": fusion_identifiers,
            "fused_multiclass_losses": fusion_multiclass_losses,
            "fused_binary_losses": fusion_binary_losses,
            # Crop-level results (one per crop)
            "crop_multiclass_scores": crop_multiclass_scores,
            "crop_binary_scores": crop_binary_scores,
            "crop_labels": crop_y,
            "crop_generator_ids": crop_generator_ids,
            "crop_identifiers": crop_identifiers,
            "crop_multiclass_losses": crop_multiclass_losss,
            "crop_binary_losses": crop_binary_losses,
            "crop_to_image_index": crop_to_image_mapping,
            # General information
            "identifier": fusion_identifiers,
        }

        return results

    def register_metrics(self):
        """
        Register the metrics to be used during training and evaluation.

        Note: it completely overrides the metrics registered in the parent class.
        """

        # multiclass classification task
        multiclass_fit_accuracy = lambda num_classes: MulticlassAccuracy(
            num_classes=num_classes
        )
        multiclass_fit_recall = lambda num_classes: MulticlassRecall(
            num_classes=num_classes
        )
        multiclass_aggregated_fit_metrics = lambda num_classes: [
            MulticlassPrecision(num_classes=num_classes),
            MulticlassSpecificity(num_classes=num_classes),
            MulticlassAveragePrecision(num_classes=num_classes),
            MulticlassAUROC(num_classes=num_classes),
            MulticlassF1Score(num_classes=num_classes),
        ]

        # binary classification task
        binary_fit_accuracy = lambda num_classes: BalancedBinaryAccuracy(
            threshold=self.classification_threshold
        )
        binary_fit_recall = lambda num_classes: BinaryRecall(
            threshold=self.classification_threshold
        )
        binary_aggregated_fit_metrics = lambda num_classes: [
            BinaryPrecision(threshold=self.classification_threshold),
            BinarySpecificity(threshold=self.classification_threshold),
            BinaryAveragePrecision(),
            BinaryAUROC(),
            BinaryF1Score(threshold=self.classification_threshold),
        ]

        # multiclass evaluation
        multiclass_eval_accuracy = lambda num_classes: MulticlassAccuracy(
            num_classes=num_classes
        )
        multiclass_eval_recall = lambda num_classes: MulticlassRecall(
            num_classes=num_classes
        )
        multiclass_aggregated_eval_metrics = lambda num_classes: [
            MulticlassPrecision(num_classes=num_classes),
            MulticlassSpecificity(num_classes=num_classes),
            MulticlassAveragePrecision(num_classes=num_classes),
            MulticlassAUROC(num_classes=num_classes),
            MulticlassF1Score(num_classes=num_classes),
        ]

        # binary evaluation
        binary_eval_accuracy = lambda num_classes: BalancedBinaryAccuracy(
            threshold=self.classification_threshold
        )
        binary_eval_recall = lambda num_classes: BinaryRecall(
            threshold=self.classification_threshold
        )
        binary_aggregated_eval_metrics = lambda num_classes: [
            BinaryPrecision(threshold=self.classification_threshold),
            BinarySpecificity(threshold=self.classification_threshold),
            BinaryAveragePrecision(),
            BinaryAUROC(),
            BinaryF1Score(threshold=self.classification_threshold),
        ]

        # Accuracy during training
        self._managers_set_metrics_from_factories(
            stage="fit",  # Training only
            generator_id="*",  # All examples
            metric_factories=binary_fit_accuracy,  # A single metric or a MetricCollection
            epoch_only=False,  # Log with both step and epoch granularity
            non_sliding_window_only=True,  # Only applies to "sliding window"-agnostic metric managers
        )

        self._managers_set_metrics_from_factories(
            stage="fit",  # Training only
            generator_id="*",  # All examples
            metric_factories=multiclass_fit_accuracy,  # A single metric or a MetricCollection
            epoch_only=False,  # Log with both step and epoch granularity
            non_sliding_window_only=True,  # Only applies to "sliding window"-agnostic metric managers
            multiclass_metrics=True,  # Only applies to multiclass metric managers
        )

        # Evaluation recall
        self._managers_set_metrics_from_factories(
            stage=["test", "validate"],  # During evaluation only
            generator_id=None,  # A different plot for each generator
            metric_factories=binary_eval_recall,
            epoch_only=True,  # Log only at epoch, not step. Common for eval-time metrics.
            non_sliding_window_only=True,  # Only applies to "sliding window"-agnostic metric managers
        )

        self._managers_set_metrics_from_factories(
            stage=["test", "validate"],  # During evaluation only
            generator_id=None,  # A different plot for each generator
            metric_factories=multiclass_eval_recall,
            epoch_only=True,  # Log only at epoch, not step. Common for eval-time metrics.
            non_sliding_window_only=True,  # Only applies to "sliding window"-agnostic metric managers
            multiclass_metrics=True,  # Only applies to multiclass metric managers
        )

        # Evaluation accuracy
        self._managers_set_metrics_from_factories(
            stage=["test", "validate"],  # During evaluation only
            generator_id="*",  # All examples
            metric_factories=binary_eval_accuracy,
            epoch_only=True,  # Log only at epoch, not step. Common for eval-time metrics.
        )

        self._managers_set_metrics_from_factories(
            stage=["test", "validate"],  # During evaluation only
            generator_id="*",  # All examples
            metric_factories=multiclass_eval_accuracy,
            epoch_only=True,  # Log only at epoch, not step. Common for eval-time metrics.
            multiclass_metrics=True,  # Only applies to multiclass metric managers
        )

        # Training metrics
        self._managers_set_metrics_from_factories(
            stage="fit",  # All stages
            generator_id="*",  # All examples
            metric_factories=[
                binary_fit_recall,  # a.k.a. Sensitivity or True Positive Rate
                binary_aggregated_fit_metrics,  # All other metrics
            ],
            epoch_only=True,  # Log only at epoch, not step
        )

        self._managers_set_metrics_from_factories(
            stage="fit",  # All stages
            generator_id="*",  # All examples
            metric_factories=[
                multiclass_fit_recall,  # a.k.a. Sensitivity or True Positive Rate
                multiclass_aggregated_fit_metrics,  # All other metrics
            ],
            epoch_only=True,  # Log only at epoch, not step
            multiclass_metrics=True,  # Only applies to multiclass metric managers
        )

        # Evaluation metrics
        self._managers_set_metrics_from_factories(
            stage=["test", "validate"],  # All stages
            generator_id="*",  # All examples
            metric_factories=[
                binary_eval_recall,  # a.k.a. Sensitivity or True Positive Rate
                binary_aggregated_eval_metrics,  # All other metrics
            ],
            epoch_only=True,  # Log only at epoch, not step
        )

        self._managers_set_metrics_from_factories(
            stage=["test", "validate"],  # All stages
            generator_id="*",  # All examples
            metric_factories=[
                multiclass_eval_recall,  # a.k.a. Sensitivity or True Positive Rate
                multiclass_aggregated_eval_metrics,  # All other metrics
            ],
            epoch_only=True,  # Log only at epoch, not step
            multiclass_metrics=True,  # Only applies to multiclass metric managers
        )

        self._managers_set_metrics(
            stage=["test", "validate"],  # During evaluation only
            generator_id="*",  # All examples
            metrics=BestBinaryThresholdF1(),
            epoch_only=True,  # Log only at epoch, not step. Common for eval-time metrics.
        )

    def _managers_set_metrics(
        self,
        *,
        stage: Optional[Union[StageT, List[StageT]]],
        generator_id: Optional[Union[GeneratorKey, List[GeneratorKey]]],
        metrics: SUPPORTED_METRICS,
        metrics_group: str = "default",
        epoch_only: bool = False,
        sliding_window_only: bool = False,
        non_sliding_window_only: bool = False,
        multiclass_metrics: bool = False,
        check_stage: bool = True,
    ):
        """
        Utility to set the metrics in the appropriate managers.
        Usually called in the register_metrics() method.
        It is recommended to not customize this method.

        Apart from check_stage, which should be left to True, you should think carefully
        about the values of all the other parameters to obtain the desired behavior.

        Consider checking the `register_metrics()` method for an example of how to use this method.

        Args:
            stage: The stage(s) for which the metrics are set. Can be a single stage, a list of stages,
                or None to apply themetrics to all stages.
            generator_id: The generator ID(s) for which the metrics are set. Can be a single ID, a list of IDs,
                an '*' to register metrics that compute a single plot for all examples, or None to compute the metrics
                separatedly for each generator.
            metrics: The metric(s) to set. Can be a single metric, a list of metrics, or a MetricCollection.
            metrics_group: The group name for the metrics.
            epoch_only: If True, the metrics will be logged only at epoch level, not at step level.
            sliding_window_only: If True, the given metrics will be applied only to sliding-window metric managers
                (the ones that compute metrics for each sliding window separately).
            non_sliding_window_only: If True, only non-sliding window metric managers will be used.
            multiclass_metrics: If True, the metrics will be set only in multiclass metric managers.
                If False, the metrics will be set only in binary metric managers.
            check_stage: If True, the stage will be checked before setting the metrics.
        """
        assert not (sliding_window_only and non_sliding_window_only)
        for manager in self.metric_managers:
            if multiclass_metrics != getattr(manager, "_is_multiclass_manager", False):
                continue

            if sliding_window_only and not isinstance(
                manager, MetricsManagerSlidingWindow
            ):
                continue

            if non_sliding_window_only and isinstance(
                manager, MetricsManagerSlidingWindow
            ):
                continue

            manager.set_metrics(
                stage=stage,
                generator_id=generator_id,
                metrics=metrics,
                metrics_group=metrics_group,
                epoch_only=epoch_only,
                check_stage=check_stage,
            )

    def _managers_set_metrics_from_factories(
        self,
        *,
        stage: Optional[Union[StageT, List[StageT]]],
        generator_id: Optional[Union[GeneratorKey, List[GeneratorKey]]],
        metric_factories: Union[
            Callable[[int], SUPPORTED_METRICS],
            List[Callable[[int], Union[Metric, List[Metric]]]],
        ],
        metrics_group: str = "default",
        epoch_only: bool = False,
        sliding_window_only: bool = False,
        non_sliding_window_only: bool = False,
        multiclass_metrics: bool = False,
        check_stage: bool = True,
    ):
        """
        Utility to set the metrics in the appropriate managers.
        This differs from _managers_set_metrics() in that it uses a factory function
        to specialize the number of classes in the metrics (different between sliding and non-sliding
        window metric managers).

        Args:
            stage: The stage(s) for which the metrics are set. Can be a single stage, a list of stages,
                or None to apply themetrics to all stages.
            generator_id: The generator ID(s) for which the metrics are set. Can be a single ID, a list of IDs,
                an '*' to register metrics that compute a single plot for all examples, or None to compute the metrics
                separatedly for each generator.
            metric_factories: The factories used to create the metrics.
                This should be a callable that takes the number of classes as an argument and returns
                a single metric, a list of metrics, or a MetricCollection.
                The number of classes is the number of generators in the current sliding window,
                or the total number of generators if sliding_window_only is False. This includes the "real" class.
                Factories for binary classification should ignore the number of classes.
            metrics_group: The group name for the metrics.
            epoch_only: If True, the metrics will be logged only at epoch level, not at step level.
            sliding_window_only: If True, the given metrics will be applied only to sliding-window metric managers
                (the ones that compute metrics for each sliding window separately).
            non_sliding_window_only: If True, only non-sliding window metric managers will be used.
            multiclass_metrics: If True, the metrics will be set only in multiclass metric managers.
                If False, the metrics will be set only in binary metric managers.
            check_stage: If True, the stage will be checked before setting the metrics.
        """
        datamodule: "DeepfakeDetectionDatamodule" = self.trainer.datamodule
        num_generators: int = datamodule.num_generators  # Includes the "real" class
        train_generators_so_far = len(datamodule.generators_so_far[0])

        if stage is None:
            stages = ["fit", "validate", "test"]
        elif isinstance(stage, str):
            stages = [stage]
        else:
            stages = list(stage)

        assert not (sliding_window_only and non_sliding_window_only)
        for manager in self.metric_managers:
            if multiclass_metrics != getattr(manager, "_is_multiclass_manager", False):
                continue

            is_sliding_manager = isinstance(manager, MetricsManagerSlidingWindow)
            if sliding_window_only and not is_sliding_manager:
                continue

            if non_sliding_window_only and is_sliding_manager:
                continue

            for stage_name in stages:
                if stage_name == "fit" and not is_sliding_manager:
                    n_classes = train_generators_so_far
                elif is_sliding_manager:
                    part = manager.compute_mechanism
                    n_classes = self.count_part_classes(part)
                else:
                    n_classes = num_generators

                if isinstance(metric_factories, list):
                    metrics = []
                    for factory in metric_factories:
                        created_metrics = factory(n_classes)
                        if isinstance(created_metrics, Metric):
                            metrics.append(created_metrics)
                        else:
                            metrics.extend(created_metrics)
                else:
                    metrics = metric_factories(n_classes)

                manager.set_metrics(
                    stage=stage_name,
                    generator_id=generator_id,
                    metrics=metrics,
                    metrics_group=metrics_group,
                    epoch_only=epoch_only,
                    check_stage=check_stage,
                )

    def _update_metrics(
        self,
        predictions,
        labels,
        generator_ids,
        identifiers,
        multiclass_metrics: bool = False,
    ):
        """
        Update the metrics in the appropriate managers.
        """
        for manager in self.metric_managers:
            if multiclass_metrics != getattr(manager, "_is_multiclass_manager", False):
                continue
            manager.update(predictions, labels, generator_ids, identifiers)


def _remove_model_head(model: Module):
    recognized_head_names = ["classifier", "head", "fc"]
    for head_name in recognized_head_names:
        if hasattr(model, head_name):
            embedding_size = getattr(model, head_name).in_features
            out_size = getattr(model, head_name).out_features
            head = getattr(model, head_name)
            setattr(model, head_name, Identity())
            return head_name, head, embedding_size, out_size

    raise ValueError(f"Unrecognized model head name: {model}")


def _get_model_head(model: Module):
    recognized_head_names = ["classifier", "head", "fc"]
    for head_name in recognized_head_names:
        if hasattr(model, head_name):
            embedding_size = getattr(model, head_name).in_features
            out_size = getattr(model, head_name).out_features
            head = getattr(model, head_name)
            return head_name, head, embedding_size, out_size

    raise ValueError(f"Unrecognized model head name: {model}")


class SeparateHeadsClassifier(Module):
    def __init__(self, backbone: Module, modify_in_place: bool = True):
        """
        A simple classifier with two separate heads for binary and multiclass classification.

        This model supposes that the backbone already features the correct head
        for the multiclass classification task.

        Args:
            backbone: the backbone model to be used.
            modify_in_place: if True, the backbone will be modified in place
                (the head will be removed and replaced with an Identity layer).
                If False, a deepcopy of the backbone will be created and modified.
        """
        super().__init__()
        self.backbone: Module
        if not modify_in_place:
            self.backbone = deepcopy(backbone)
        else:
            self.backbone = backbone

        head_name, head, embedding_size, out_size = _remove_model_head(self.backbone)

        self.binary_head: Module = Linear(embedding_size, 1)
        self.multiclass_head: Module = head

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.backbone(x)
        binary_out = self.binary_head(x)
        multiclass_out = self.multiclass_head(x)
        return multiclass_out, binary_out


class LayeredMultiHeadClassifier(Module):
    def __init__(self, backbone: Module):
        super().__init__()
        self.backbone: Module = backbone

        head_name, head, embedding_size, out_size = _get_model_head(self.backbone)

        self.binary_head: Module = Linear(out_size, 1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.backbone(x)
        multiclass_out = x
        x = torch.nn.functional.relu(x, inplace=False)
        binary_out = self.binary_head(x)
        return multiclass_out, binary_out


__all__ = ["DoubleHeadModel"]
