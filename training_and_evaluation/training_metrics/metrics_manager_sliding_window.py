from typing import Dict, List, Literal, Optional, Set, TYPE_CHECKING
from lightning import LightningDataModule, LightningModule, Trainer
from torch import Tensor
import torch

from training_metrics.metrics_manager import remap_labels_and_preds

from .metrics_manager import GroupT, GeneratorKey, MetricsManager, StageT

if TYPE_CHECKING:
    from lightning_data_modules.deepfake_detection_datamodule import (
        DeepfakeDetectionDatamodule,
    )


class MetricsManagerSlidingWindow(MetricsManager):
    def __init__(
        self,
        compute_mechanism: Literal[
            "growing", "immediate_future", "past", "growing_whole"
        ],
    ):
        super().__init__()

        self.compute_mechanism: Literal[
            "growing", "immediate_future", "past", "growing_whole"
        ] = compute_mechanism

        # Sliding windows order stores the lists of generator indices (one set for each window)
        self.windows_order: Optional[List[Set[int]]] = None

        # Flag to track if the metrics have been updated at least once in the current epoch
        self._updated_this_epoch = False

    def set_windows_order(self, order: List[Set[int]]):
        self.windows_order = order

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: StageT):
        super().setup(trainer, pl_module, stage)
        datamodule: DeepfakeDetectionDatamodule = trainer.datamodule
        generator_id_to_name: List[str] = datamodule.generator_id_to_name
        window_ids: List[List[str]] = datamodule.windows_timeline
        windows_order = [
            set([generator_id_to_name.index(gen_name) for gen_name in window])
            for window in window_ids
        ]

        self.set_windows_order(windows_order)

    def _metric_name_add_info(
        self, metric_name: str, granularity: str, metrics_group: GroupT
    ):
        # Input name: "<metric_base_name>/<generator_name>"
        # Desired output name: "<metric_base_name>_<granularity>/<generator_name>"
        # metric_base_name may contain "/" characters

        test_set_suffix = self.compute_mechanism

        parts = metric_name.rsplit("/", maxsplit=1)
        if len(parts) == 1:
            metric_name = f"{test_set_suffix}_{parts[0]}_{granularity}"
        else:
            metric_name = f"{test_set_suffix}_{parts[0]}_{granularity}/{parts[1]}"

        if metrics_group != "default":
            metric_name = f"{metrics_group}/{metric_name}"

        return metric_name

    def _create_metrics(
        self,
        data_module: LightningDataModule,
        stage: Optional[StageT] = None,
        metrics_group: Optional[GroupT] = None,
    ):
        if stage == "fit":
            return

        return super()._create_metrics(data_module, stage, metrics_group)

    def _reset_metrics(
        self,
        stage: Optional[StageT] = None,
        dataloader_idx: Optional[GeneratorKey] = None,
        metrics_group: Optional[GroupT] = None,
    ):
        if not self._updated_this_epoch:
            return

        return super()._reset_metrics(stage, dataloader_idx, metrics_group)

    def update(
        self,
        predictions: Tensor,
        labels: Tensor,
        generator_ids: Tensor,
        image_identifiers: Tensor,
        stage: Optional[StageT] = None,
        metrics_group: GroupT = "default",
    ):
        if stage is None:
            stage = self._stage_tracker

        if stage == "fit":
            return

        assert self.windows_order is not None

        metrics_group_dict = self.metrics[metrics_group]

        datamodule: "DeepfakeDetectionDatamodule" = self._trainer.datamodule
        assert datamodule is not None
        assert datamodule.sliding_windows_definition is not None
        assert datamodule.sliding_windows_definition.current_window is not None

        current_window: int = (
            0
            if datamodule.sliding_windows_definition.current_window is None
            else datamodule.sliding_windows_definition.current_window
        )
        is_multiclass: bool = predictions.ndim == 2 and predictions.shape[1] > 1

        generators_in_batch = torch.unique(generator_ids).tolist() + ["*"]
        generator_key: GeneratorKey
        for generator_key in generators_in_batch:
            generators_to_consider: Set[int]
            generator_window_idx = None
            if generator_key not in {"*", 0}:
                for window, window_generators in enumerate(self.windows_order):
                    # Find the window this generator belongs to
                    if generator_key in window_generators:
                        generator_window_idx = window
                        break

                # Check if the generator should be considered
                if generator_window_idx is None:
                    raise RuntimeError(f"Generator {generator_key}: window not found!")

                if self.compute_mechanism == "immediate_future":
                    if generator_window_idx != (current_window + 1):
                        continue
                elif self.compute_mechanism == "growing":
                    if generator_window_idx > current_window:
                        continue
                elif self.compute_mechanism == "growing_whole":
                    if generator_window_idx > (current_window + 1):
                        continue
                elif self.compute_mechanism == "past":
                    if generator_window_idx >= current_window:
                        continue
                else:
                    raise ValueError(
                        f"Invalid compute mechanism: {self.compute_mechanism}"
                    )

            if (
                stage in metrics_group_dict
                and generator_key in metrics_group_dict[stage]
            ):
                step_results = None
                if generator_key == "*":
                    if self.compute_mechanism == "immediate_future":
                        next_window_idx = min(
                            current_window + 1, len(self.windows_order) - 1
                        )
                        generators_to_consider = set(
                            self.windows_order[next_window_idx]
                        )

                    elif self.compute_mechanism == "growing":
                        generators_to_consider = set()
                        for window in range(current_window + 1):
                            generators_to_consider.update(self.windows_order[window])

                    elif self.compute_mechanism == "growing_whole":
                        generators_to_consider = set()
                        for window in range(
                            min(current_window + 2, len(self.windows_order))
                        ):
                            generators_to_consider.update(self.windows_order[window])

                    elif self.compute_mechanism == "past":
                        generators_to_consider = set()
                        for window in range(current_window):
                            generators_to_consider.update(self.windows_order[window])
                    else:
                        raise ValueError(
                            f"Invalid compute mechanism: {self.compute_mechanism}"
                        )

                    mask = torch.zeros_like(generator_ids, dtype=torch.bool)
                    splitting: Dict[int, int] = {}
                    if self.include_real_examples_in_metrics:
                        splitting = datamodule.get_dataset_splitting(stage)

                    for gen_id in generators_to_consider:
                        assert gen_id != 0
                        mask = mask | (generator_ids == gen_id)
                        if self.include_real_examples_in_metrics:
                            # Select real images by pairing
                            real_mask = self._select_real_images_by_pairing(
                                generator_ids, image_identifiers, splitting, gen_id
                            )

                            # Combine the fake and real masks
                            mask = mask | real_mask
                else:
                    generators_to_consider = {generator_key}
                    mask = generator_ids == generator_key
                    if generator_key != 0 and self.include_real_examples_in_metrics:
                        splitting: Dict[int, int] = datamodule.get_dataset_splitting(
                            stage
                        )

                        # Select real images by pairing
                        real_mask = self._select_real_images_by_pairing(
                            generator_ids, image_identifiers, splitting, generator_key
                        )

                        # Combine the fake and real masks
                        mask = mask | real_mask

                if mask.sum() == 0:
                    # It may happen if generator_key == "*" and no
                    # example from considered generators is present
                    continue

                generator_predictions = predictions[mask]
                generator_labels = labels[mask]

                # Fix for multiclass classification (with non-contiguous labels)
                if is_multiclass:
                    generators_to_consider.add(0)
                    adapted_labels, adapted_preds = remap_labels_and_preds(
                        generator_labels,
                        generator_predictions,
                        sorted(generators_to_consider),
                    )
                else:
                    adapted_labels, adapted_preds = (
                        generator_labels,
                        generator_predictions,
                    )

                step_metrics, epoch_metrics = metrics_group_dict[stage][generator_key]
                if step_metrics is not None:
                    self._updated_this_epoch = True
                    step_results = step_metrics(adapted_preds, adapted_labels)
                if epoch_metrics is not None:
                    self._updated_this_epoch = True
                    epoch_metrics.update(adapted_preds, adapted_labels)

                if step_results is not None:
                    self._log_metrics_step(
                        self._trainer,
                        self._pl_module,
                        stage,
                        generator_key,
                        step_results,
                        metrics_group=metrics_group,
                    )

    def _on_epoch_start(self, stage: StageT):
        super()._on_epoch_start(stage)
        self._updated_this_epoch = False

    def _allow_metrics_error(self, trainer: Optional[Trainer]):
        if (
            self.compute_mechanism == "immediate_future"
            and not self._updated_this_epoch
        ):
            return True
        return super()._allow_metrics_error(trainer)

    def __repr__(self) -> str:
        return (
            f"MetricsManagerSlidingWindow(compute_mechanism={self.compute_mechanism})"
        )


__all__ = ["MetricsManagerSlidingWindow"]
