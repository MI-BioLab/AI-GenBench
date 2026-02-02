from collections import defaultdict
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    NamedTuple,
    TYPE_CHECKING,
)
from lightning import Callback, LightningDataModule
import torch
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics import MetricCollection
from lightning import Trainer, LightningModule
import warnings

if TYPE_CHECKING:
    from lightning_data_modules.deepfake_detection_datamodule import (
        DeepfakeDetectionDatamodule,
    )


SUPPORTED_METRICS = Union[Metric, List[Metric], MetricCollection]

GroupT = str
StageT = str
GeneratorT = int
GeneratorTCumulative = Literal["*"]
GeneratorKey = Union[GeneratorT, GeneratorTCumulative]


class StepEpochMetrics(NamedTuple):
    step_metrics: Optional[MetricCollection]
    epoch_metrics: Optional[MetricCollection]


INSTANTIATED_METRICS_T = Dict[StageT, Dict[GeneratorKey, StepEpochMetrics]]

METRICS_DEF_T = Dict[Optional[StageT], Dict[Optional[GeneratorKey], StepEpochMetrics]]


class MetricsManager(Callback):
    def __init__(self, include_real_examples_in_metrics: bool = True):
        # Dictionary format stage -> loader_idx/name -> (step metrics, epoch metrics)

        self.metrics: Dict[GroupT, INSTANTIATED_METRICS_T] = defaultdict(dict)
        self.registered_metrics: Dict[GroupT, METRICS_DEF_T] = defaultdict(dict)
        self.generator_names: Dict[GeneratorKey, str] = dict()
        self.include_real_examples_in_metrics: bool = include_real_examples_in_metrics

        self._stage_tracker: StageT = "fit"
        self._generator_tracker: GeneratorT = 0
        self._trainer: Trainer = None
        self._pl_module: LightningModule = None

    def set_metrics(
        self,
        stage: Optional[Union[StageT, List[StageT]]],
        generator_id: Optional[Union[GeneratorKey, List[GeneratorKey]]],
        metrics: SUPPORTED_METRICS,
        metrics_group: str = "default",
        epoch_only: bool = False,
        check_stage: bool = True,
    ):
        assert stage is None or isinstance(stage, (StageT, List))
        assert generator_id in (None, "*") or isinstance(
            generator_id, (GeneratorT, List)
        )

        if isinstance(stage, List):
            for p in stage:
                self.set_metrics(
                    p,
                    generator_id,
                    metrics,
                    metrics_group=metrics_group,
                    epoch_only=epoch_only,
                )
            return

        if check_stage and stage not in [None, "fit", "validate", "test"]:
            raise ValueError(f"Invalid stage: {stage}")

        if isinstance(generator_id, List):
            for idx in generator_id:
                self.set_metrics(
                    stage,
                    idx,
                    metrics,
                    metrics_group=metrics_group,
                    epoch_only=epoch_only,
                )
            return

        if not isinstance(metrics, MetricCollection):
            metrics = MetricCollection(metrics)

        stage_dict: Dict[Optional[GeneratorKey], StepEpochMetrics]

        registered_metrics_dict = self.registered_metrics[metrics_group]

        if stage not in registered_metrics_dict:
            stage_dict = dict()
            registered_metrics_dict[stage] = stage_dict
        else:
            stage_dict = registered_metrics_dict[stage]

        if generator_id not in stage_dict:
            generator_metrics = StepEpochMetrics(None, None)
            stage_dict[generator_id] = generator_metrics
        else:
            generator_metrics = stage_dict[generator_id]

        # TODO: WIP, to be tested
        if epoch_only:
            new_metrics_list = generator_metrics[1]
            if new_metrics_list is None:
                new_metrics_list = metrics
            else:
                new_metrics_list.add_metrics(metrics)

            generator_metrics = StepEpochMetrics(generator_metrics[0], new_metrics_list)
        else:
            new_metrics_list = generator_metrics[0]
            if new_metrics_list is None:
                new_metrics_list = metrics
            else:
                new_metrics_list = new_metrics_list.add_metrics(metrics)

            generator_metrics = StepEpochMetrics(new_metrics_list, generator_metrics[1])

        stage_dict[generator_id] = generator_metrics

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: StageT):
        self._trainer = trainer
        self._pl_module = pl_module

    def _create_metrics(
        self,
        data_module: LightningDataModule,
        stage: Optional[StageT] = None,
        metrics_group: Optional[GroupT] = None,
    ):
        if metrics_group is None:
            for metrics_group in self.registered_metrics.keys():
                self._create_metrics(data_module, stage, metrics_group)
            return

        registered_metrics_dict = self.registered_metrics[metrics_group]

        if stage is None:
            for stage in registered_metrics_dict:
                self._create_metrics(data_module, stage, metrics_group)
            return
        elif None in registered_metrics_dict:
            pass
        elif stage not in registered_metrics_dict:
            return

        all_keys: List[Tuple[StageT, GeneratorKey, str]] = self._make_keys(
            data_module, stage
        )

        metrics_group_dict = self.metrics[metrics_group]

        generator_id: GeneratorKey
        generator_name: str
        for stage, generator_id, generator_name in all_keys:
            if (
                stage in metrics_group_dict
                and generator_id in metrics_group_dict[stage]
            ):
                continue

            stage_dicts: List[Dict[Optional[GeneratorKey], StepEpochMetrics]] = []
            if stage in registered_metrics_dict:
                stage_dicts.append(registered_metrics_dict[stage])
            if None in registered_metrics_dict:
                stage_dicts.append(registered_metrics_dict[None])

            for stage_dict in stage_dicts:
                generator_lists: List[StepEpochMetrics] = []
                if generator_id in stage_dict:
                    generator_lists.append(stage_dict[generator_id])
                if None in stage_dict:
                    generator_lists.append(stage_dict[None])

                for generator_metrics_tuple in generator_lists:
                    if stage not in metrics_group_dict:
                        metrics_group_dict[stage] = dict()

                    instiantiated_metrics = self._instantiate_metrics(
                        stage, generator_id, generator_name, generator_metrics_tuple
                    )
                    if generator_id not in metrics_group_dict[stage]:
                        metrics_group_dict[stage][generator_id] = instiantiated_metrics
                    else:
                        metrics_group_dict[stage][generator_id] = _merge_metrics_defs(
                            metrics_group_dict[stage][generator_id],
                            instiantiated_metrics,
                        )

    def _instantiate_metrics(
        self,
        stage: StageT,
        generator_id: GeneratorKey,
        generator_name: str,
        metrics_template: StepEpochMetrics,
    ) -> StepEpochMetrics:
        step_metrics, epoch_metrics = metrics_template
        metrics_suffix = f"/{generator_name}"
        metrics_prefix = f"{stage}_"

        step_metrics = (
            step_metrics.clone(prefix=metrics_prefix, postfix=metrics_suffix).to(
                self._pl_module.device
            )
            if step_metrics is not None
            else None
        )
        epoch_metrics = (
            epoch_metrics.clone(prefix=metrics_prefix, postfix=metrics_suffix).to(
                self._pl_module.device
            )
            if epoch_metrics is not None
            else None
        )

        return StepEpochMetrics(step_metrics, epoch_metrics)

    def _make_keys(
        self, data_module: "DeepfakeDetectionDatamodule", stage: StageT
    ) -> List[Tuple[StageT, GeneratorKey, str]]:

        keys = []
        keys.append(
            (
                stage,
                "*",
                self._make_generator_name(data_module, "*"),
            )
        )

        for generator_id, _ in enumerate(data_module.generator_id_to_name):
            keys.append(
                (
                    stage,
                    generator_id,
                    self._make_generator_name(data_module, generator_id),
                )
            )

        return keys

    def _make_generator_name(
        self,
        data_module: "DeepfakeDetectionDatamodule",
        generator_id: GeneratorKey,
    ) -> str:
        if len(self.generator_names) == 0:
            available_generators = data_module.generator_id_to_name
            for g_id, generator_name in enumerate(available_generators):
                self.generator_names[g_id] = self._clean_generator_name(generator_name)

        if generator_id == "*":
            return "all"
        else:
            return self.generator_names.get(generator_id, str(generator_id))

    def _clean_generator_name(self, gen_name: str):
        if gen_name == "":
            return "real"
        return gen_name.replace(".", "_").replace(" ", "_")

    def _reset_metrics(
        self,
        stage: Optional[StageT] = None,
        generator_id: Optional[GeneratorKey] = None,
        metrics_group: Optional[GroupT] = None,
    ):
        if metrics_group is None:
            for metrics_group in self.metrics:
                self._reset_metrics(stage, generator_id, metrics_group)
            return

        metrics_group_dict = self.metrics[metrics_group]

        if stage is None:
            for stage in metrics_group_dict:
                self._reset_metrics(stage, generator_id, metrics_group)
            return
        elif stage not in metrics_group_dict:
            return

        if generator_id is None:
            for generator_id in metrics_group_dict[stage]:
                self._reset_metrics(stage, generator_id, metrics_group)
            return

        step_metrics, epoch_metrics = metrics_group_dict[stage][generator_id]
        if step_metrics is not None:
            step_metrics.reset()
        if epoch_metrics is not None:
            epoch_metrics.reset()

    def _log_metrics_epoch(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: StageT,
        generator_id: GeneratorKey,
        metrics_group: Optional[GroupT] = None,
    ):
        if metrics_group is None:
            for metrics_group in self.metrics:
                self._log_metrics_epoch(
                    trainer, pl_module, stage, generator_id, metrics_group
                )
            return

        metrics_group_dict = self.metrics[metrics_group]

        if (
            stage not in metrics_group_dict
            or generator_id not in metrics_group_dict[stage]
        ):
            return

        step_metrics, epoch_metrics = metrics_group_dict[stage][generator_id]
        values = None
        if epoch_metrics is not None:
            values = self._compute_metrics_safe(epoch_metrics)

        if step_metrics is not None:
            if values is None:
                values = self._compute_metrics_safe(step_metrics)
            else:
                values.update(self._compute_metrics_safe(step_metrics))

        if values is not None:
            # Append "_epoch" to the keys
            values = {
                self._metric_name_add_info(key, "epoch", metrics_group): value
                for key, value in values.items()
            }
            self._log_dict(pl_module, values)

    def _log_metrics_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: StageT,
        generator_id: GeneratorKey,
        values: Optional[Dict[str, Any]] = None,
        metrics_group: Optional[GroupT] = None,
    ):
        if metrics_group is None:
            for metrics_group in self.metrics:
                self._log_metrics_step(
                    trainer, pl_module, stage, generator_id, values, metrics_group
                )
            return

        metrics_group_dict = self.metrics[metrics_group]

        if (
            stage not in metrics_group_dict
            or generator_id not in metrics_group_dict[stage]
        ):
            return

        if values is None:
            step_metrics, epoch_metrics = metrics_group_dict[stage][generator_id]
            if step_metrics is not None:
                values = self._compute_metrics_safe(step_metrics)

        if values is not None:
            # Append "_step" to the keys
            values = {
                self._metric_name_add_info(key, "step", metrics_group): value
                for key, value in values.items()
            }
            self._log_dict(pl_module, values)

    def _log_dict(self, pl_module: LightningModule, values: Dict[str, Any]):
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings(
                "ignore", message=r"It is recommended to use.+sync_dist=True.*"
            )
            pl_module.log_dict(values, add_dataloader_idx=False)

    def _metric_name_add_info(
        self, metric_name: str, granularity: str, metrics_group: GroupT
    ):
        # Input name: "<metric_base_name>/<generator_name>"
        # Desired output name: "<metric_base_name>_<granularity>/<generator_name>"
        # metric_base_name may contain "/" characters

        parts = metric_name.rsplit("/", maxsplit=1)
        if len(parts) == 1:
            metric_name = f"{parts[0]}_{granularity}"
        else:
            metric_name = f"{parts[0]}_{granularity}/{parts[1]}"

        if metrics_group != "default":
            metric_name = f"{metrics_group}/{metric_name}"

        return metric_name

    def _log_metrics_epoch_all_generators(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: StageT,
        metrics_group: Optional[GroupT] = None,
    ):
        if metrics_group is None:
            for metrics_group in self.metrics:
                self._log_metrics_epoch_all_generators(
                    trainer, pl_module, stage, metrics_group
                )
            return

        metrics_group_dict = self.metrics[metrics_group]

        if stage not in metrics_group_dict:
            return

        for generator_id in metrics_group_dict[stage]:
            self._log_metrics_epoch(
                trainer, pl_module, stage, generator_id, metrics_group=metrics_group
            )

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # print('[METRICS] on_fit_start')
        self._create_metrics(trainer.datamodule, "fit")
        self._reset_metrics("fit")
        self._stage_tracker = "fit"

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        # print('[METRICS] on_train_epoch_start')
        self._on_epoch_start("fit")
        self._reset_metrics("fit")
        self._stage_tracker = "fit"

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # print('[METRICS] on_train_epoch_end')
        self._log_metrics_epoch_all_generators(trainer, pl_module, "fit")

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # print('[METRICS] on_fit_end')
        self._reset_metrics("fit")

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        # print('[METRICS] on_validation_epoch_start')
        self._on_epoch_start("validate")
        self._create_metrics(trainer.datamodule, "validate")
        self._reset_metrics("validate")
        self._stage_tracker = "validate"

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        # print('[METRICS] on_validation_epoch_end')
        self._log_metrics_epoch_all_generators(trainer, pl_module, "validate")
        self._reset_metrics("validate")

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # print('[METRICS] on_test_epoch_start')
        self._on_epoch_start("test")
        self._create_metrics(trainer.datamodule, "test")
        self._reset_metrics("test")
        self._stage_tracker = "test"

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # print('[METRICS] on_test_epoch_end')
        self._log_metrics_epoch_all_generators(trainer, pl_module, "test")
        self._reset_metrics("test")

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

        metrics_group_dict = self.metrics[metrics_group]

        datamodule: "DeepfakeDetectionDatamodule" = self._trainer.datamodule
        assert datamodule is not None

        is_multiclass: bool = predictions.ndim == 2 and predictions.shape[1] > 1
        fit_generators_to_consider = sorted(datamodule.generators_so_far[1])

        generators_in_batch = torch.unique(generator_ids).tolist() + ["*"]
        generator_key: GeneratorKey
        for generator_key in generators_in_batch:
            if (
                stage in metrics_group_dict
                and generator_key in metrics_group_dict[stage]
            ):
                step_results = None
                if generator_key == "*":
                    generator_predictions = predictions
                    generator_labels = labels
                else:
                    # Select fake images for this generator
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

                    generator_predictions = predictions[mask]
                    generator_labels = labels[mask]

                # Fix for multiclass classification (with non-contiguous labels)
                if is_multiclass and stage == "fit":
                    # Remove columns for generators not in fit_generators_to_consider
                    adapted_labels, adapted_preds = remap_labels_and_preds(
                        generator_labels,
                        generator_predictions,
                        fit_generators_to_consider,
                    )
                else:
                    adapted_labels, adapted_preds = (
                        generator_labels,
                        generator_predictions,
                    )

                step_metrics, epoch_metrics = metrics_group_dict[stage][generator_key]
                if step_metrics is not None:
                    step_results = step_metrics(adapted_preds, adapted_labels)
                if epoch_metrics is not None:
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

    def _compute_metrics_safe(
        self, metrics: MetricCollection, trainer: Optional[Trainer] = None
    ):
        try:
            return metrics.compute()
        except Exception as e:
            if not self._allow_metrics_error(trainer):
                raise e

        return dict()

    def _allow_metrics_error(self, trainer: Optional[Trainer]):
        return trainer is not None and trainer.sanity_checking

    def _on_epoch_start(self, stage: StageT):
        pass

    def _select_real_images_by_pairing(
        self,
        generators: Tensor,
        identifiers: Tensor,
        splitting: Dict[int, int],
        generator_key: GeneratorT,
    ) -> Tensor:
        """
        Selects real images based on the pairing with a specific generator.

        Args:
            generators (Tensor): Tensor of generator IDs for each image in the batch.
            identifiers (Tensor): Tensor of image identifiers for each image in the batch.
            splitting (Dict[int, int]): Dictionary mapping image identifiers to generator IDs.
            generator_key (GeneratorKey): The generator ID to pair real images with.

        Returns:
            Tensor: A boolean mask indicating which images are selected.
        """
        device = generators.device
        # Create a boolean mask for real images that should be included
        real_mask = (generators == 0) & torch.isin(
            identifiers,
            torch.tensor(
                [k for k, v in splitting.items() if v == generator_key],
                device=device,
            ),
        )
        return real_mask

    def __repr__(self) -> str:
        return f"MetricsManager(include_real_examples_in_metrics={self.include_real_examples_in_metrics})"


def _merge_metrics_defs(
    a: Optional[StepEpochMetrics], b: Optional[StepEpochMetrics]
) -> StepEpochMetrics:
    if a is None and b is None:
        return StepEpochMetrics(None, None)
    elif a is None:
        return b
    elif b is None:
        return a
    else:
        assert a is not None
        assert b is not None
        step_metrics = dict()
        if a.step_metrics is not None:
            step_metrics.update(a.step_metrics)
        if b.step_metrics is not None:
            step_metrics.update(b.step_metrics)

        epoch_metrics = dict()
        if a.epoch_metrics is not None:
            epoch_metrics.update(a.epoch_metrics)
        if b.epoch_metrics is not None:
            epoch_metrics.update(b.epoch_metrics)

        if len(step_metrics) == 0:
            step_metrics = None
        else:
            step_metrics = MetricCollection(step_metrics)

        if len(epoch_metrics) == 0:
            epoch_metrics = None
        else:
            epoch_metrics = MetricCollection(epoch_metrics)

        return StepEpochMetrics(step_metrics, epoch_metrics)


__all__ = ["MetricsManager"]


def remap_labels_and_preds(
    targets: Tensor,
    preds: Tensor,
    original_labels: List[int],
) -> tuple[Tensor, Tensor]:
    """
    Remap original non-contiguous class labels and predictions to contiguous format
    suitable for torchmetrics.

    Args:
        targets (Tensor): shape (batch_size,), containing original class IDs.
        preds (Tensor): shape (batch_size, full_num_classes) OR (batch_size, num_classes),
                        predicted logits or probabilities. If larger than num_classes,
                        assumes full label space.
        original_labels (List[int]): list of class IDs actually used, e.g., [0, 4, 12, 21, 33]

    Returns:
        mapped_targets (Tensor): shape (batch_size,), mapped to [0, num_classes-1]
        mapped_preds (Tensor): shape (batch_size, num_classes), ordered to match new labels
    """
    label_to_index = {label: idx for idx, label in enumerate(original_labels)}
    index_tensor = torch.tensor(original_labels, device=preds.device)

    # Map targets to new contiguous indices
    mapped_targets = torch.tensor(
        [label_to_index[int(t)] for t in targets],
        device=targets.device,
        dtype=targets.dtype,
    )

    # If preds has more columns than original_labels, select only relevant columns
    if preds.shape[1] > len(original_labels):
        preds = preds[:, index_tensor]

    return mapped_targets, preds
