from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING
import warnings
from lightning import Trainer, LightningModule
import shutil

import numpy as np
import torch
from torch import Tensor

from evaluation_scripts.evaluate_predictions_utils import (
    _default_multicrop_predictions_fusion,
    _multicrop_fusion_with_ids,
)
from training_metrics.multi_stage_prediction_writer import (
    MultiStageBasePredictionWriter,
)
import lightning.pytorch as pl
from lightning.fabric.utilities import move_data_to_device

if TYPE_CHECKING:
    from lightning_data_modules.deepfake_detection_datamodule import (
        DeepfakeDetectionDatamodule,
    )


CPU_DEVICE = torch.device("cpu")


class DeepFakePredictionsDump(MultiStageBasePredictionWriter):

    def __init__(
        self,
        gt_keys: Optional[List[str]] = None,
        pred_keys: Optional[List[str]] = None,
        save_dir: Optional[Path] = None,
    ):
        super().__init__(write_interval="batch_and_epoch")

        self._current_data_true: List[Tuple[Tensor, Tensor, Tensor]] = []
        self._current_data_pred: List[Tuple[Tensor, ...]] = []
        self._current_data_dict: List[Dict[str, Tensor]] = []

        self._gt_keys = gt_keys
        self._pred_keys = pred_keys
        self._save_dir = save_dir

        if self._gt_keys is not None and self._pred_keys is not None:
            # Assert no intersections
            assert len(set(self._gt_keys).intersection(set(self._pred_keys))) == 0

    def write_on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        prediction: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
        stage: str,
    ):
        if isinstance(prediction, dict):
            self._current_data_dict.append(move_data_to_device(prediction, CPU_DEVICE))
        else:
            scores, labels, generator_ids, identifiers, losses = prediction

            self._current_data_true.append(
                move_data_to_device((labels, generator_ids, identifiers), CPU_DEVICE)
            )
            self._current_data_pred.append(
                move_data_to_device((scores, losses), CPU_DEVICE)
            )

    def write_on_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        predictions: Sequence[Any],
        stage: str,
    ):
        save_dir = self._make_save_dir(trainer)

        if save_dir is None:
            print("No logdir set, skipping saving predictions")
            return

        if (
            len(self._current_data_true) == 0
            and len(self._current_data_pred) == 0
            and len(self._current_data_dict) == 0
        ):
            warnings.warn("No predictions collected, skipping saving predictions")
            return

        to_save = dict()

        if len(self._current_data_dict) > 0:
            assert (
                len(self._current_data_true) == 0 and len(self._current_data_pred) == 0
            ), "If using dictionary-based predictions, _current_data_true and _current_data_pred should be empty."
            # Handle dictionary-based predictions
            all_keys = set()
            for batch_dict in self._current_data_dict:
                all_keys.update(batch_dict.keys())

            # Concatenate data for each key across all batches
            for key in all_keys:
                batch_data = [
                    batch_dict[key]
                    for batch_dict in self._current_data_dict
                    if key in batch_dict
                ]
                if batch_data:
                    concatenated = torch.cat(batch_data, dim=0)
                    to_save[key] = DeepFakePredictionsDump.to_numpy(concatenated)
        else:
            assert (
                len(self._current_data_dict) == 0
            ), "If using tuple-based predictions, _current_data_dict should be empty."

            data_true_batch_elements = len(self._current_data_true[0])
            data_pred_batch_elements = len(self._current_data_pred[0])

            for i in range(data_true_batch_elements):
                data_true = DeepFakePredictionsDump.to_numpy(
                    torch.cat([batch[i] for batch in self._current_data_true], dim=0)
                )

                if self._gt_keys is not None and i < len(self._gt_keys):
                    to_save[self._gt_keys[i]] = data_true
                else:
                    to_save[f"data_true_{i}"] = data_true

            for i in range(data_pred_batch_elements):
                data_pred = DeepFakePredictionsDump.to_numpy(
                    torch.cat([batch[i] for batch in self._current_data_pred], dim=0)
                )

                if self._pred_keys is not None and i < len(self._pred_keys):
                    to_save[self._pred_keys[i]] = data_pred
                else:
                    to_save[f"data_pred_{i}"] = data_pred

        additional_elements = None
        if hasattr(pl_module, "elements_to_dump"):
            # If the model has a method to dump additional elements, use it
            additional_elements = pl_module.elements_to_dump

        assert additional_elements is None or isinstance(
            additional_elements, dict
        ), "Module's elements_to_dump field should be a dictionary of additional elements to save."

        if additional_elements is not None and trainer.global_rank == 0:
            # Merge additional elements into the to_save dictionary
            for key, value in additional_elements.items():
                assert key not in to_save, (
                    f"Key '{key}' from elements_to_dump already exists in to_save. "
                    "Please ensure unique keys."
                )
                if isinstance(value, Tensor):
                    value = DeepFakePredictionsDump.to_numpy(value)
                to_save[key] = value

        # Save predictions
        rank = trainer.global_rank
        path = save_dir / f"predictions_{stage}_{rank}.npz"

        np.savez(path, **to_save)
        self._clear_data()

    def merge_multiprocess_predictions(
        self, stage: str, trainer: Trainer, pl_module: LightningModule
    ):
        save_dir = self._make_save_dir(trainer)

        if save_dir is None:
            self._clear_data()
            return

        current_epoch = trainer.current_epoch
        global_step = trainer.global_step
        how_many_files = trainer.world_size
        latest_predictions_filename = f"predictions_{stage}_all.npz"
        result_filename = (
            f"predictions_{stage}_epoch={current_epoch}-step={global_step}.npz"
        )

        latest_result_path = save_dir / latest_predictions_filename
        result_path = save_dir / result_filename

        if trainer.global_rank == 0:
            # Wait for all prediction files to be available
            # (Alas, torch dist barrier may have some issues in some systems here...)
            max_wait_time = 60  # 1 minute timeout
            wait_interval = 1  # Check every 1 second
            waited_time = 0

            while waited_time < max_wait_time:
                all_files_exist = True
                for rank in range(how_many_files):
                    path = save_dir / f"predictions_{stage}_{rank}.npz"
                    if not path.exists():
                        all_files_exist = False
                        break

                if all_files_exist:
                    break

                time.sleep(wait_interval)
                waited_time += wait_interval

            if waited_time >= max_wait_time:
                print(
                    f"Warning: Timeout waiting for all prediction files. Some files may be missing."
                )

            # Additional small delay to ensure files are fully written
            time.sleep(2)

            # Merge all files into a single one
            elements = []
            concat_elements = dict()
            error_path = save_dir / "predictions_store_error"  # to be touched on error
            had_errors = False
            for rank in range(how_many_files):
                path = save_dir / f"predictions_{stage}_{rank}.npz"
                if path.exists():
                    try:
                        data = np.load(path)
                        elements.append(data)
                    except Exception as e:
                        had_errors = True
                        print(f"Warning: Failed to load {path}: {e}")
                        continue
                else:
                    had_errors = True
                    print(f"Warning: Missing prediction file {path}")

            if had_errors:
                # Create an error file to indicate issues
                error_path.touch()
                print(f"Warning: Some prediction files could not be loaded/merged.")

            if not elements:
                print("Error: No prediction files could be loaded")
                self._clear_data()
                return

            # Concatenate each element
            keys = list(elements[0].keys())
            other_keys = list(elements[1 if trainer.world_size > 1 else 0].keys())
            additional_keys = set(keys) - set(other_keys)
            common_keys = set(keys) & set(other_keys)

            for key in common_keys:
                concat_elements[key] = np.concatenate(
                    [data[key] for data in elements], axis=0
                )

            # Add the additional keys from the first element (rank 0)
            for key in additional_keys:
                concat_elements[key] = elements[0][key]

            datamodule: "DeepfakeDetectionDatamodule" = trainer.datamodule
            concat_elements["generator_id_to_name"] = np.array(
                datamodule.generator_id_to_name
            )
            concat_elements["windows_timeline"] = np.array(datamodule.windows_timeline)

            assert "identifier" in keys
            # Image identifier -> generator_id
            data_metrics_splitting: Dict[int, int] = datamodule.get_dataset_splitting(
                stage
            )

            # Make a tensor from data_metrics_splitting, aligned with the key "identifier"
            generator_pairing = np.array(
                [
                    data_metrics_splitting.get(int(identifier), -1)
                    for identifier in concat_elements["identifier"]
                ]
            )
            concat_elements["pairing"] = generator_pairing

            np.savez(result_path, **concat_elements)

            if latest_result_path.exists():
                latest_result_path.unlink()

            shutil.copyfile(result_path, latest_result_path)

            # Clean up individual rank files
            for rank in range(how_many_files):
                path = save_dir / f"predictions_{stage}_{rank}.npz"
                path.unlink(missing_ok=True)

        self._clear_data()

    @staticmethod
    def to_numpy(tensor: Tensor) -> np.ndarray:
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            # Manages bfloat16, which is not automatically converted to numpy (raises an error!)
            # https://stackoverflow.com/a/78128663
            return tensor.float().numpy()

        return tensor.numpy()

    def _make_save_dir(self, trainer: Trainer) -> Optional[Path]:
        save_dir = self._save_dir
        if save_dir is None:
            save_dir = trainer.log_dir

        if save_dir is None:
            return None

        save_dir = Path(save_dir)
        return save_dir

    def _clear_data(self):
        """Clear the current data buffers."""
        self._current_data_true.clear()
        self._current_data_pred.clear()
        self._current_data_dict.clear()


__all__ = ["DeepFakePredictionsDump"]
