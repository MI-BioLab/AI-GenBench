from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING,
)
import traceback
import warnings
from lightning.pytorch.cli import OptimizerCallable
from lightning.pytorch.callbacks.progress.progress_bar import ProgressBar
import numpy as np
from tqdm import tqdm
from algorithms.base_model import BaseDeepfakeDetectionModel

import torch
from torch import Tensor
import torch.distributed as dist
from sklearn.cluster import KMeans

if TYPE_CHECKING:
    from lightning_data_modules.deepfake_detection_datamodule import (
        DeepfakeDetectionDatamodule,
    )


class KNNModel(BaseDeepfakeDetectionModel):

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
        training_task: Literal["multiclass"] = "multiclass",
        evaluation_type: Literal["binary", "multiclass"] = "binary",
        extract_centroids: bool = True,
        n_centroids_per_class: int = 1,
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
        )

        self.extract_centroids = extract_centroids
        self.n_centroids_per_class = n_centroids_per_class

        # Storage for centroids and kNN classifier
        self.centroids: Optional[Tensor] = None
        self.centroid_labels: Optional[Tensor] = None
        self.centroid_labels_unique: Optional[List[int]] = None

        self.save_hyperparameters()

    @property
    def elements_to_dump(self) -> Optional[Dict[str, Any]]:
        """
        Property obtained by the predictions dumper.

        Should return a dictionary of elements to be saved in the predictions dump (taken only from rank 0).
        """
        # Store the centroids
        if self.centroids is None:
            return None

        assert self.centroid_labels is not None
        assert self.centroid_labels_unique is not None

        to_dump = {
            "centroids": self.centroids.cpu(),
            "centroid_labels": (self.centroid_labels.cpu()),
        }

        return to_dump

    def adapt_model_for_evaluation(self, *args, **kwargs):
        """Extract features with final model state and compute centroids (DDP-safe)"""
        super().adapt_model_for_evaluation(*args, **kwargs)

        # Get training dataloader automatically
        train_dataloader = self._get_train_dataloader()
        if train_dataloader is None:
            if self.trainer.is_global_zero:
                warnings.warn("Warning: Could not access training dataloader.")
            return

        # Extract features using the updated model (DDP-safe)
        print("Extracting features and setting up kNN classifier...")
        self._extract_and_setup_centroids_ddp(train_dataloader)
        print("Done setting up kNN classifier.")

    def _get_train_dataloader(self):
        """Get the DDP-prepared training dataloader"""
        # Only use the trainer's prepared dataloader for DDP safety
        if hasattr(self, "trainer") and self.trainer is not None:
            try:
                dataloader = self.trainer.train_dataloader
                assert dataloader is not None, "Trainer's train_dataloader is None"
                return dataloader
            except:
                warnings.warn(
                    "Could not access the train_dataloader from the trainer. "
                    "This might happen if the dataloader is not set up yet (or directly running an evaluation from CLI). "
                    "Will try to force-initialize the training dataloader..."
                )

            try:
                datamodule: "DeepfakeDetectionDatamodule" = self.trainer.datamodule
                datamodule.setup("fit")
                self.trainer.fit_loop.setup_data()
                dataloader = self.trainer.train_dataloader
                assert dataloader is not None, "Trainer's train_dataloader is None"
                return dataloader
            except:
                traceback.print_exc()
                warnings.warn("Could not force-initialize the training dataloader.")
        else:
            warnings.warn(
                "No trainer available to access the train_dataloader. "
                "This might happen if the model is not being trained or evaluated in a Lightning context."
            )

        # Fallback to manually set reference (user responsibility for DDP)
        if (
            hasattr(self, "train_dataloader_ref")
            and self.train_dataloader_ref is not None
        ):
            return self.train_dataloader_ref

        return None

    def is_progress_bar_enabled(self) -> bool:
        """Check if the progress bar is enabled"""
        progress_bars = [
            c for c in self.trainer.callbacks if isinstance(c, ProgressBar)
        ]
        return len(progress_bars) > 0

    @torch.inference_mode()
    def _extract_and_setup_centroids_ddp(self, train_dataloader):
        """Extract features and setup kNN in a DDP-safe manner"""
        # Only compute centroids on rank 0 to avoid conflicts
        centroids = None
        centroid_labels = None

        # Extract features on main process only
        initial_mode = self.model.training
        self.model.eval()
        all_features_list = []
        all_labels_list = []

        x: Tensor
        generator_ids: Tensor

        loader_tqdm = tqdm(
            train_dataloader,
            desc="Extracting features",
            disable=not self.is_progress_bar_enabled(),
        )
        for batch in loader_tqdm:
            x, _, generator_ids, *_ = batch
            x = x.to(self.device, non_blocking=True)
            features = self.model(x)[:, self.multiclass_output_mask]
            # print(features.shape, flush=True)
            all_features_list.append(features.cpu())
            all_labels_list.append(generator_ids.cpu())

        print(" Done extracting features.")

        # Concatenate all features and labels
        all_features = torch.cat(all_features_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0)
        del all_features_list
        del all_labels_list

        # Get data from other ranks
        if self.trainer.world_size > 1:
            all_features = gather_variable_tensors(all_features.to(self.device))
            all_labels = gather_variable_tensors(all_labels.to(self.device))

            if self.trainer.is_global_zero:
                assert (
                    all_features is not None
                ), "Rank 0: all features should not be None"
                assert all_labels is not None, "Rank 0: all labels should not be None"
                all_features = torch.cat([x.cpu() for x in all_features], dim=0)
                all_labels = torch.cat([x.cpu() for x in all_labels], dim=0)

        # Compute centroids
        if self.trainer.is_global_zero:
            assert isinstance(all_features, Tensor)
            assert isinstance(all_labels, Tensor)

            all_features = all_features.cpu().numpy()
            all_labels = all_labels.cpu().numpy()

            if self.extract_centroids:
                centroids, centroid_labels = self._compute_centroids(
                    all_features, all_labels
                )
            else:
                centroids, centroid_labels = all_features, all_labels

        del all_features
        del all_labels

        self.trainer.strategy.barrier()  # Ensure all ranks are synchronized

        # Broadcast from rank 0
        elements_list = [centroids, centroid_labels]

        if self.trainer.world_size > 1:
            dist.broadcast_object_list(elements_list, src=0)

        centroids, centroid_labels = elements_list
        del elements_list

        assert centroids is not None, "Centroids tensor should not be None"
        assert centroid_labels is not None, "Centroid labels tensor should not be None"

        self.centroids = torch.from_numpy(centroids).to(self.device)
        self.centroid_labels = torch.from_numpy(centroid_labels).to(self.device)
        self.centroid_labels_unique = np.unique(centroid_labels).tolist()

        # Set model back to training mode
        self.model.train(initial_mode)

    def _compute_centroids(self, features: np.ndarray, labels: np.ndarray):
        """Compute centroids and return them (helper for DDP)"""
        centroids_list = []
        centroid_labels_list = []

        num_classes: int = self.trainer.datamodule.num_generators

        for class_idx in range(num_classes):
            class_mask = labels == class_idx
            class_features = features[class_mask]

            if len(class_features) == 0:
                continue

            # Original multi-centroid approach using K-means
            n_centroids = min(self.n_centroids_per_class, len(class_features))
            kmeans = KMeans(n_clusters=n_centroids, random_state=42, n_init=10)
            kmeans.fit(class_features)
            class_centroids = kmeans.cluster_centers_

            centroids_list.append(class_centroids)
            centroid_labels_list.extend([class_idx] * len(class_centroids))

        if centroids_list:
            centroids = np.vstack(centroids_list)
            centroid_labels = np.array(centroid_labels_list)
            return centroids, centroid_labels

        return np.empty((0, features.shape[1])), np.empty(0, dtype=int)

    def _cleanup_centroids(self):
        """Cleanup the kNN classifier"""
        self.centroids = None
        self.centroid_labels = None
        self.centroid_labels_unique = None

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        self._cleanup_centroids()

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()
        self._cleanup_centroids()

    def on_predict_epoch_end(self) -> None:
        super().on_predict_epoch_end()
        self._cleanup_centroids()

    def evaluation_step(
        self,
        batch: Tuple[Tensor, ...],
        batch_idx: int,
        dataloader_idx=0,
        stage: str = "val",
    ):
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

        # Remove outputs for uninitialized classes using self.multiclass_output_mask
        original_n_classes = out.shape[1]
        out = out[
            :, self.multiclass_output_mask
        ]  # (batch_size, only_initialized_classes or 1 for binary)

        # For each element in out, compute the kNN as a score for each class based on distances.
        # For each class, obtain the distance to the nearest centroid and use that distance
        # to compute the score. The lower the distance, the higher the score.
        if (
            self.centroids is None
            or self.centroid_labels is None
            or self.centroid_labels_unique is None
        ):
            raise RuntimeError(
                "Centroids and centroid labels must be set before evaluation."
            )

        # Compute distances to centroids
        distances = torch.cdist(
            out.unsqueeze(0),
            self.centroids,
        ).squeeze(
            0
        )  # shape: (batch_size, num_centroids)

        # Here we compute the score as the inverse of the distance
        # Note: we do this for each class separately, by only considering the nearest centroid
        # for each class, and then using that distance to compute the score.

        # For each class, find the minimum distance to any centroid of that class
        class_scores = torch.full(
            (out.shape[0], original_n_classes), 0.0, device=out.device
        )

        for (
            class_id
        ) in (
            self.centroid_labels_unique
        ):  # Note: class_id is also the index of its output
            class_centroid_mask = self.centroid_labels == class_id
            # distances[:, class_centroid_mask]: shape (batch_size, num_centroids_for_class)
            min_distances, _ = distances[:, class_centroid_mask].min(dim=1)

            # Convert distance to score (higher score for closer centroid)
            # Epsilon is here to avoid div by zero
            class_scores[:, class_id] = 1.0 / (min_distances + 1e-8)

        # Update out to use the class scores
        out = class_scores

        # Convert multiclass to binary if needed
        if self.evaluation_type == "binary":
            out = self.multiclass_to_binary_prediction(out)

        if self.evaluation_type == "binary":
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


def gather_variable_tensors(tensor, dst=0) -> Optional[List[torch.Tensor]]:
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Each rank sends its tensor (can be of different size in first dim)
    gathered = [None for _ in range(world_size)] if rank == dst else None

    dist.gather_object(tensor, gathered, dst=dst)

    if rank == dst:
        return gathered  # List of tensors from all ranks
    else:
        return None


__all__ = ["KNNModel"]
