from functools import partial
import numpy as np
from scipy.special import softmax
from typing import Set


def _default_multiclass_to_binary_prediction(
    batch_multiclass_scores: np.ndarray,
    generators_to_consider: Set[int],
) -> np.ndarray:
    generators_to_consider = set(generators_to_consider)
    generators_to_consider.add(0)  # Ensure the "real" generator is included
    n_generators_in_benchmark = batch_multiclass_scores.shape[1]

    # Create a mask to select only the generators that have been encountered so far
    multiclass_output_mask = np.zeros((n_generators_in_benchmark,), dtype=np.bool_)
    multiclass_output_mask[list(generators_to_consider)] = (
        True  # Includes the "real" generator
    )

    batch_multiclass_scores = batch_multiclass_scores[:, multiclass_output_mask]

    softmax_scores: np.ndarray = softmax(batch_multiclass_scores, axis=1)

    # Compute fake probability by summing probabilities for classes with index >=1
    binary_softmax_scores = softmax_scores[:, 1:].sum(axis=1)

    # It may happen, due to numerical instability, that the sum may get > 1.0 (like 1.0000001)
    binary_softmax_scores = np.clip(binary_softmax_scores, 0.0, 1.0)

    binary_scores = 1 / (1 + np.exp(-binary_softmax_scores))

    return binary_scores


def _default_multiclass_to_binary_prediction_torch_bf16(
    batch_multiclass_scores1: np.ndarray,
    generators_to_consider: Set[int],
) -> np.ndarray:
    import torch

    generators_to_consider = set(generators_to_consider)
    generators_to_consider.add(0)  # Ensure the "real" generator is included
    batch_multiclass_scores = torch.as_tensor(
        batch_multiclass_scores1, dtype=torch.bfloat16
    )
    n_generators_in_benchmark = batch_multiclass_scores.shape[1]

    # Create a mask to select only the generators that have been encountered so far
    multiclass_output_mask = torch.zeros((n_generators_in_benchmark,), dtype=torch.bool)
    multiclass_output_mask[list(generators_to_consider)] = (
        True  # Includes the "real" generator
    )

    batch_multiclass_scores = batch_multiclass_scores[:, multiclass_output_mask]

    softmax_scores = batch_multiclass_scores.softmax(dim=1)

    # Compute fake probability by summing probabilities for classes with index >=1
    binary_softmax_scores = softmax_scores[:, 1:].sum(dim=1).flatten()

    # It may happen, due to numerical instability, that the sum may get > 1.0 (like 1.0000001)
    binary_softmax_scores = binary_softmax_scores.clamp(0.0, 1.0)

    binary_scores = binary_softmax_scores.sigmoid().flatten()
    binary_scores_np = binary_scores.to(torch.float32).numpy()

    return binary_scores_np


def _multiclass_to_binary_sum_no_sigmoid(
    batch_multiclass_scores: np.ndarray,
    generators_to_consider: Set[int],
) -> np.ndarray:
    """
    Not very useful to compare AUROCs with the default one.
    """
    generators_to_consider = set(generators_to_consider)
    generators_to_consider.add(0)  # Ensure the "real" generator is included
    n_generators_in_benchmark = batch_multiclass_scores.shape[1]

    # Create a mask to select only the generators that have been encountered so far
    multiclass_output_mask = np.zeros((n_generators_in_benchmark,), dtype=np.bool_)
    multiclass_output_mask[list(generators_to_consider)] = (
        True  # Includes the "real" generator
    )

    batch_multiclass_scores = batch_multiclass_scores[:, multiclass_output_mask]

    softmax_scores: np.ndarray = softmax(batch_multiclass_scores, axis=1)

    # Compute fake probability by summing probabilities for classes with index >=1
    binary_softmax_scores = softmax_scores[:, 1:].sum(axis=1)

    # It may happen, due to numerical instability, that the sum may get > 1.0 (like 1.0000001)
    binary_softmax_scores = np.clip(binary_softmax_scores, 0.0, 1.0)

    return binary_softmax_scores


def _multiclass_to_binary_max(
    batch_multiclass_scores: np.ndarray,
    generators_to_consider: Set[int],
) -> np.ndarray:
    """
    Not very useful to compare AUROCs with the default one.
    """
    generators_to_consider = set(generators_to_consider)
    generators_to_consider.add(0)  # Ensure the "real" generator is included
    n_generators_in_benchmark = batch_multiclass_scores.shape[1]

    # Create a mask to select only the generators that have been encountered so far
    multiclass_output_mask = np.zeros((n_generators_in_benchmark,), dtype=np.bool_)
    multiclass_output_mask[list(generators_to_consider)] = (
        True  # Includes the "real" generator
    )

    batch_multiclass_scores = batch_multiclass_scores[:, multiclass_output_mask]

    softmax_scores: np.ndarray = softmax(batch_multiclass_scores, axis=1)

    binary_score = softmax_scores[:, 1:].max(axis=1)

    return binary_score


def _default_multiclass_to_binary_prediction_knn(
    batch_multiclass_scores,
    centroids,
    centroid_labels,
    generators_to_consider,
    fusion_strategy="sum",
):
    from scipy.spatial.distance import cdist
    from scipy.special import softmax

    generators_to_consider = set(generators_to_consider)
    generators_to_consider.add(0)  # Ensure the "real" generator is included
    n_generators_in_benchmark = batch_multiclass_scores.shape[1]

    # Create a mask to select only the generators that have been encountered so far
    multiclass_output_mask = np.zeros((n_generators_in_benchmark,), dtype=np.bool_)
    multiclass_output_mask[list(generators_to_consider)] = (
        True  # Includes the "real" generator
    )

    batch_multiclass_scores = batch_multiclass_scores[:, multiclass_output_mask]

    # batch_multiclass_scores = torch.as_tensor(batch_multiclass_scores)
    # centroids = torch.as_tensor(centroids)
    # centroid_labels = torch.as_tensor(centroid_labels)

    distances = cdist(
        batch_multiclass_scores,
        centroids,
    )

    # For each class, find the minimum distance to any centroid of that class
    class_scores = np.full(
        (batch_multiclass_scores.shape[0], n_generators_in_benchmark), 0.0
    )
    centroid_labels_unique = np.unique(centroid_labels).tolist()

    for (
        class_id
    ) in centroid_labels_unique:  # Note: class_id is also the index of its output
        class_centroid_mask = centroid_labels == class_id
        # distances[:, class_centroid_mask]: shape (batch_size, num_centroids_for_class)
        min_distances = distances[:, class_centroid_mask].min(axis=1)

        # Convert distance to score (higher score for closer centroid)
        # Epsilon is here to avoid div by zero
        class_scores[:, class_id] = 1.0 / (min_distances + 1e-8)

    # Softmax
    softmax_scores = softmax(class_scores, axis=1)

    binary_scores: np.ndarray
    if fusion_strategy == "sum":
        binary_scores = np.sum(softmax_scores[:, 1:], axis=1)
        # flatten
    elif fusion_strategy == "max":
        binary_scores = np.max(softmax_scores[:, 1:], axis=1)
    else:
        raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")

    # flatten
    binary_scores = binary_scores.flatten()

    binary_scores = np.clip(binary_scores, 0.0, 1.0)

    return binary_scores


_default_multiclass_to_binary_prediction_knn_max = partial(
    _default_multiclass_to_binary_prediction_knn, fusion_strategy="max"
)


__all__ = [
    "_default_multiclass_to_binary_prediction",
    "_default_multiclass_to_binary_prediction_torch_bf16",
    "_multiclass_to_binary_sum_no_sigmoid",
    "_multiclass_to_binary_max",
    "_default_multiclass_to_binary_prediction_knn",
    "_default_multiclass_to_binary_prediction_knn_max",
]
