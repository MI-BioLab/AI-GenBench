from typing import Callable, List, Literal, Optional, Set, Tuple, Union
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
    det_curve,
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
)
from scipy.special import softmax

from evaluation_scripts.multiclass_to_binary_fusion_strategies import *

import datasets
from datasets import load_from_disk
from large_image import LargeImage

datasets.features.features.register_feature(LargeImage, "LargeImage")


def evaluate_predictions(
    dataset_path: Path,    
    predictions_file: Union[str, Path],
    computed_on: str,
    window_id: int,
    threshold: Union[float, Literal["auto"]] = 0.5,
    use_pairing: bool = True,
    unknown_model_predictions_converter: Optional[Callable] = None,
    save_plots_dir: Optional[Union[str, Path]] = None,
    multiclass_to_binary_converter: Optional[
        Callable[[np.ndarray, Set[int]], np.ndarray]
    ] = None,
    multiclass_to_binary_converter_knn: Optional[
        Callable[[np.ndarray, np.ndarray, np.ndarray, Set[int]], np.ndarray]
    ] = None,
    multicrop_predictions_fusion: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    evaluation_task: Literal["binary", "multiclass"] = "binary",
    dataset_split: str = "validation",
):
    if multiclass_to_binary_converter is None:
        multiclass_to_binary_converter = _default_multiclass_to_binary_prediction

    if multicrop_predictions_fusion is None:
        multicrop_predictions_fusion = _default_multicrop_predictions_fusion

    if multiclass_to_binary_converter_knn is None:
        multiclass_to_binary_converter_knn = (
            _default_multiclass_to_binary_prediction_knn
        )

    data = np.load(predictions_file)

    # Get benchmark data
    generator_id_to_name: np.ndarray = data["generator_id_to_name"]
    generator_id_to_name_list: List[str] = generator_id_to_name.tolist()
    pairing: np.ndarray = data["pairing"]
    windows_timeline: List[List[str]] = data["windows_timeline"].tolist()
    set_windows_order: List[Set[int]] = [
        set([generator_id_to_name_list.index(gen_name) for gen_name in window])
        for window in windows_timeline
    ]
    used_multiclass_logits_so_far = _considered_generators(
        "growing", set_windows_order, window_id
    )
    eval_generators_to_consider = _considered_generators(
        computed_on, set_windows_order, window_id
    )

    # Obtain evaluation type
    is_multiclass_training = "crop_multiclass_predictions" in data
    is_knn = "centroids" in data
    # is_multiclass_evaluation = data["fused_scores"].shape[1] > 1

    if evaluation_task == "multiclass" and not is_multiclass_training:
        raise ValueError(
            "The model was not trained on multiclass data, but you are trying to evaluate it as such."
        )

    if is_multiclass_training:
        predictions = data["crop_multiclass_predictions"]
    else:
        predictions = data["crop_scores"]
    crop_generator_ids = data["crop_generator_ids"]
    crop_identifiers = data["crop_identifiers"]

    # Fuse multi-crop scores
    predictions, ground_truth, crop_identifiers = _multicrop_fusion_with_ids(
        crop_scores=predictions,
        crop_labels=crop_generator_ids,
        crop_identifiers=crop_identifiers,
        multicrop_predictions_fusion=multicrop_predictions_fusion,
    )

    # Compute metrics
    if evaluation_task == "binary":
        if is_multiclass_training:
            if is_knn:
                # Predictions based the distance from centroids
                predictions = multiclass_to_binary_converter_knn(
                    predictions,
                    data["centroids"],
                    data["centroid_labels"],
                    used_multiclass_logits_so_far,
                )
            else:
                # If the model was trained on multiclass data, we need to convert predictions to binary
                predictions = multiclass_to_binary_converter(
                    predictions, used_multiclass_logits_so_far
                )

        image_identifiers = data["fused_identifiers"]
        dataset = load_from_disk(dataset_path)[dataset_split]
        conditioning_column: List[str] = dataset["conditioning"][:]
        height_column: List[int] = dataset["height"][:]
        width_column: List[int] = dataset["width"][:]
        conditioning_column = [
            "class" if cond.startswith("class/") else cond
            for cond in conditioning_column
        ]
        conditioning_column = [
            "image" if cond.startswith("image/") else cond
            for cond in conditioning_column
        ]
        pixel_sizes = [
            height * width for height, width in zip(height_column, width_column)
        ]

        conditioning_per_example = [conditioning_column[i] for i in image_identifiers]
        pixel_sizes_per_example = [pixel_sizes[i] for i in image_identifiers]

        return _binary_metrics(
            predictions=predictions,
            ground_truth=ground_truth,
            generator_id_to_name=generator_id_to_name,
            conditioning_per_example=conditioning_per_example,
            pixel_sizes_per_example=pixel_sizes_per_example,
            pairing=pairing,
            generators_to_consider=eval_generators_to_consider,
            window_id=window_id,
            threshold=threshold,
            use_pairing=use_pairing,
            unknown_model_predictions_converter=unknown_model_predictions_converter,
            save_plots_dir=save_plots_dir,
        )
    else:
        return _multiclass_metrics(
            predictions=predictions,
            ground_truth=ground_truth,
            generator_id_to_name=generator_id_to_name,
            pairing=pairing,
            generators_to_consider=eval_generators_to_consider,
            use_pairing=use_pairing,
            unknown_model_predictions_converter=unknown_model_predictions_converter,
        )


def _considered_generators(
    compute_mechanism: str, windows_order: List[Set[int]], current_window: int
) -> Set[int]:
    generators_to_consider = None
    if compute_mechanism == "all":
        generators_to_consider = set()
        for window in windows_order:
            generators_to_consider.update(window)

    elif compute_mechanism == "immediate_future":
        next_window_idx = min(current_window + 1, len(windows_order) - 1)
        generators_to_consider = set(windows_order[next_window_idx])

    elif compute_mechanism == "growing":
        generators_to_consider = set()
        for window in range(current_window + 1):
            generators_to_consider.update(windows_order[window])

    elif compute_mechanism == "growing_whole":
        generators_to_consider = set()
        for window in range(min(current_window + 2, len(windows_order))):
            generators_to_consider.update(windows_order[window])

    elif compute_mechanism == "past":
        generators_to_consider = set()
        for window in range(current_window):
            generators_to_consider.update(windows_order[window])
    else:
        raise ValueError(f"Invalid compute mechanism: {compute_mechanism}")

    return generators_to_consider


def _binary_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    generator_id_to_name: np.ndarray,
    conditioning_per_example: List[str],
    pixel_sizes_per_example: List[int],
    pairing: np.ndarray,
    generators_to_consider: Set[int],
    window_id: int,
    threshold: Union[float, Literal["auto"]],
    use_pairing: bool,
    save_plots_dir: Optional[Union[str, Path]] = None,
    unknown_model_predictions_converter: Optional[Callable] = None,
):
    # Ensure predictions are binary class labels (0 for real, 1 for fake)
    if predictions.ndim > 1:
        if predictions.shape[1] == 1:
            # Already a binary prediction, just stored as (N, 1) -> flatten to be (N,)
            predictions = predictions.flatten()
        elif predictions.shape[1] == 2:
            # A 2-class prediction, stored as (N, 2) -> convert to binary using a softmax, take the fake (class 1) score
            predictions = softmax(predictions, axis=1)
            predictions = predictions[:, 1]
        else:
            if unknown_model_predictions_converter is None:
                raise ValueError(
                    f"Unexpected prediction shape {predictions.shape} for model predictions. "
                    "Please provide a custom `unknown_model_predictions_converter` function to handle this case."
                )
            predictions = unknown_model_predictions_converter(
                predictions, ground_truth, generator_id_to_name
            )

    mask = np.zeros_like(ground_truth, dtype=bool)
    for gen_id in generators_to_consider:
        mask |= ground_truth == gen_id

    if use_pairing:
        # Only take real images (generator 0) paired with considered generators
        for i, generator_id, paired_generator_id in zip(
            range(len(ground_truth)), ground_truth, pairing
        ):
            if generator_id != 0:
                continue

            assert paired_generator_id != -1

            if paired_generator_id in generators_to_consider:
                mask[i] = True
    else:
        mask |= ground_truth == 0

    predictions = predictions[mask]
    ground_truth = ground_truth[mask]
    conditioning_per_example_np = np.array(conditioning_per_example)[mask]
    pixel_sizes_per_example_np = np.array(pixel_sizes_per_example)[mask]

    del conditioning_per_example
    del pixel_sizes_per_example

    # Get number of real and fake images
    real_images = np.sum(ground_truth == 0)
    fake_images = np.sum(ground_truth != 0)

    # Convert ground truth to binary (0 for real, 1 for fake)
    ground_truth_binary = (ground_truth != 0).astype(int)

    # Convert continuous predictions to binary (0 for real, 1 for fake) using a threshold
    if threshold == "auto":
        # Automatically compute the best threshold using the BestBinaryThresholdF1 class
        threshold = compute_best_threshold(predictions, ground_truth_binary)
        print(f"Best threshold found: {threshold:.4f}")

    binary_predictions = (predictions > threshold).astype(int)

    # Calculate overall metrics
    accuracy = accuracy_score(ground_truth_binary, binary_predictions)
    balanced_accuracy = balanced_accuracy_score(ground_truth_binary, binary_predictions)
    precision = precision_score(ground_truth_binary, binary_predictions)
    recall = recall_score(ground_truth_binary, binary_predictions)
    f1 = f1_score(ground_truth_binary, binary_predictions)
    tn, fp, fn, tp = confusion_matrix(ground_truth_binary, binary_predictions).ravel()
    tnr = tn / (tn + fp)
    average_precision = average_precision_score(ground_truth_binary, predictions)
    roc_auc = roc_auc_score(ground_truth_binary, predictions)

    # Compute recall for each class
    recalls = {}
    counts_per_generator = np.unique(ground_truth, return_counts=True)
    print(f"Counts per generator: {counts_per_generator}")
    for generator_id in generators_to_consider:
        if generator_id == 0:
            continue
        mask = ground_truth == generator_id
        if np.sum(mask) == 0:
            recalls[generator_id] = 0.0
            continue
        generator_predictions = binary_predictions[mask]
        generator_ground_truth = ground_truth_binary[mask]
        generator_recall = recall_score(
            generator_ground_truth, generator_predictions, zero_division=0
        )
        recalls[generator_id] = generator_recall

    # Compute recall for each conditioning type
    conditioning_recalls = {}
    for conditioning in np.unique(conditioning_per_example_np).tolist():
        mask = conditioning_per_example_np == conditioning
        mask = mask & (ground_truth != 0)  # Only consider fake images
        if np.sum(mask) == 0:
            continue
        conditioning_predictions = binary_predictions[mask]
        conditioning_ground_truth = ground_truth_binary[mask]
        conditioning_recall = recall_score(
            conditioning_ground_truth, conditioning_predictions, zero_division=0
        )
        conditioning_recalls[conditioning] = conditioning_recall

    # Plot histogram of pixel sizes of wrong images for conditioning.startswith("inpainting")
    inpainting_mask = np.array(
        [cond.startswith("inpainting") for cond in conditioning_per_example_np]
    )
    fake_mask = ground_truth != 0  # Only consider fake images
    wrong_predictions_mask = (
        binary_predictions != ground_truth_binary
    )  # Wrong predictions

    # Combine all masks: inpainting + fake + wrong predictions
    combined_mask = inpainting_mask & fake_mask & wrong_predictions_mask

    if np.sum(combined_mask) > 0:
        wrong_inpainting_pixel_sizes = pixel_sizes_per_example_np[combined_mask]
        print(
            "Min size of wrongly classified inpainting images:",
            np.min(wrong_inpainting_pixel_sizes),
        )
        print(
            "Max size of wrongly classified inpainting images:",
            np.max(wrong_inpainting_pixel_sizes),
        )
        plt.figure(figsize=(10, 6))
        plt.hist(wrong_inpainting_pixel_sizes, bins=50, alpha=0.7, edgecolor="black")
        plt.title("Histogram of Pixel Sizes for Wrongly Classified Inpainting Images")
        plt.xlabel("Pixel Size (Height x Width)")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

        # Add some statistics as text on the plot
        mean_size = np.mean(wrong_inpainting_pixel_sizes)
        median_size = np.median(wrong_inpainting_pixel_sizes)
        plt.axvline(
            mean_size, color="red", linestyle="--", label=f"Mean: {mean_size:.0f}"
        )
        plt.axvline(
            median_size,
            color="orange",
            linestyle="--",
            label=f"Median: {median_size:.0f}",
        )
        plt.legend()

        plt.tight_layout()
        plt.savefig(
            f"eval_plots/inpainting_wrong_pixel_sizes_histogram_{window_id}.png"
        )
        plt.close()

        print(
            f"Saved histogram for {np.sum(combined_mask)} wrongly classified inpainting images"
        )
    else:
        print("No wrongly classified inpainting images found or save_plots_dir is None")

    # Plot histogram of pixel sizes of all fake images for conditioning.startswith("inpainting")

    # Combine all masks: inpainting + fake + wrong predictions
    combined_mask = inpainting_mask & fake_mask

    if np.sum(combined_mask) > 0:
        wrong_inpainting_pixel_sizes = pixel_sizes_per_example_np[combined_mask]

        plt.figure(figsize=(10, 6))
        plt.hist(wrong_inpainting_pixel_sizes, bins=50, alpha=0.7, edgecolor="black")
        plt.title("Histogram of Pixel Sizes for All Inpainting Images")
        plt.xlabel("Pixel Size (Height x Width)")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

        # Add some statistics as text on the plot
        mean_size = np.mean(wrong_inpainting_pixel_sizes)
        median_size = np.median(wrong_inpainting_pixel_sizes)
        plt.axvline(
            mean_size, color="red", linestyle="--", label=f"Mean: {mean_size:.0f}"
        )
        plt.axvline(
            median_size,
            color="orange",
            linestyle="--",
            label=f"Median: {median_size:.0f}",
        )
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"eval_plots/inpainting_all_pixel_sizes_histogram_{window_id}.png")
        plt.close()

        print(
            f"Saved histogram for {np.sum(combined_mask)} wrongly classified inpainting images"
        )
    else:
        print("No wrongly classified inpainting images found or save_plots_dir is None")

    if save_plots_dir is not None:
        # Plot Precision-Recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(
            ground_truth_binary, predictions
        )
        plt.figure()
        plt.plot(recall_curve, precision_curve, marker=".")
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig(Path(save_plots_dir) / f"precision_recall_curve_{window_id}.png")
        plt.close()

        # Plot DET curve
        fpr, fnr, _ = det_curve(ground_truth_binary, predictions)
        plt.figure()
        plt.plot(fpr, fnr, marker=".")
        plt.title("DET Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("False Negative Rate")
        plt.savefig(Path(save_plots_dir) / f"det_curve_{window_id}.png")
        plt.close()

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(ground_truth_binary, predictions)
        plt.figure()
        plt.plot(fpr, tpr, marker=".")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.savefig(Path(save_plots_dir) / f"roc_curve_{window_id}.png")
        plt.close()

    return {
        "n_real_images": real_images,
        "n_fake_images": fake_images,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_precision": average_precision,
        "roc_auc": roc_auc,
        "tnr": tnr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "counts_per_generator": np.unique(ground_truth, return_counts=True),
        "recall_per_generator": recalls,
        "recall_per_conditioning": conditioning_recalls,
    }


def compute_best_threshold(scores: np.ndarray, labels: np.ndarray, metric_option="f1"):
    best_threshold = 0.0
    best_score = -1.0

    for threshold in np.linspace(0, 1, 1001):
        # Convert threshold to a float if needed
        preds_bin = scores >= threshold

        if metric_option == "accuracy":
            score = (preds_bin == labels).mean()
        elif metric_option == "f1":
            score = f1_score(labels, preds_bin, zero_division=0)
        else:
            raise ValueError("Unsupported metric. Use 'accuracy' or 'f1'.")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold


def _multiclass_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    generator_id_to_name: np.ndarray,
    pairing: np.ndarray,
    generators_to_consider: Set[int],
    use_pairing: bool,
    unknown_model_predictions_converter: Optional[Callable] = None,
):
    # Ensure predictions are binary class labels (0 for real, 1 for fake)
    assert predictions.ndim > 1
    if predictions.shape[1] != len(generator_id_to_name):
        if unknown_model_predictions_converter is None:
            raise ValueError(
                f"Unexpected prediction shape {predictions.shape} for model predictions. "
                "Please provide a custom `unknown_model_predictions_converter` function to handle this case."
            )
        predictions = unknown_model_predictions_converter(
            predictions, ground_truth, generator_id_to_name
        )

    mask = np.zeros_like(ground_truth, dtype=bool)
    for gen_id in generators_to_consider:
        mask |= ground_truth == gen_id

    if use_pairing:
        # Only take real images (generator 0) paired with considered generators
        for i, generator_id, paired_generator_id in zip(
            range(len(ground_truth)), ground_truth, pairing
        ):
            if generator_id != 0:
                continue

            assert paired_generator_id != -1

            if paired_generator_id in generators_to_consider:
                mask[i] = True
    else:
        mask |= ground_truth == 0

    predictions = predictions[mask]
    ground_truth = ground_truth[mask]

    # Get number of real and fake images
    real_images = np.sum(ground_truth == 0)
    fake_images = np.sum(ground_truth != 0)

    # Get predicted class labels (argmax of probabilities)
    predicted_labels = np.argmax(predictions, axis=1)

    # Calculate multiclass metrics
    accuracy = accuracy_score(ground_truth, predicted_labels)
    balanced_accuracy = balanced_accuracy_score(ground_truth, predicted_labels)

    # Weighted averages for multiclass
    precision = precision_score(
        ground_truth, predicted_labels, average="weighted", zero_division=0
    )
    recall = recall_score(
        ground_truth, predicted_labels, average="weighted", zero_division=0
    )
    f1 = f1_score(ground_truth, predicted_labels, average="weighted", zero_division=0)
    average_precision = average_precision_score(
        ground_truth, predictions, average="weighted"
    )
    roc_auc = roc_auc_score(
        ground_truth, predictions, multi_class="ovr", average="weighted"
    )

    return {
        "n_real_images": real_images,
        "n_fake_images": fake_images,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_precision": average_precision,
        "roc_auc": roc_auc,
        "counts_per_generator": np.unique(ground_truth, return_counts=True),
    }


def _multicrop_fusion_with_ids(
    crop_scores: np.ndarray,
    crop_labels: np.ndarray,
    crop_identifiers: np.ndarray,
    multicrop_predictions_fusion: Callable[[np.ndarray], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert len(crop_scores) == len(
        crop_identifiers
    ), "Crop scores and crop identifiers must have the same length."
    image_to_score_indices = []
    last_identifier = None

    for crop_idx in range(len(crop_identifiers)):
        crop_identifier = crop_identifiers[crop_idx]
        if crop_identifier != last_identifier:
            image_to_score_indices.append([])
            last_identifier = crop_identifier
        image_to_score_indices[-1].append(crop_idx)

    fused_predictions = []
    fused_labels = []  # May be multi-class as well, it doesn't matter here
    fused_identifiers = []
    for indices in image_to_score_indices:
        fused_labels.append(crop_labels[indices[0]])
        fused_identifiers.append(crop_identifiers[indices[0]])
        if len(indices) == 1:
            # If there's only one crop, just take its score
            fused_predictions.append(crop_scores[indices[0]])
        else:
            # If there are multiple crops, fuse them using the provided multicrop_predictions_fusion function
            fused_predictions.append(multicrop_predictions_fusion(crop_scores[indices]))

    return (
        np.array(fused_predictions),
        np.array(fused_labels, dtype=crop_labels.dtype),
        np.array(fused_identifiers, dtype=crop_identifiers.dtype),
    )


def _default_multicrop_predictions_fusion(image_crop_scores: np.ndarray) -> np.ndarray:

    if len(image_crop_scores.shape) == 1 or image_crop_scores.shape[1] == 1:
        # If scores are 1D or single-channel (binary scores) just return the mean
        return image_crop_scores.mean()
    else:
        # Multiclass scores, of shape (n_crops, n_classes): return the mean across crops
        return image_crop_scores.mean(axis=0)


__all__ = [
    "evaluate_predictions",
]
