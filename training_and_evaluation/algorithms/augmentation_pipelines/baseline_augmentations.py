from typing import Callable, Literal, Optional, Tuple, Union
import torch
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import InterpolationMode
from albumentations.augmentations import CoarseDropout, GaussNoise

from dataset_loading.fixed_augmentations.format_adapter_dataset import (
    ComposeMixedAugmentations,
)
from lightning_data_modules.augmentation_utils.augmentations_functions import (
    data_augment_blur,
    data_augment_cmp,
    data_augment_rot90,
)


def make_baseline_train_aug(
    model_input_size: Union[int, Tuple[int, int]],
    resize_or_crop_mechanism: Literal["resize", "random_crop", "center_crop", "as_is"],
    resize_interpolation: InterpolationMode = InterpolationMode.BILINEAR,
) -> Union[Callable, Tuple[Callable, Optional[Callable]]]:

    assert resize_or_crop_mechanism in {
        "resize",
        "random_crop",
        "center_crop",
        "as_is",
    }

    if resize_or_crop_mechanism == "resize":
        input_size_adaptation_transform = transforms.Resize(
            size=model_input_size, interpolation=resize_interpolation
        )
    elif resize_or_crop_mechanism == "random_crop":
        input_size_adaptation_transform = transforms.RandomCrop(
            size=model_input_size, pad_if_needed=True, padding_mode="constant"
        )
    elif resize_or_crop_mechanism == "center_crop":
        input_size_adaptation_transform = transforms.CenterCrop(size=model_input_size)
    else:
        # resize_or_crop_mechanism == "as_is"
        input_size_adaptation_transform = transforms.Identity()

    cutout = CoarseDropout(
        num_holes_range=(1, 1),
        hole_height_range=(1, 48),
        hole_width_range=(1, 48),
        fill=128,
        p=0.2,
    )

    gaussian_noise = GaussNoise(p=0.2)

    deterministic_transforms = ComposeMixedAugmentations(
        [
            transforms.RandomApply(
                [
                    transforms.RandomResizedCrop(
                        size=256,
                        scale=(0.08, 1.0),
                        ratio=(0.75, 1.0 / 0.75),
                        interpolation=resize_interpolation,
                    )
                ],
                p=0.2,
            ),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            cutout,
            gaussian_noise,
            transforms.RandomApply(
                [transforms.Lambda(lambda img: data_augment_blur(img, [0.0, 3.0]))],
                p=0.5,
            ),
            transforms.RandomApply(
                [
                    transforms.Lambda(
                        lambda img: data_augment_cmp(
                            img, ["cv2", "pil"], list(range(30, 101))
                        )
                    )
                ],
                p=0.5,
            ),
        ]
    )

    # Rotations, flips and random crops are allowed to be non-deterministic
    non_deterministic_transforms = transforms.Compose(
        [
            transforms.RandomApply(
                [
                    transforms.Lambda(lambda img: data_augment_rot90(img)),
                ],
                p=1.0,
            ),
            transforms.RandomHorizontalFlip(),
            input_size_adaptation_transform,
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    return deterministic_transforms, non_deterministic_transforms


import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


def make_albumentations_train_aug(
    model_input_size: Union[int, Tuple[int, int]],
    resize_or_crop_mechanism: Literal["resize", "random_crop", "center_crop", "as_is"],
    resize_interpolation: int = cv2.INTER_LINEAR,
) -> Callable:
    """
    Create Albumentations-based training augmentation pipeline identical to baseline.

    Args:
        model_input_size: Target size for model input
        resize_or_crop_mechanism: How to adapt input size
        resize_interpolation: Interpolation method for resizing

    Returns:
        Albumentations transform pipeline
    """
    assert resize_or_crop_mechanism in {
        "resize",
        "random_crop",
        "center_crop",
        "as_is",
    }

    # Convert model_input_size to tuple if int
    if isinstance(model_input_size, int):
        target_size = (model_input_size, model_input_size)
    else:
        target_size = model_input_size

    # Build transform list
    transforms_list = []

    # Random resized crop (equivalent to torchvision's RandomResizedCrop)
    transforms_list.append(
        A.RandomResizedCrop(
            height=256,
            width=256,
            scale=(0.08, 1.0),
            ratio=(0.75, 1.0 / 0.75),
            interpolation=resize_interpolation,
            p=0.2,
        )
    )

    # Color jitter
    transforms_list.append(
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8)
    )

    # Random grayscale
    transforms_list.append(A.ToGray(p=0.2))

    # Cutout (CoarseDropout)
    transforms_list.append(
        A.CoarseDropout(
            max_holes=1,
            max_height=48,
            max_width=48,
            min_holes=1,
            min_height=1,
            min_width=1,
            fill_value=128,
            p=0.2,
        )
    )

    # Gaussian noise
    transforms_list.append(A.GaussNoise(p=0.2))

    # Blur augmentation
    transforms_list.append(
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 3), p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ],
            p=0.5,
        )
    )

    # Compression artifacts (JPEG compression)
    transforms_list.append(
        A.ImageCompression(quality_lower=30, quality_upper=100, p=0.5),
    )

    # Rotation (90-degree rotations)
    transforms_list.append(A.RandomRotate90(p=1.0))

    # Horizontal flip
    transforms_list.append(A.HorizontalFlip(p=0.5))

    # Size adaptation based on mechanism
    if resize_or_crop_mechanism == "resize":
        transforms_list.append(
            A.Resize(
                height=target_size[0],
                width=target_size[1],
                interpolation=resize_interpolation,
            )
        )
    elif resize_or_crop_mechanism == "random_crop":
        transforms_list.append(
            A.RandomCrop(height=target_size[0], width=target_size[1], p=1.0)
        )
    elif resize_or_crop_mechanism == "center_crop":
        transforms_list.append(
            A.CenterCrop(height=target_size[0], width=target_size[1], p=1.0)
        )
    # "as_is" means no size transformation

    # Normalization and tensor conversion
    transforms_list.extend(
        [
            A.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    composed_transforms = A.Compose(transforms_list)
    return lambda a: composed_transforms(image=a)["image"]


def make_baseline_val_aug():
    # Note: also check the mandatory_val_preprocessing() method in the datamodule!
    # The mandatory augmentations are applied before the crop and, as the name suggests,
    # are mandatory when using this pipeline in the context of the proposed benchmark.

    before_crop_transforms = transforms.Identity()
    after_crop_transforms = transforms.Compose(
        [
            # Note: resize/crop strategy is defined in the model!
            # Images arrive here already resized to the expected model input size
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    return before_crop_transforms, after_crop_transforms


def make_baseline_test_aug():
    return make_baseline_val_aug()


def make_baseline_predict_aug():
    return make_baseline_val_aug()


__all__ = [
    "make_baseline_train_aug",
    "make_baseline_val_aug",
    "make_baseline_test_aug",
    "make_baseline_predict_aug",
]


def benchmark_training_pipeline(
    dataset_path: str,
    model_input_size: int = 224,
    num_images: int = 10,
    num_warmup: int = 3,
    num_runs: int = 5,
) -> dict:
    """
    Benchmark the training augmentation pipeline.

    Args:
        dataset_path: Path to the dataset directory
        model_input_size: Input size for the model
        num_images: Number of images to process in each run
        num_warmup: Number of warmup runs (excluded from timing)
        num_runs: Number of benchmark runs to average

    Returns:
        Dictionary with timing statistics
    """
    import time
    from typing import List
    import torch
    from datasets import load_from_disk
    from PIL import Image
    import numpy as np
    import random

    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)["train"]

    # Filter dataset for DALL-E 3 generator
    generator_column = dataset["generator"][:]
    dalle3_indices = [
        i
        for i, generator_name in enumerate(generator_column)
        if generator_name == "DALL-E 3"
    ]
    print(f"Found {len(dalle3_indices)} DALL-E 3 images in dataset")

    if len(dalle3_indices) < num_images:
        print(
            f"Warning: Only {len(dalle3_indices)} DALL-E 3 images available, but {num_images} requested"
        )
        num_images = len(dalle3_indices)

    # Deterministic random sampling using fixed seed
    random.seed(42)  # Fixed seed for reproducibility
    test_indices = random.sample(dalle3_indices, num_images)
    test_images = [dataset[i]["image"] for i in test_indices]

    print(f"Selected {len(test_images)} DALL-E 3 images for benchmarking")
    print(f"Sample image shapes: {[np.array(img).shape for img in test_images[:3]]}")

    # Create augmentation pipeline
    all_transforms = make_baseline_train_aug(
        model_input_size=model_input_size, resize_or_crop_mechanism="resize"
    )

    if isinstance(all_transforms, tuple) and len(all_transforms) == 2:
        deterministic_transforms, non_deterministic_transforms = all_transforms
    else:
        deterministic_transforms = all_transforms
        non_deterministic_transforms = None

    def process_batch(images: List[Image.Image]) -> List[torch.Tensor]:
        """Process a batch of images through the augmentation pipeline."""
        results = []
        for img in images:
            # # Convert PIL to tensor format expected by transforms
            img_array = np.array(img)
            # if len(img_array.shape) == 2:  # Grayscale
            #     img_array = np.stack([img_array] * 3, axis=-1)

            # Apply deterministic transforms
            img_array = deterministic_transforms(img_array)

            # Apply non-deterministic transforms
            if non_deterministic_transforms:
                img_array = non_deterministic_transforms(img_array)

            results.append(img_array)

        return results

    # Warmup runs
    print(f"Running {num_warmup} warmup iterations...")
    for _ in range(num_warmup):
        _ = process_batch(test_images)

    # Benchmark runs
    print(f"Running {num_runs} benchmark iterations...")
    execution_times = []

    for run_idx in range(num_runs):
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        start_time = time.perf_counter()
        processed_images = process_batch(test_images)
        end_time = time.perf_counter()

        execution_time = end_time - start_time
        execution_times.append(execution_time)

        print(
            f"Run {run_idx + 1}/{num_runs}: {execution_time:.4f}s "
            f"({execution_time/num_images*1000:.2f}ms per image)"
        )

    # Calculate statistics
    mean_time = np.mean(execution_times)
    std_time = np.std(execution_times)
    min_time = np.min(execution_times)
    max_time = np.max(execution_times)

    # Verify output format
    sample_output = processed_images[0]
    print(f"\nOutput tensor shape: {sample_output.shape}")
    print(f"Output tensor dtype: {sample_output.dtype}")
    print(
        f"Output tensor range: [{sample_output.min():.3f}, {sample_output.max():.3f}]"
    )

    results = {
        "num_images": num_images,
        "num_runs": num_runs,
        "execution_times": execution_times,
        "mean_time_total": mean_time,
        "std_time_total": std_time,
        "min_time_total": min_time,
        "max_time_total": max_time,
        "mean_time_per_image": mean_time / num_images,
        "throughput_images_per_second": num_images / mean_time,
        "output_shape": sample_output.shape,
        "output_dtype": str(sample_output.dtype),
    }

    return results


def print_benchmark_results(results: dict):
    """Print formatted benchmark results."""
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Number of images processed: {results['num_images']}")
    print(f"Number of benchmark runs: {results['num_runs']}")
    print(f"Output tensor shape: {results['output_shape']}")
    print(f"Output tensor dtype: {results['output_dtype']}")
    print("\nTiming Results:")
    print(
        f"  Mean execution time: {results['mean_time_total']:.4f} ± {results['std_time_total']:.4f}s"
    )
    print(f"  Min execution time:  {results['min_time_total']:.4f}s")
    print(f"  Max execution time:  {results['max_time_total']:.4f}s")
    print(f"  Mean time per image: {results['mean_time_per_image']*1000:.2f}ms")
    print(
        f"  Throughput:          {results['throughput_images_per_second']:.1f} images/second"
    )
    print("=" * 60)


def save_comparison_images(
    dataset_path: str,
    output_dir: str = "compare_images",
    model_input_size: int = 224,
    num_images: int = 10,
):
    from datasets import load_from_disk
    import numpy as np

    from algorithms.augmentation_pipelines.baseline_augmentations import (
        make_baseline_train_aug,
    )

    """Save comparison images from both pipelines."""
    import os
    from torchvision.utils import save_image

    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)["train"]

    # Filter dataset for DALL-E 3 generator
    generator_column = dataset["generator"][:]
    dalle3_indices = [
        i
        for i, generator_name in enumerate(generator_column)
        if generator_name == "DALL-E 3"
    ]
    print(f"Found {len(dalle3_indices)} DALL-E 3 images in dataset")

    if len(dalle3_indices) < num_images:
        print(
            f"Warning: Only {len(dalle3_indices)} DALL-E 3 images available, but {num_images} requested"
        )
        num_images = len(dalle3_indices)

    # Deterministic random sampling using fixed seed
    import random

    random.seed(42)  # Fixed seed for reproducibility
    test_indices = random.sample(dalle3_indices, num_images)
    test_images = [dataset[i]["image"] for i in test_indices]

    print(f"Selected {len(test_images)} DALL-E 3 images for comparison")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save original images
    original_dir = os.path.join(output_dir, "original")
    os.makedirs(original_dir, exist_ok=True)

    for i, img in enumerate(test_images):
        img.save(os.path.join(original_dir, f"image_{i:02d}.png"))

    # Create baseline augmentation pipeline
    baseline_det, baseline_non_det = make_baseline_train_aug(
        model_input_size=model_input_size, resize_or_crop_mechanism="resize"
    )

    # Create albumentations pipeline
    albumentations_pipeline = make_albumentations_train_aug(
        model_input_size=model_input_size, resize_or_crop_mechanism="resize"
    )

    # Process and save baseline images
    baseline_dir = os.path.join(output_dir, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)

    print("Processing baseline pipeline images...")
    for i, img in enumerate(test_images):
        img_array = np.array(img)
        # Apply baseline transforms
        img_det = baseline_det(img_array)
        img_final = baseline_non_det(img_det)

        # Save tensor as image
        save_image(img_final, os.path.join(baseline_dir, f"image_{i:02d}.png"))

    # Process and save albumentations images
    albumentations_dir = os.path.join(output_dir, "albumentations")
    os.makedirs(albumentations_dir, exist_ok=True)

    print("Processing albumentations pipeline images...")
    for i, img in enumerate(test_images):
        img_array = np.array(img)
        # Apply albumentations transforms
        img_tensor = albumentations_pipeline(img_array)

        # Save tensor as image
        save_image(img_tensor, os.path.join(albumentations_dir, f"image_{i:02d}.png"))

    print(f"\nComparison images saved to: {output_dir}")
    print(f"  Original images: {original_dir}")
    print(f"  Baseline pipeline: {baseline_dir}")
    print(f"  Albumentations pipeline: {albumentations_dir}")


if __name__ == "__main__":
    import torch
    import numpy as np
    from pathlib import Path
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark training augmentation pipeline or save comparison images"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset directory",
        default="<default_path_here>",
    )
    parser.add_argument(
        "--model-input-size",
        type=int,
        default=224,
        help="Model input size (default: 224)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=30,
        help="Number of images to process (default: 10)",
    )
    parser.add_argument(
        "--num-warmup", type=int, default=3, help="Number of warmup runs (default: 3)"
    )
    parser.add_argument(
        "--num-runs", type=int, default=5, help="Number of benchmark runs (default: 5)"
    )
    parser.add_argument(
        "--save-results", type=str, default=None, help="Path to save results as JSON"
    )
    parser.add_argument(
        "--save-comparison-images",
        action="store_true",
        help="Save comparison images instead of running benchmark",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="compare_images",
        help="Output directory for comparison images (default: compare_images)",
    )

    args = parser.parse_args()

    # Verify dataset path exists
    if not Path(args.dataset_path).exists():
        print(f"Error: Dataset path '{args.dataset_path}' does not exist")
        exit(1)

    if args.save_comparison_images:
        # Save comparison images
        save_comparison_images(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            model_input_size=args.model_input_size,
            num_images=args.num_images,
        )
    else:
        # Run benchmark
        results = benchmark_training_pipeline(
            dataset_path=args.dataset_path,
            model_input_size=args.model_input_size,
            num_images=args.num_images,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
        )

        # Print results
        print_benchmark_results(results)

        # Save results if requested
        if args.save_results:
            import json

            with open(args.save_results, "w") as f:
                # Convert numpy types to native Python for JSON serialization
                json_results = {
                    k: (
                        v.tolist()
                        if isinstance(v, np.ndarray)
                        else float(v) if isinstance(v, np.floating) else v
                    )
                    for k, v in results.items()
                }
                json.dump(json_results, f, indent=2)
            print(f"\nResults saved to: {args.save_results}")
