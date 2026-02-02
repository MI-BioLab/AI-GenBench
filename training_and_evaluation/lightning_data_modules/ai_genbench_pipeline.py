from typing import Callable, Optional, Union

import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import InterpolationMode

from dataset_loading.fixed_augmentations.format_adapter_dataset import (
    ComposeMixedAugmentations,
)

from lightning_data_modules.deepfake_detection_datamodule import (
    DatasetLoader,
    DeepfakeDetectionDatamodule,
)
from albumentations.augmentations import GaussNoise
from lightning_data_modules.augmentation_utils.augmentations_functions import (
    data_augment_cmp,
)
from lightning_data_modules.augmentation_utils import (
    RandomCropIfLarge,
    RandomResizedCropVariable,
)
from lightning_data_modules.replay_strategies import ReplayStrategy
from training_utils.sliding_windows_experiment_data import SlidingWindowsDefinition


class AIGenBenchPipeline(DeepfakeDetectionDatamodule):

    def __init__(
        self,
        dataset_loader: DatasetLoader,
        train_batch_size: int,
        eval_batch_size: int,
        num_workers: int,
        sliding_windows_definition: SlidingWindowsDefinition,
        train_subset_size: Union[int, float] = 0,
        validation_subset_size: Union[int, float] = 0,
        move_data_to_local_storage: bool = False,
        deterministic_augmentations: bool = True,
        augmentation_factor: int = 1,
        augmentations_base_seed: int = 4321,
        maximum_processing_size: Optional[int] = None,
        replay_strategy: Optional[ReplayStrategy] = None,
    ):
        """
        Args:
            dataset_loader (DatasetLoader): The dataset loader that implements
                the `load_dataset()` and `get_generators_timeline()` methods.
            train_batch_size (int): Batch size for training.
            eval_batch_size (int): Batch size for evaluation.
            num_workers (int): Number of workers for data loading.
            train_subset_size (Union[int, float]): Size of the training subset.
            validation_subset_size (Union[int, float]): Size of the validation subset.
            sliding_windows_definition (SlidingWindowsDefinition): Sliding windows definition.
            move_data_to_local_storage (bool): Whether to move data to local storage.
            deterministic_augmentations (bool): Whether to use deterministic augmentations.
            augmentation_factor (int): Factor for augmentations.
            augmentations_base_seed (int): Base seed for augmentations.
            maximum_processing_size (Optional[int]): Maximum processing size for images. Defaults to None.
                A default value of 1080 is recommended (and already set in the YAML files), but otherwise
                it's not the default value for this parameter! When not None, the image will be
                random-cropped (at training time) or center-cropped (at evaluation time)
                to maximum_processing_size X maximum_processing_size (without padding) before being passed
                to the augmentation pipeline. Images whose sides are smaller than maximum_processing_size
                are not affected. This is done to avoid slowing down the augmentation pipeline when using
                large images (which often happens in the field of deepfake / synthetic images detection).
            replay_strategy (Optional[ReplayStrategy]): The replay strategy to use. Must be None
                if not using continual learning with replay. Must be a ReplayStrategy object otherwise.
        """
        super().__init__(
            dataset_loader=dataset_loader,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            train_subset_size=train_subset_size,
            validation_subset_size=validation_subset_size,
            sliding_windows_definition=sliding_windows_definition,
            move_data_to_local_storage=move_data_to_local_storage,
            deterministic_augmentations=deterministic_augmentations,
            augmentation_factor=augmentation_factor,
            augmentations_base_seed=augmentations_base_seed,
            replay_strategy=replay_strategy,
        )

        self.maximum_processing_size = maximum_processing_size

    def mandatory_train_preprocessing(self) -> Optional[Callable]:
        crop_if_big = None
        if self.maximum_processing_size is not None:
            crop_if_big = RandomCropIfLarge(
                threshold=(self.maximum_processing_size, self.maximum_processing_size)
            )

        return crop_if_big

    def mandatory_val_preprocessing(self) -> Optional[Callable]:
        crop_if_big = transforms.Identity()
        if self.maximum_processing_size is not None:
            crop_if_big = RandomCropIfLarge(
                threshold=(self.maximum_processing_size, self.maximum_processing_size),
                force_central_crop=True,
            )

        # Note: ComposeMixedAugmentations is used when needing to mix torchvision
        # and albumentations augmentations.
        return ComposeMixedAugmentations(
            [
                crop_if_big,
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomApply(
                    [
                        transforms.Lambda(
                            lambda img: data_augment_cmp(
                                img, ["cv2", "pil"], list(range(50, 101))
                            )
                        )
                    ],
                    p=0.50,
                ),
                GaussNoise(std_range=(0.015, 0.075), p=0.3),
                transforms.RandomApply(
                    [
                        transforms.Lambda(
                            lambda img: data_augment_cmp(
                                img, ["cv2", "pil"], list(range(50, 101))
                            )
                        )
                    ],
                    p=0.20,
                ),
                transforms.RandomApply(
                    [
                        RandomResizedCropVariable(
                            min_size=256,
                            max_size=512,
                            scale=(0.5, 1.0),
                            ratio=(0.9, 1 / 0.9),
                            interpolation_options=[
                                InterpolationMode.BILINEAR,
                                InterpolationMode.BICUBIC,
                                InterpolationMode.LANCZOS,
                            ],
                        )
                    ],
                    p=0.25,
                ),
                transforms.RandomApply(
                    [
                        transforms.Lambda(
                            lambda img: data_augment_cmp(
                                img, ["cv2", "pil"], list(range(50, 101))
                            )
                        )
                    ],
                    p=0.10,
                ),
            ]
        )

    def mandatory_test_preprocessing(self) -> Optional[Callable]:
        return self.mandatory_val_preprocessing()

    def mandatory_predict_preprocessing(self) -> Optional[Callable]:
        return self.mandatory_val_preprocessing()


__all__ = ["AIGenBenchPipeline"]
