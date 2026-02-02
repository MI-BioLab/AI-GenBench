from typing import Callable, Literal, Optional, Tuple, Union
import torch
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import InterpolationMode
from albumentations.augmentations import GaussNoise

from dataset_loading.fixed_augmentations.format_adapter_dataset import (
    ComposeMixedAugmentations,
)
from lightning_data_modules.augmentation_utils.augmentations_functions import (
    data_augment_cmp,
    data_augment_rot90,
)
from lightning_data_modules.augmentation_utils.random_resized_crop_variable import (
    RandomResizedCropVariable,
)


def make_soft_train_aug(
    model_input_size: Union[int, Tuple[int, int]],
    resize_or_crop_mechanism: Literal["resize", "random_crop", "center_crop", "as_is"],
    resize_interpolation: InterpolationMode = InterpolationMode.BILINEAR,
) -> Union[Callable, Tuple[Callable, Optional[Callable]]]:
    """
    Uses the same augmentations used in the mandatory evaluation preprocessing, plus adding
    the usual flipping and rotations. Evaluation preprocessing is defined in the
    `mandatory_val_preprocessing()` method in the datamodule and those augmentations are a bit
    softer that the ones defined in `baseline_agumentations.py`. However, they feature
    multiple compression steps which are not found in the baseline augmentations.
    """

    assert resize_or_crop_mechanism in {
        "resize",
        "random_crop",
        "center_crop",
        "as_is",
    }

    deterministic_transforms = ComposeMixedAugmentations(
        [
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


__all__ = [
    "make_soft_train_aug",
]
