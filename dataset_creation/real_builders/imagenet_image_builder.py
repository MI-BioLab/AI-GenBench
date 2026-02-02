from functools import lru_cache
import os
from pathlib import Path
from typing import Iterable, List, Optional

from dataset_utils.common_utils import (
    REAL_IMAGES_LABEL,
    PathAlike,
    RowDictPath,
    saturating_balanced_choice,
)

from real_builders.real_images_builder import RealImagesBuilder


class ImagenetDatasetManager(RealImagesBuilder):

    def __init__(
        self,
        root_path: PathAlike,
        split_name: str,
        convert_to_jpeg: bool = False,
        tmp_cache_dir: Optional[Path] = None,
    ):
        self.root_path = Path(root_path)
        self.split_name = split_name
        self.convert_to_jpeg = convert_to_jpeg
        self.tmp_cache_dir = Path(tmp_cache_dir) if tmp_cache_dir is not None else None

    def get_prefix(self) -> str:
        return f"ILSVRC2012_{self.split_name}"

    def get_builder_name(self) -> str:
        return f"ILSVRC2012 (a.k.a. ImageNet-1k)"

    def available_images(self) -> Iterable[str]:
        return list(_imagenet_image_list(self.get_prefix(), self.root_path))

    def get_image(
        self,
        image_id: str,
    ) -> RowDictPath:
        prefix = self.get_prefix()
        id_prefix = image_id.split("/", maxsplit=1)[0]
        if id_prefix != prefix:
            raise ValueError(f"Invalid image_id: {image_id}")

        image_id = image_id.split("/", maxsplit=1)[1]
        rel_path = image_id.replace("/", os.sep)
        rel_path = f"{rel_path}.JPEG"

        fpath = self.root_path / rel_path

        return {
            "image": fpath,
            "label": REAL_IMAGES_LABEL,
            "generator": "",
            "file_id": f"{prefix}/{image_id}",
            "description": "",
            "positive_prompt": "",
            "negative_prompt": "",
            "conditioning": "",
            "origin_dataset": prefix,
            "paired_real_images": [],
        }

    def select_random_images(
        self,
        num_images: int,
        seed: Optional[int] = None,
        excluding: Optional[Iterable[str]] = None,
        allowed: Optional[Iterable[str]] = None,
    ) -> Optional[List[str]]:
        if allowed is not None and excluding is not None:
            raise ValueError("Only one of 'allowed' or 'excluding' can be specified.")

        available_images = list(self.available_images())
        if allowed is not None:
            allowed = set(allowed)
            available_images = [
                image_id for image_id in available_images if image_id in allowed
            ]

        if excluding is not None:
            excluding = set(excluding)
            available_images = [
                image_id for image_id in available_images if image_id not in excluding
            ]

        if len(available_images) < num_images:
            return available_images

        available_images = sorted(available_images)

        possible_choices = []
        associated_classes = []
        for image_id in available_images:
            class_id = image_id.split("/")[1]
            possible_choices.append(image_id)
            associated_classes.append(class_id)

        return saturating_balanced_choice(
            num_images, possible_choices, associated_classes, seed
        )[0]


@lru_cache(maxsize=4)
def _imagenet_image_list(prefix: str, folder: Path):
    all_available_files = list(folder.rglob("*.JPEG"))

    for image_path in all_available_files:
        rel_path = str(image_path.relative_to(folder))
        image_id = rel_path.split(".")[0]
        image_id = image_id.replace("\\", "/")
        yield f"{prefix}/{image_id}"


if __name__ == "__main__":
    imagenet_manager_train = ImagenetDatasetManager(
        "/datasets/imagenet/train",
        "train",
        convert_to_jpeg=True,
    )

    imagenet_manager_val = ImagenetDatasetManager(
        "/datasets/imagenet/val",
        "val",
        convert_to_jpeg=True,
    )

    img_row = imagenet_manager_train.get_image(
        "ILSVRC2012_train/n01514668/n01514668_222"
    )
    print(img_row)

    print(imagenet_manager_train.select_random_images(1, seed=1))


__all__ = ["ImagenetDatasetManager"]
