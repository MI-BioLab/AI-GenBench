from collections import OrderedDict
from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional
import warnings

from dataset_utils.common_utils import (
    REAL_IMAGES_LABEL,
    PathAlike,
    RowDictPath,
    sort_paths,
)

from real_builders.real_images_builder import RealImagesBuilder


class RaiseDatasetManager(RealImagesBuilder):

    def __init__(
        self,
        root_path: PathAlike,
        convert_to_jpeg: bool = False,
        tmp_cache_dir: Optional[Path] = None,
    ):
        self.root_path = Path(root_path)
        self.convert_to_jpeg = convert_to_jpeg
        self.tmp_cache_dir = Path(tmp_cache_dir) if tmp_cache_dir is not None else None
        if not self.root_path.exists():
            raise FileNotFoundError(f"Root path does not exist: {self.root_path}")

    def get_prefix(self) -> str:
        return "RAISE"

    def get_builder_name(self) -> str:
        return "RAISE (all)"

    def available_images(self) -> Iterable[str]:
        return list(self.image_paths.keys())

    @property
    def image_paths(self) -> OrderedDict[str, Path]:
        return _raise_image_list(self.get_prefix(), self.root_path)

    def get_image(
        self,
        image_id: str,
    ) -> RowDictPath:
        prefix = self.get_prefix()
        id_prefix, raise_id = image_id.split("/")
        if id_prefix != prefix:
            raise ValueError(f"Invalid image_id: {image_id}")

        fpath = self.image_paths[image_id]

        return {
            "image": fpath,
            "label": REAL_IMAGES_LABEL,
            "generator": "",
            "file_id": image_id,
            "description": "",
            "positive_prompt": "",
            "negative_prompt": "",
            "conditioning": "",
            "origin_dataset": prefix,
            "paired_real_images": [],
        }


@lru_cache(maxsize=1)
def _raise_image_list(prefix: str, folder: Path):
    image_ids_and_paths: OrderedDict[str, Path] = OrderedDict()
    all_available_files = sort_paths(folder.rglob("*.png"))

    for image_path in all_available_files:
        json_path = image_path.with_suffix(".json")

        if not json_path.exists():
            warnings.warn(f"JSON file not found for {image_path}")
            continue

        with json_path.open("r") as f:
            json_data = json.load(f)
            original_img_url = json_data["url"]

        original_img_filename = original_img_url.split("/")[-1]
        original_img_stem = original_img_filename.split(".")[0]

        image_ids_and_paths[f"{prefix}/{original_img_stem}"] = image_path

    assert len(image_ids_and_paths) == len(image_ids_and_paths)

    return image_ids_and_paths


if __name__ == "__main__":
    raise_manager = RaiseDatasetManager(
        "/deepfake/RAISE_all/RAISE_all_TIF",
        convert_to_jpeg=True,
    )

    img_row = raise_manager.get_image("RAISE/r3ac3042ft")  # 00004/00004088.png
    print(img_row)

    print(raise_manager.select_random_images(1, seed=1))


__all__ = ["RaiseDatasetManager"]
