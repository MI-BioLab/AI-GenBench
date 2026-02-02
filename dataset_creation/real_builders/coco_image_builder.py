from collections import OrderedDict
from functools import partial
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import copy
from dataset_utils.common_utils import (
    DEEPFAKE_DATASET_FEATURES,
    REAL_IMAGES_LABEL,
    PathAlike,
    RowDictPath,
    prepare_image,
)
from datasets import Dataset
from PIL import Image

from real_builders.real_images_builder import RealImagesBuilder


class COCODatasetManager(RealImagesBuilder):

    def __init__(
        self,
        root_path: PathAlike,
        captions_json: PathAlike,
        split_name: str,
        convert_to_jpeg: bool = False,
        num_proc: int = 1,
        tmp_cache_dir: Optional[Path] = None,
    ):
        self.root_path = Path(root_path)
        self.captions_json_path = Path(captions_json)
        self.split_name = split_name
        self.convert_to_jpeg = convert_to_jpeg
        self.num_proc = num_proc
        self.tmp_cache_dir = Path(tmp_cache_dir) if tmp_cache_dir is not None else None

        self._captions_dict = None

        self._available_images = None

    def check_images_availability(
        self, searched_images: Iterable[int]
    ) -> Tuple[Set[int], Set[int]]:
        searched_images = set(searched_images)
        found_imgs = set(self.captions_dict.keys()).intersection(searched_images)
        missing_images = searched_images - found_imgs

        return found_imgs, missing_images

    def get_img_data(
        self, img_id: int
    ) -> Optional[Tuple[Path, Tuple[Dict[Any, Any], List[str]], str]]:
        if img_id in self.captions_dict:
            return self.root_path, self.captions_dict[img_id], self.split_name
        return None

    @property
    def captions_dict(self) -> OrderedDict[int, Tuple[Dict[Any, Any], List[str]]]:
        if self._captions_dict is None:
            self._captions_dict = self._load_captions(self.captions_json_path)
        return self._captions_dict

    def _load_captions(
        self, caption_path: PathAlike
    ) -> Dict[int, Tuple[Dict[Any, Any], List[str]]]:
        result: Dict[int, Tuple[Dict[Any, Any], List[str]]] = OrderedDict()
        with open(caption_path, "r") as f:
            json_content = json.load(f, object_pairs_hook=OrderedDict)

        for img_def in json_content["images"]:
            result[img_def["id"]] = [img_def, list()]

        all_annotations = json_content["annotations"]
        all_annotations.sort(key=lambda x: x["id"])

        for caption in all_annotations:
            result[caption["image_id"]][1].append(caption["caption"])

        for img_id in result:
            result[img_id] = tuple(result[img_id])

        result = OrderedDict(sorted(result.items(), key=lambda x: x[0]))

        return result

    def get_prefix(self) -> str:
        return f"COCO2017_{self.split_name}"

    def get_builder_name(self) -> str:
        return f"COCO2017"

    def available_images(self) -> Iterable[str]:
        if self._available_images is not None:
            return list(self._available_images)

        result = []
        prefix = self.get_prefix()
        for image_entry in self.captions_dict.values():
            image_def = image_entry[0]
            image_id: int = image_def["id"]
            image_rel_path: str = image_def["file_name"]
            abs_path = self.root_path / image_rel_path
            if abs_path.exists():
                result.append(f"{prefix}/{image_id}")

        self._available_images = result
        return list(result)

    def get_image(
        self,
        image_id: str,
    ) -> Optional[RowDictPath]:
        prefix = self.get_prefix()
        id_prefix = image_id.split("/")[0]
        if id_prefix != prefix:
            raise ValueError(f"Invalid image_id: {image_id}")

        img_id = int(image_id.split("/")[1])
        root_path, img_def, _ = self.get_img_data(img_id)
        img_metadata, img_captions = img_def

        fpath = root_path / img_metadata["file_name"]

        description = " ".join(img_captions)

        return {
            "image": fpath,
            "label": REAL_IMAGES_LABEL,
            "generator": "",
            "file_id": f"{prefix}/{img_id}",
            "description": description,
            "positive_prompt": "",
            "negative_prompt": "",
            "conditioning": "",
            "origin_dataset": prefix,
            "paired_real_images": [],
        }


if __name__ == "__main__":
    coco_manager_val = COCODatasetManager(
        (
            "/datasets/coco/val2017",
            "/datasets/coco/captions_val2017.json",
            "val",
        ),
        convert_to_jpeg=True,
        num_proc=1,
    )

    check_exist = coco_manager_val.check_images_availability([548267, 79408, 5])
    print(check_exist)


__all__ = ["COCODatasetManager"]
