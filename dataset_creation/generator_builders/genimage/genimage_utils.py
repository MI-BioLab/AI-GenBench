import io
from pathlib import Path
import subprocess
from typing import Any, Dict, Generator, List, Tuple
from PIL import Image

from dataset_utils.common_utils import (
    RowDict,
    check_max_samples,
    check_needed_samples,
    is_image_valid,
    prepare_image,
)
from dataset_utils.file_extraction import MultiSourceFilesIterator
from generator_builders.deepfake_dataset_builder import AvailableFile
from generator_builders.generator_utils import DatasetContext, process_generator
from generator_builders.imagenet_class_names import IMAGENET_CLASS_ID_TO_STR
from io import BytesIO


def genimage_generator(
    label: int,
    convert_to_jpeg: bool,
    generator_config: Dict[str, Any],
    max_samples,
    lock,
    filelist: List[AvailableFile],
) -> Generator[RowDict, None, None]:
    only_filepaths = [f.file_path for f in filelist]

    with MultiSourceFilesIterator(only_filepaths, batch_size=64) as file_iter:
        for file_info, fpath in zip(filelist, file_iter):
            file_id = file_info.file_id

            generator = file_id.split("/")[1]

            if not check_needed_samples(max_samples, generator, lock):
                continue

            try:
                # Read Bytes
                with open(fpath, "rb") as fh:
                    buf = BytesIO(fh.read())
                img = Image.open(buf)
                img.load()
            except Exception as e:
                print(f"Problematic image: {file_info.file_path} produce error {e}")
                continue

            if not is_image_valid(img):
                print(f"Disarded image: {fpath}")
                continue

            if not check_max_samples(max_samples, generator, lock):
                continue

            # Convert to RGB (and JPEG if required)
            img = prepare_image(img, convert_to_jpeg=convert_to_jpeg)

            split = file_id.split("/")[2]
            context = DatasetContext()
            context.add_values(
                {
                    "class_id": lambda: int(
                        file_id.split("/")[-1].split("_")[
                            4 if generator in {"VQ-Diffusion", "Glide"} else 0
                        ]
                    ),
                    "description": lambda ctx=context: IMAGENET_CLASS_ID_TO_STR[
                        ctx["class_id"]
                    ],
                }
            )
            values = ["conditioning", "description", "positive_prompt"]
            final_values = process_generator(
                generator_config, generator, values, context
            )
            conditioning = final_values["conditioning"]
            description = final_values["description"]
            positive_prompt = final_values["positive_prompt"]
            yield {
                "image": img,
                "height": img.height,
                "width": img.width,
                "label": label,
                "generator": generator,
                "file_id": file_id,
                "description": description,
                "positive_prompt": positive_prompt,
                "negative_prompt": "",
                "conditioning": conditioning,
                "origin_dataset": f"GenImage/{split}",
                "paired_real_images": [],
            }


FOLDER_NAME_TO_GEN_NAME = {
    "ADM": "ADM",
    "glide": "Glide",
    "Midjourney": "Midjourney",
    "stable_diffusion_v_1_4": "Stable Diffusion 1.4",
    "stable_diffusion_v_1_5": "Stable Diffusion 1.5",
    "VQDM": "VQ-Diffusion",
    "wukong": "Wukong",
}

FOLDER_NAME_TO_SUBFOLDER_NAME = {
    "ADM": "imagenet_ai_0508_adm",
    "glide": "imagenet_glide",
    "Midjourney": "imagenet_midjourney",
    "stable_diffusion_v_1_4": "imagenet_ai_0419_sdv4",
    "stable_diffusion_v_1_5": "imagenet_ai_0424_sdv5",
    "VQDM": "imagenet_ai_0419_vqdm",
    "wukong": "imagenet_ai_0424_wukong",
}


def _open_genimage_image(path: str) -> Tuple[Image.Image, str, str, str]:
    if "!" in path:
        zip_path = path.split("!", maxsplit=1)[0]
        internal_path = path.split("!", maxsplit=1)[1]
        with subprocess.Popen(
            f"7zz e '{zip_path}' -so '{internal_path}'",
            shell=True,
            stdout=subprocess.PIPE,
        ) as proc:
            raw_bytes = proc.communicate()[0]

        with io.BytesIO(raw_bytes) as f:
            im = Image.open(f)
            im.load()
            im = im.convert("RGB")

        generator_name = str(Path(zip_path).parent.name)
        internal_path_parts = Path(internal_path.removeprefix("/")).parts[1:]
        internal_path = "/".join(internal_path_parts)

        relative_path = f"{generator_name}/{internal_path}"
    else:
        im = Image.open(path).convert("RGB")
        # Two possibilities (depending on the way the dataset was unzipped):
        # <generator_name>/imagenet_something_something/train/ai/img.jpeg
        # or
        # <generator_name>/train/ai/img.jpeg
        intermediate_parent = Path(path).parent.parent.parent
        if intermediate_parent.name.startswith("imagenet"):
            generator_name = str(intermediate_parent.parent.name)
        else:
            generator_name = str(intermediate_parent.name)

        internal_path = Path(path).relative_to(intermediate_parent)
        relative_path = f"{generator_name}/{internal_path}"

    split = relative_path.split("/")[1]

    return im, FOLDER_NAME_TO_GEN_NAME[generator_name], relative_path, split


__all__ = ["genimage_generator"]


if __name__ == "__main__":
    test_path = "/GenImage/ADM/imagenet_ai_0508_adm/train/ai/0_adm_31.PNG"
    img, generator, relative_path = _open_genimage_image(test_path)
    print(img, generator, relative_path)
