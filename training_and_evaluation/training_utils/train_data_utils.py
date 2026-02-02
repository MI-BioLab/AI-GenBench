from typing import Dict, List, Tuple
from torch.utils.data import Dataset
import datasets


class DatasetWithGeneratorID(Dataset):

    def __init__(self, dataset: datasets.Dataset, generator_name_to_id: Dict[str, int]):
        self.dataset: datasets.Dataset = dataset

        self.dataset = self.dataset.select_columns(
            ["image", "label", "generator", "ID"]
        )

        self.generator_name_to_id = generator_name_to_id

        self._dataset_contains_real_images = self._contains_real_images(dataset)

        if self._dataset_contains_real_images:
            assert (
                self.generator_name_to_id[""] == 0
            ), 'The "real" generator should have generator ID 0 and should be named "" (empty string)'

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        elem = self.dataset[index]
        label = elem["label"]
        generator_name = elem["generator"]
        identifier = elem["ID"]

        generator_id = self.generator_name_to_id[generator_name]

        if label == 0:
            assert generator_name == ""
            assert generator_id == 0
        else:
            assert generator_name != ""
            assert generator_id != 0

        if isinstance(elem, dict):
            image = elem["image"]
        else:
            image = elem[0]

        return image, label, generator_id, identifier

    def __getattr__(self, name: str):
        if name == "__getitems__":
            raise AttributeError()
        return getattr(self.dataset, name)

    @staticmethod
    def _contains_real_images(dataset):
        labels = dataset["label"]
        return 0 in labels


def split_dataset_by_generator(
    dataset: datasets.Dataset,
) -> Tuple[List[str], List[datasets.Dataset]]:
    dataset_generators = dataset["generator"][:]
    generators: List[str] = sorted(set(dataset_generators))

    datasets = []
    # Note: real is the empty string in the dataset
    # And it's always the first one in generators
    for gen in generators:
        indices = []
        for i, g in enumerate(dataset_generators):
            if g == gen or (gen == "real" and g == ""):
                indices.append(i)

        datasets.append(dataset.select(indices))

    return generators, datasets


__all__ = [
    "DatasetWithGeneratorID",
    "split_dataset_by_generator",
]
