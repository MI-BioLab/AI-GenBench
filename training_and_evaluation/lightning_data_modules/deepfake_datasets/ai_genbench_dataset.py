from datetime import datetime
from pathlib import Path
from typing import Dict
import warnings
from datasets import load_from_disk, DatasetDict

from ai_gen_bench_metadata.benchmark_generators import BENCHMARK_GENERATORS


class AIGenBenchDatasetLoader:
    def __init__(
        self,
        dataset_path: Path,
    ):
        super().__init__()
        self.dataset_path = dataset_path

    def load_dataset(self) -> DatasetDict:
        loaded_dataset = load_from_disk(self.dataset_path)
        assert isinstance(loaded_dataset, DatasetDict)

        generators_in_benchmark = set(BENCHMARK_GENERATORS.keys())

        for split in loaded_dataset:
            dataset = loaded_dataset[split]
            dataset_len = len(dataset)
            # Add an ID column to the dataset
            dataset = dataset.add_column("ID", list(range(dataset_len)))

            # Only keep rows where the generator is in the benchmark (and also keep rows with generator "", which are the real images)
            generators = dataset["generator"][:]
            indices = [
                i
                for i, gen in enumerate(generators)
                if gen in generators_in_benchmark or gen == ""
            ]

            loaded_dataset[split] = dataset.select(indices)

        if "validation" not in loaded_dataset:
            warnings.warn("Validation dataset not found, will use the training set")
            loaded_dataset["validation"] = loaded_dataset["train"]

        if "test" not in loaded_dataset:
            warnings.warn("Test dataset not found, will use the validation set")
            loaded_dataset["test"] = loaded_dataset["validation"]

        return loaded_dataset

    def get_generators_timeline(self) -> Dict[str, datetime]:
        generators_timeline: Dict[str, datetime] = dict()
        for generator_name, generator_rel_date in BENCHMARK_GENERATORS.items():
            generators_timeline[generator_name] = datetime.strptime(
                generator_rel_date, "%Y-%m-%d"
            )
        return generators_timeline


__all__ = ["AIGenBenchDatasetLoader"]
