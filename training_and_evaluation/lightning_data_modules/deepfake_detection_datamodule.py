from abc import ABC
import random
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    Protocol,
    TYPE_CHECKING,
)
import warnings

from lightning_data_modules.augmentation_utils.image_augmenters import (
    TrainImageAugmenter,
)
from lightning_data_modules.augmentation_utils.image_augmenters import (
    EvaluationImageAugmenter,
)
from lightning_data_modules.replay_strategies.replay_strategy import ReplayStrategy
from training_utils.sliding_windows_experiment_data import SlidingWindowsDefinition
from training_utils.train_data_utils import (
    DatasetWithGeneratorID,
)
from datetime import datetime
from PIL.Image import Image
from torch.utils.data import default_collate
from torch import Tensor

import lightning as L
from datasets import DatasetDict, Dataset
from torch.utils.data import DataLoader


from sklearn.model_selection import train_test_split

from dataset_loading.pytorch_dataset import PyTorchDataset

import copy

if TYPE_CHECKING:
    from algorithms.abstract_model import AbstractBaseDeepfakeDetectionModel


from large_image import LargeImage
import datasets

datasets.features.features.register_feature(LargeImage, "LargeImage")


REAL_IMAGES_LABEL = 0
FAKE_IMAGES_LABEL = 1


class DatasetLoader(Protocol):
    def load_dataset(self) -> DatasetDict: ...

    def get_generators_timeline(self) -> Dict[str, datetime]: ...


class DeepfakeDetectionDatamodule(L.LightningDataModule, ABC):
    def __init__(
        self,
        dataset_loader: DatasetLoader,
        sliding_windows_definition: SlidingWindowsDefinition,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        train_subset_size: Union[int, float] = 0,
        validation_subset_size: Union[int, float] = 0,
        data_management_seed: int = 1337,
        move_data_to_local_storage: bool = False,
        deterministic_augmentations: bool = True,
        augmentation_factor: int = 1,
        augmentations_base_seed: int = 4321,
        replay_strategy: Optional[ReplayStrategy] = None,
    ):
        super().__init__()

        self.dataset_loader = dataset_loader

        self.data_management_seed = data_management_seed

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.per_device_train_batch_size = train_batch_size
        self.per_device_eval_batch_size = eval_batch_size
        self.num_workers = num_workers

        self.train_subset_size = train_subset_size
        self.validation_subset_size = validation_subset_size
        self.move_data_to_local_storage = move_data_to_local_storage

        self.deterministic_augmentations = deterministic_augmentations
        self.augmentation_factor = augmentation_factor
        self.augmentations_base_seed = augmentations_base_seed

        self.train_loader_parameters = {
            "batch_size": self.train_batch_size,
            "num_workers": self.num_workers,
            "shuffle": True,
            "drop_last": True,
            "pin_memory": True,
        }

        self.eval_loader_parameters = {
            "batch_size": self.eval_batch_size,
            "num_workers": self.num_workers,
            "shuffle": False,
            "drop_last": False,
            "collate_fn": multicrop_collate,
            "pin_memory": True,
        }

        if sliding_windows_definition.benchmark_type == "none":
            assert sliding_windows_definition.n_generators_per_window == 0
            assert sliding_windows_definition.current_window in {0, None}
        else:
            assert sliding_windows_definition.benchmark_type in {
                "continual_learning",
                "cumulative",
            }
            assert sliding_windows_definition.n_generators_per_window > 0
            assert (
                sliding_windows_definition.current_window is None
                or sliding_windows_definition.current_window >= 0
            )

        has_replay = replay_strategy is not None
        is_cl = sliding_windows_definition.benchmark_type == "continual_learning"
        assert (
            has_replay == is_cl
        ), "Replay strategy must be provided if and only if the benchmark type is 'continual_learning'."

        self.sliding_windows_definition: SlidingWindowsDefinition = (
            sliding_windows_definition
        )
        self.replay_strategy = replay_strategy

        self.train_dataset: Optional[PyTorchDataset] = None
        self.valid_dataset: Optional[PyTorchDataset] = None
        self.test_dataset: Optional[PyTorchDataset] = None

        self._generator_id_to_name: Optional[List[str]] = None

        self._dataset: Optional[DatasetDict] = None
        self._validation_dataset_splitting: Optional[Dict[int, int]] = None
        self._test_dataset_splitting: Optional[Dict[int, int]] = None

        self._windows_timeline: Optional[List[List[str]]] = None

    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = self.dataset_loader.load_dataset()

        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value
        self._generator_id_to_name = None
        self._validation_dataset_splitting = None
        self._test_dataset_splitting = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    @property
    def generator_id_to_name(self) -> List[str]:
        """
        Returns a list of generator names, where the index corresponds to the generator ID.

        The "real" generator is always at index 0 and is a "" string. The rest of the generators
        are sorted by their names. If the dataset has not been loaded yet, it will be loaded automatically.
        """
        if not self._generator_id_to_name:
            self._generator_id_to_name = self._make_generator_id_to_name()
        return self._generator_id_to_name

    @property
    def generator_name_to_id(self) -> Dict[str, int]:
        """
        Returns a dictionary mapping generator names to their corresponding IDs.
        """
        return {name: i for i, name in enumerate(self.generator_id_to_name)}

    def _make_generator_id_to_name(
        self, dataset: Optional[Dataset] = None
    ) -> List[str]:
        if dataset is None:
            dataset = self.dataset["train"]

        generators: List[str] = sorted(set(dataset["generator"][:]))
        return generators

    @property
    def num_generators(self) -> int:
        """
        Returns the number of generators in the dataset (including the "real" generator).
        """
        return len(self.generator_id_to_name)

    @property
    def validation_dataset_splitting(self) -> Dict[int, int]:
        """
        Returns a dictionary mapping real image IDs to the corresponding fake generator ID.

        This is the static "pairing" used to compute the validation metrics.
        """
        if self._validation_dataset_splitting is None:
            self.setup_val()

        assert self._validation_dataset_splitting is not None
        return self._validation_dataset_splitting

    @property
    def test_dataset_splitting(self) -> Dict[int, int]:
        """
        Returns a dictionary mapping real image IDs to the corresponding fake generator ID.

        This is the static "pairing" used to compute the test metrics.
        """
        if self._test_dataset_splitting is None:
            self.setup_test()

        assert self._test_dataset_splitting is not None
        return self._test_dataset_splitting

    def get_dataset_splitting(self, stage: str) -> Dict[int, int]:
        if stage == "validate":
            return self.validation_dataset_splitting
        elif stage == "test":
            return self.test_dataset_splitting
        else:
            raise ValueError(f"Invalid stage: {stage}")

    @property
    def windows_timeline(self):
        if self._windows_timeline is None:
            self._windows_timeline = self._make_sliding_windows_generators_order()

        return copy.deepcopy(self._windows_timeline)

    @property
    def generators_so_far(self) -> Tuple[Set[str], Set[int]]:
        """
        Returns the list of generators that have been used so far in the current and past sliding windows.

        The first element is a list of generator names, the second element is a list of their corresponding indices
        in the dataset's generator_id_to_name list.

        If the current window is None, it returns all generators.
        If the current window is set, it returns all generators from the current and past windows,
        ensuring that "real" is always included.
        """

        if self.sliding_windows_definition.current_window is None:
            # If no current window is set, return all generators
            generators = set(self.generator_id_to_name)
            generator_indices = set(range(len(self.generator_id_to_name)))
        else:
            # If a current window is set, return all generators from the current and past windows
            generators_so_far = set(
                [self.generator_id_to_name[0]]
            )  # Always include "real"
            generator_indices_so_far = set([0])

            for window in self.windows_timeline[
                : self.sliding_windows_definition.current_window + 1
            ]:
                generators_so_far.update(window)
                generator_indices_so_far.update(
                    self.generator_id_to_name.index(gen) for gen in window
                )

            generators = generators_so_far
            generator_indices = generator_indices_so_far

        return generators, generator_indices

    def setup(self, stage: str):
        # stage values: fit, validate, test, predict
        assert self.trainer is not None

        n_devices = self.trainer.world_size
        self.per_device_train_batch_size = self.train_batch_size // n_devices
        self.per_device_eval_batch_size = self.eval_batch_size // n_devices

        self.train_loader_parameters["batch_size"] = self.per_device_train_batch_size
        self.eval_loader_parameters["batch_size"] = self.per_device_eval_batch_size

        if self.is_global_zero:
            # Print batch size information
            accumulate_grad_batches = self.trainer.accumulate_grad_batches
            overall_batch_size = (
                self.per_device_train_batch_size * n_devices * accumulate_grad_batches
            )

            print(
                f"Using {n_devices} devices, per-device train batch size: {self.per_device_train_batch_size}, "
                f"per-device eval batch size: {self.per_device_eval_batch_size}, "
                f"overall train batch size is: {self.per_device_train_batch_size} * world_size={n_devices} * accumulate_grad_batches={accumulate_grad_batches} = {overall_batch_size}"
            )

        if stage in {"fit"} or stage is None:
            self.setup_train()

        if stage in {"fit", "validate"} or stage is None:
            self.setup_val()

        if stage in {"fit", "validate", "test"} or stage is None:
            self.setup_test()

        if stage == "predict" or stage is None:
            self.setup_predict()

    def setup_train(self):
        train_dataset = self._make_sliding_window_subset()

        if self.train_subset_size > 0:
            if (
                isinstance(self.train_subset_size, float)
                and self.train_subset_size < 1.0
            ):
                n_elements_to_keep = int(self.train_subset_size * len(train_dataset))
            else:
                n_elements_to_keep = int(self.train_subset_size)

            if self.is_global_zero:
                print("Will use a train subset of size", n_elements_to_keep)

            # Select stratified subset
            generator_labels = train_dataset["generator"][:]
            subset_indices, _ = train_test_split(
                list(range(len(train_dataset))),
                train_size=n_elements_to_keep,
                stratify=generator_labels,
            )

            train_dataset = train_dataset.select(subset_indices)

        deterministic_augmentation, non_deterministic_augmentation = self.train_aug()

        self.train_dataset = PyTorchDataset(
            dataset=self._make_dataset(train_dataset),
            augmentation=deterministic_augmentation,
            augmentation_factor=self.augmentation_factor,
            non_deterministic_post_process=non_deterministic_augmentation,
            base_seed=self.augmentations_base_seed,
            enable_deterministic_augmentations=self.deterministic_augmentations,
        )

        train_dataset_labels = train_dataset["label"][:]
        n_real = sum(1 for label in train_dataset_labels if label == 0)
        n_fake = sum(1 for label in train_dataset_labels if label == 1)

        if self.is_global_zero:
            print(
                "Training dataset contains",
                len(self.train_dataset),
                "examples:",
                n_real,
                "real images and",
                n_fake,
                "fake images",
            )

    def setup_val(self):
        validation_dataset = self.dataset["validation"]
        generator_labels = validation_dataset["generator"][:]

        validation_generators: List[str] = sorted(set(generator_labels))

        if validation_generators != self.generator_id_to_name:
            warnings.warn(
                "Validation dataset has different generators than the training dataset"
            )
            self._generator_id_to_name = validation_generators

        n_real = sum([1 for gen in generator_labels if gen == ""])
        n_fake = len(generator_labels) - n_real

        if self.is_global_zero:
            print(
                "Original validation dataset contains",
                n_real,
                "real images and",
                n_fake,
                "fake images",
            )

        real_indices = [i for i, gen in enumerate(generator_labels) if gen == ""]
        fake_indices = [i for i, gen in enumerate(generator_labels) if gen != ""]

        # Balance images (stratified)
        if n_real > n_fake:
            n_to_select = n_fake

            random.seed(self.data_management_seed + 1)
            selected_real_images = random.sample(real_indices, n_to_select)
            selected_fake_images = fake_indices
        elif n_fake > n_real:
            n_to_select = n_real
            selected_real_images = real_indices

            selected_fake_images, _ = train_test_split(
                fake_indices,
                train_size=n_to_select,
                stratify=[generator_labels[i] for i in fake_indices],
                random_state=self.data_management_seed + 1,
            )

        subset_indices = selected_real_images + selected_fake_images
        validation_dataset = validation_dataset.select(subset_indices)
        generator_labels = validation_dataset["generator"][:]
        n_real = sum([1 for gen in generator_labels if gen == ""])
        n_fake = len(generator_labels) - n_real

        if self.is_global_zero:
            print(
                "Balanced validation dataset contains",
                n_real,
                "real images and",
                n_fake,
                "fake images",
            )

        # For debugging purposes
        if self.validation_subset_size > 0:
            if (
                isinstance(self.validation_subset_size, float)
                and self.validation_subset_size < 1.0
            ):
                n_elements_to_keep = int(
                    self.validation_subset_size * len(validation_dataset)
                )
            else:
                n_elements_to_keep = int(self.validation_subset_size)

            if self.is_global_zero:
                print("Will use a validation subset of size", n_elements_to_keep)

            subset_indices, _ = train_test_split(
                list(range(len(validation_dataset))),
                train_size=n_elements_to_keep,
                stratify=generator_labels,
                random_state=self.data_management_seed + 2,
            )
            validation_dataset = validation_dataset.select(subset_indices)
            generator_labels = validation_dataset["generator"][:]

        # Pre-shuffle the dataset to prevent eval-time performance issues
        validation_dataset = validation_dataset.shuffle(
            seed=self.data_management_seed + 3,
            keep_in_memory=True,
        )

        generator_labels = validation_dataset["generator"][:]

        # Pair each fake image with a real image
        real_indices = [i for i, gen in enumerate(generator_labels) if gen == ""]
        fake_indices = [i for i, gen in enumerate(generator_labels) if gen != ""]
        assert (
            abs(len(real_indices) - len(fake_indices)) < 2
        ), f"Number of real and fake images must match: real={len(real_indices)} != fake={len(fake_indices)}"

        random.seed(self.data_management_seed + 4)
        random.shuffle(real_indices)

        validation_dataset_splitting = dict()
        image_identifiers = validation_dataset["ID"]

        for fake_idx, real_idx in zip(fake_indices, real_indices):
            generator_name = generator_labels[fake_idx]
            validation_dataset_splitting[image_identifiers[real_idx]] = (
                self.generator_name_to_id[generator_name]
            )

        self._validation_dataset_splitting = validation_dataset_splitting

        self.valid_dataset = PyTorchDataset(
            dataset=self._make_dataset(validation_dataset),
            augmentation=self.val_aug(),
            augmentation_factor=1,
            base_seed=self.augmentations_base_seed,
            enable_deterministic_augmentations=self.deterministic_augmentations,
        )

        if self.is_global_zero:
            print(
                "Validation dataset contains",
                len(validation_dataset),
                "examples",
            )

    def setup_test(self):
        test_dataset = self.dataset["validation"]
        generator_labels = test_dataset["generator"][:]

        test_generators: List[str] = sorted(set(generator_labels))

        if test_generators != self.generator_id_to_name:
            warnings.warn(
                "Test dataset has different generators than the training dataset"
            )
            self._generator_id_to_name = test_generators

        n_real = sum([1 for gen in generator_labels if gen == ""])
        n_fake = len(generator_labels) - n_real

        if self.is_global_zero:
            print(
                "Original test dataset contains",
                n_real,
                "real images and",
                n_fake,
                "fake images",
            )

        real_indices = [i for i, gen in enumerate(generator_labels) if gen == ""]
        fake_indices = [i for i, gen in enumerate(generator_labels) if gen != ""]

        # Balance images (stratified)
        if n_real > n_fake:
            n_to_select = n_fake

            random.seed(self.data_management_seed + 1)
            selected_real_images = random.sample(real_indices, n_to_select)
            selected_fake_images = fake_indices
        elif n_fake > n_real:
            n_to_select = n_real
            selected_real_images = real_indices

            selected_fake_images, _ = train_test_split(
                fake_indices,
                train_size=n_to_select,
                stratify=[generator_labels[i] for i in fake_indices],
                random_state=self.data_management_seed + 1,
            )

        subset_indices = selected_real_images + selected_fake_images
        test_dataset = test_dataset.select(subset_indices)
        generator_labels = test_dataset["generator"][:]
        n_real = sum([1 for gen in generator_labels if gen == ""])
        n_fake = len(generator_labels) - n_real

        if self.is_global_zero:
            print(
                "Balanced test dataset contains",
                n_real,
                "real images and",
                n_fake,
                "fake images",
            )

        # For debugging purposes
        if False:  # TODO: make configurable option
            subset_indices, _ = train_test_split(
                list(range(len(test_dataset))),
                train_size=512,
                stratify=generator_labels,
                random_state=self.data_management_seed + 2,
            )
            test_dataset = test_dataset.select(subset_indices)
            generator_labels = test_dataset["generator"][:]

        # Pre-shuffle the dataset to prevent eval-time performance issues
        test_dataset = test_dataset.shuffle(
            seed=self.data_management_seed + 3,
            keep_in_memory=True,
        )

        generator_labels = test_dataset["generator"][:]

        # Pair each fake image with a real image
        real_indices = [i for i, gen in enumerate(generator_labels) if gen == ""]
        fake_indices = [i for i, gen in enumerate(generator_labels) if gen != ""]
        assert len(real_indices) == len(fake_indices)

        random.seed(self.data_management_seed + 3)
        random.shuffle(real_indices)

        test_dataset_splitting = dict()
        image_identifiers = test_dataset["ID"][:]
        for fake_idx, real_idx in zip(fake_indices, real_indices):
            generator_name = generator_labels[fake_idx]
            test_dataset_splitting[image_identifiers[real_idx]] = (
                self.generator_name_to_id[generator_name]
            )

        self._test_dataset_splitting = test_dataset_splitting

        self.test_dataset = PyTorchDataset(
            dataset=self._make_dataset(test_dataset),
            augmentation=self.test_aug(),
            augmentation_factor=1,
            base_seed=self.augmentations_base_seed,
            enable_deterministic_augmentations=self.deterministic_augmentations,
        )

        if self.is_global_zero:
            print(
                "Test dataset contains",
                len(self.test_dataset),
                "examples",
            )

    def setup_predict(self):
        predict_dataset = self.dataset["validation"]

        # For debugging purposes
        if False:
            generator_labels = predict_dataset["generator"][:]
            subset_indices, _ = train_test_split(
                list(range(len(predict_dataset))),
                train_size=1000,
                stratify=generator_labels,
            )
            predict_dataset = predict_dataset.select(subset_indices)

        self.prediction_dataset = PyTorchDataset(
            dataset=self._make_dataset(predict_dataset),
            augmentation=self.predict_aug(),
            augmentation_factor=1,
            base_seed=self.augmentations_base_seed,
            enable_deterministic_augmentations=self.deterministic_augmentations,
        )

        if self.is_global_zero:
            print(
                "Prediction (validation) dataset contains",
                len(self.prediction_dataset),
                "examples",
            )

    def train_dataloader(self):
        return self._make_loader(self.train_dataset, self.train_loader_parameters)

    def val_dataloader(self):
        return self._make_loader(self.valid_dataset, self.eval_loader_parameters)

    def test_dataloader(self):
        return self._make_loader(self.test_dataset, self.eval_loader_parameters)

    def predict_dataloader(self):
        return self._make_loader(self.prediction_dataset, self.eval_loader_parameters)

    def _make_loader(self, dataset, dataloader_parameters) -> DataLoader:
        return DataLoader(
            dataset, **self._adapt_num_workers(len(dataset), dataloader_parameters)
        )

    def _adapt_num_workers(self, num_batches: int, dataloader_parameters):
        dataloader_parameters = dict(dataloader_parameters)
        if dataloader_parameters.get("num_workers", 0) > 0:
            num_workers = min(dataloader_parameters["num_workers"], num_batches)
            dataloader_parameters["num_workers"] = num_workers

        if (
            dataloader_parameters.get("persistent_workers", False)
            and dataloader_parameters.get("num_workers", 0) == 0
        ):
            dataloader_parameters["persistent_workers"] = False

        if (
            dataloader_parameters.get("prefetch_factor", None) is not None
            and dataloader_parameters.get("num_workers", 0) == 0
        ):
            dataloader_parameters["prefetch_factor"] = None

        return dataloader_parameters

    def train_aug(self) -> Tuple[Callable, Optional[Callable]]:
        """
        A function that returns the augmentation for the training set.

        This function merges the mandatory preprocessing with the model-specific training augmentation
        provided in model.training_augmentation().

        Important note: it returns 2 Callable values: the first one is the deterministic part of the augmentation,
        the second (optional) one is the non-deterministic part of the augmentation (random rotation, crop, etcetera).

        If you only need purely-deterministic augmentations, return None as the second value.
        """
        model: AbstractBaseDeepfakeDetectionModel = self.trainer.lightning_module

        manadatory_preprocessing = self.mandatory_train_preprocessing()
        model_train_aug: Union[Callable, Tuple[Callable, Optional[Callable]]] = (
            model.train_augmentation()
        )

        if isinstance(model_train_aug, Callable):
            deterministic_part, nondeterministic_part = model_train_aug, None
        else:
            assert len(model_train_aug) in {1, 2}
            if len(model_train_aug) == 1:
                deterministic_part, nondeterministic_part = model_train_aug[0], None
            else:
                deterministic_part, nondeterministic_part = model_train_aug

        assert manadatory_preprocessing is None or isinstance(
            manadatory_preprocessing, Callable
        )
        assert isinstance(deterministic_part, Callable)
        assert nondeterministic_part is None or isinstance(
            nondeterministic_part, Callable
        )

        deterministic_part = TrainImageAugmenter(
            mandatory_preprocessing=manadatory_preprocessing,
            augmentation=deterministic_part,
        )

        return deterministic_part, nondeterministic_part

    def val_aug(self) -> Callable:
        """
        A function that returns the augmentation for the validation set.

        This function merges the mandatory preprocessing with the model-specific augmentation
        provided in model.val_augmentation() and the cropping strategy provided by model.make_val_crops().
        """
        return self._make_eval_aug("val")

    def test_aug(self) -> Callable:
        """
        A function that returns the augmentation for the test set.

        This function merges the mandatory preprocessing with the model-specific augmentation
        provided in model.test_augmentation() and the cropping strategy provided by model.make_test_crops().
        """
        return self._make_eval_aug("test")

    def predict_aug(self) -> Callable:
        """
        A function that returns the augmentation for the prediction set.

        This function merges the mandatory preprocessing with the model-specific augmentation
        provided in model.predict_augmentation() and the cropping strategy provided by model.make_predict_crops().
        """
        return self._make_eval_aug("predict")

    def make_val_crops(
        self, image: Union[Tensor, Image]
    ) -> Union[Tensor, Image, Sequence[Tensor], Sequence[Image]]:
        """
        A function that returns one or multiple crops for each image.

        When returning multiple views/crops for each image, an appropriate predictions/scores
        fusion mechanism should be implemented in the validation_step() method of the model.

        Each returned crop will be augmented with the second callable returned by the the val_aug() function.

        In its default implementation, this function simply calls the model's make_val_crops() method and returns its result.

        Hint: this function can also be used to implement a plain resize.
        """
        model: AbstractBaseDeepfakeDetectionModel = self.trainer.lightning_module

        return model.make_val_crops(image)

    def make_test_crops(
        self, image: Union[Tensor, Image]
    ) -> Union[Tensor, Image, Sequence[Tensor], Sequence[Image]]:
        """
        A function that returns one or multiple crops for each image.

        When returning multiple views/crops for each image, an appropriate predictions/scores
        fusion mechanism should be implemented in the test_step() method of the model.

        Each returned crop will be augmented with the second callable returned by the the test_aug() function.

        Hint: this function can also be used to implement a plain resize.
        """
        model: AbstractBaseDeepfakeDetectionModel = self.trainer.lightning_module

        return model.make_test_crops(image)

    def make_predict_crops(
        self, image: Union[Tensor, Image]
    ) -> Union[Tensor, Image, Sequence[Tensor], Sequence[Image]]:
        """
        A function that returns one or multiple crops for each image.

        When returning multiple views/crops for each image, an appropriate predictions/scores
        fusion mechanism should be implemented in the predict_step() method of the model.

        Each returned crop will be augmented with the second callable returned by the the predict_aug() function.

        Hint: this function can also be used to implement a plain resize.
        """
        model: AbstractBaseDeepfakeDetectionModel = self.trainer.lightning_module

        return model.make_predict_crops(image)

    def mandatory_train_preprocessing(self) -> Optional[Callable]:
        return None

    def mandatory_val_preprocessing(self) -> Optional[Callable]:
        return None

    def mandatory_test_preprocessing(self) -> Optional[Callable]:
        return None

    def mandatory_predict_preprocessing(self) -> Optional[Callable]:
        return None

    def _make_dataset(self, dataset: Dataset) -> DatasetWithGeneratorID:
        return DatasetWithGeneratorID(dataset, self.generator_name_to_id)

    @property
    def available_generators(self) -> Set[str]:
        """
        A list of available synthetic images generators (excluding "real").
        """
        generators = set(self.dataset["train"]["generator"][:])
        generators.remove("")
        return generators

    def _make_sliding_window_subset(self) -> Dataset:
        """
        Creates the subset associated to the "current" sliding window.

        The current subset is given by self.sliding_windows_definition.current_window.
        """
        if self.sliding_windows_definition.benchmark_type == "none":
            return self.dataset["train"]

        windows = self.make_sliding_windows()
        assert self.sliding_windows_definition.current_window is not None
        return self.dataset["train"].select(
            windows[self.sliding_windows_definition.current_window]
        )

    def make_sliding_windows(
        self,
        reference_dataset: Optional[Dataset] = None,
        real_images_pairing_seed: int = 1337,
    ) -> List[List[int]]:
        """
        Returns a list of sliding windows, each window being a list of indices of elements
        in the training dataset.

        The elements included in each window are:
        - for fake images, the ones defined in self.windows_timeline
        - randomly selected real images from the training dataset.
        The number of real images is equal to the number of fake images in the same window.
        Real images are not repeated across different windows.
        """
        if reference_dataset is None:
            reference_dataset = self.dataset["train"]

        if self.sliding_windows_definition.benchmark_type == "none":
            return [list(range(len(reference_dataset)))]

        indices_timeline: List[List[int]] = []

        generator_labels = reference_dataset["generator"][:]
        available_real_images = set(
            [i for i, gen in enumerate(generator_labels) if gen == ""]
        )

        for window_idx, window_generators in enumerate(self.windows_timeline):
            window_generators = set(window_generators)
            fakes_indices = [
                i for i, gen in enumerate(generator_labels) if gen in window_generators
            ]
            n_to_select = len(fakes_indices)
            random.seed(real_images_pairing_seed + window_idx)
            selected_real_images = random.sample(
                sorted(available_real_images), n_to_select
            )
            indices_timeline.append(selected_real_images + fakes_indices)
            available_real_images -= set(selected_real_images)

        if self.sliding_windows_definition.benchmark_type == "cumulative":
            for i in range(1, len(indices_timeline)):
                indices_timeline[i] += indices_timeline[i - 1]

        if self.replay_strategy is None:
            return indices_timeline

        # Include replay samples
        if self.is_global_zero:
            print("Using replay strategy:", self.replay_strategy.__class__.__name__)
        replay_timeline: List[List[int]] = []

        for window_idx in range(len(indices_timeline)):
            if window_idx == 0:
                replay_timeline.append([])
            else:
                # Select the IDs of the replay samples
                current_buffer_ids = self.replay_strategy.make_current_buffer(
                    full_training_dataset=reference_dataset,
                    past_dataset=reference_dataset.select(
                        sum(indices_timeline[:window_idx], [])
                    ),
                    sliding_windows_timeline=self.windows_timeline,
                    n_th_window=window_idx,
                    generator_name_to_id=self.generator_name_to_id,
                )

                # Convert IDs to indices
                current_buffer_indices = dataset_ids_to_indices(
                    reference_dataset, current_buffer_ids
                )
                replay_timeline.append(current_buffer_indices)

        timeline_with_replay = []
        for window_idx in range(len(indices_timeline)):
            timeline_with_replay.append(
                indices_timeline[window_idx] + replay_timeline[window_idx]
            )

        return timeline_with_replay

    def _make_sliding_windows_generators_order(self) -> List[List[str]]:
        """
        Returns a list of sliding windows, each window is a list of generator names.

        Internally used to compute the content of self.windows_timeline.
        """

        available_generators = self.available_generators
        generators_timeline = self.dataset_loader.get_generators_timeline()
        # for generator_name, generator_rel_date in self.dataset_loader.get_generator_timeline().items():
        #     generators_timeline[generator_name] = datetime.strptime(
        #         generator_rel_date, "%Y-%m-%d"
        #     )

        expected_generators = set(generators_timeline.keys())
        assert (
            available_generators == expected_generators
        ), f"Some generators are not in the timeline: {available_generators - expected_generators}, {expected_generators - available_generators}"

        # Available generators ordered by date
        sorted_generators = sorted(
            available_generators, key=lambda x: generators_timeline[x]
        )

        if self.sliding_windows_definition.benchmark_type == "none":
            return [sorted_generators]

        # Assign windows
        windows: List[List[str]] = []
        while len(sorted_generators) > 0:
            current_generators: List[str] = []
            for _ in range(self.sliding_windows_definition.n_generators_per_window):
                if len(sorted_generators) == 0:
                    break
                next_generator = sorted_generators.pop(0)
                next_generator_time = generators_timeline[next_generator]
                current_generators.append(next_generator)
                while (
                    len(sorted_generators) > 0
                    and generators_timeline[sorted_generators[0]] == next_generator_time
                ):
                    current_generators.append(sorted_generators.pop(0))
            windows.append(current_generators)

        return windows

    def _make_eval_aug(self, stage: str) -> "EvaluationImageAugmenter":
        """
        Internal method to create the evaluation augmentation.

        This merges the mandatory preprocessing with the model-defined augmentations and cropping mechanism.
        """
        model: AbstractBaseDeepfakeDetectionModel = self.trainer.lightning_module

        manadatory_preprocessing = getattr(self, f"mandatory_{stage}_preprocessing")()
        model_train_aug: Tuple[Callable, Callable] = getattr(
            model, f"{stage}_augmentation"
        )()

        assert manadatory_preprocessing is None or isinstance(
            manadatory_preprocessing, Callable
        )
        assert isinstance(model_train_aug[0], Callable)
        assert isinstance(model_train_aug[1], Callable)

        # Integrates the cropping strategy defined in methods like make_val_crops, make_test_crops, etc.
        return EvaluationImageAugmenter(
            mandatory_preprocessing=manadatory_preprocessing,
            pre_crop_augmentation=model_train_aug[0],
            post_crop_augmentation=model_train_aug[1],
            cropping_strategy=getattr(self, f"make_{stage}_crops"),
        )

    @property
    def is_global_zero(self) -> bool:
        return self.trainer.is_global_zero if self.trainer is not None else True


def multicrop_collate(batch: List[Sequence[Any]]):
    """
    A general custom collate function that manages multicropping, reusing the default collate function.

    It behaves like the default PyTorch collate function, but it also supports
    multiple crops for each item in the batch. Each item in the batch may have a different number of crops [1, N].

    This function assumes that the first element of each sample is the image (or a list of images),
    and it expands all other elements accordingly.

    The collate function will also add an additional, final, element to the collated data, which is a list of integers
    mapping each crop to the original image index in the batch.

    For instance [0, 0, 0, 1, 2, 2, ...] means that the first 3 crops belong to the first image in the batch,
    the next (1) crop belong to the second image, the next 2 crops belong to the third image, and so on.
    This tensor is useful to keep track of the original image index when evaluating the model on the crops
    and for score/prediction fusion.
    """

    # Initialize lists to hold the collated data for each field
    collated_data = [[] for _ in range(len(batch[0]))]
    crops_to_image_idx = []

    # Iterate through each sample in the batch
    for i, sample in enumerate(batch):
        # The first element is assumed to be the image (or list of images)
        image = sample[0]

        # Check if the image is a list of crops (multicrop case)
        if isinstance(image, list):
            num_crops = len(image)
            # Iterate through each field in the sample
            for field_idx, field in enumerate(sample):
                if field_idx == 0:  # Image field
                    collated_data[field_idx].extend(field)
                else:  # Other fields: extend the list with the field repeated for each crop
                    collated_data[field_idx].extend([field] * num_crops)

            # Keep track of the mapping from crop index to image index
            crops_to_image_idx.extend([i] * num_crops)
        else:
            # If it's a single image, just append the data to each field
            for field_idx, field in enumerate(sample):
                collated_data[field_idx].append(field)

            # Keep track of the mapping from crop index to image index
            crops_to_image_idx.append(i)

    collated_data.append(crops_to_image_idx)

    # Use default_collate to handle the collected lists
    # Create a list of tuples for default_collate
    data_to_collate = list(zip(*collated_data))
    collated_result = default_collate(data_to_collate)

    return collated_result


def dataset_ids_to_indices(dataset: Dataset, ids: List[int]) -> List[int]:
    """
    Converts a list of dataset IDs to their corresponding indices in the dataset.

    Args:
        dataset (Dataset): The dataset containing the data.
        ids (List[int]): A list of dataset IDs to be converted.

    Returns:
        List[int]: A list of indices corresponding to the provided dataset IDs.
    """
    id_to_index = {id_: idx for idx, id_ in enumerate(dataset["ID"][:])}
    return [id_to_index[id_] for id_ in ids]


def dataset_indices_to_ids(dataset: Dataset, indices: List[int]) -> List[int]:
    """
    Converts a list of dataset indices to their corresponding dataset IDs.

    Args:
        dataset (Dataset): The dataset containing the data.
        indices (List[int]): A list of dataset indices to be converted.

    Returns:
        List[int]: A list of dataset IDs corresponding to the provided indices.
    """
    return [dataset["ID"][idx] for idx in indices]


__all__ = ["REAL_IMAGES_LABEL", "FAKE_IMAGES_LABEL", "DeepfakeDetectionDatamodule"]
