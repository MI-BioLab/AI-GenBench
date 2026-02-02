from lightning_data_modules.replay_strategies.replay_strategy import ReplayStrategy
from datasets import Dataset
from typing import Dict, List
import random
from collections import defaultdict


class ClassBalancedReplayStrategy(ReplayStrategy):
    """
    A replay strategy that maintains a class-balanced buffer of samples from previous sliding windows.
    This strategy ensures that the replay buffer contains an equal number of samples from each class
    present.

    The buffer is fixed, which means that, as new classes are introduced, the number of samples per class
    may decrease to accommodate the new classes while keeping the total buffer size constant.
    """

    def __init__(
        self, replay_seed: int, buffer_size: int, label_field: str = "generator"
    ):
        super().__init__()
        self.replay_seed = replay_seed
        self.buffer_size = buffer_size
        self.label_field = label_field

    def make_current_buffer(
        self,
        full_training_dataset: Dataset,
        past_dataset: Dataset,
        sliding_windows_timeline: List[List[str]],
        n_th_window: int,
        generator_name_to_id: Dict[str, int],
    ) -> List[int]:
        """
        Returns the IDs of the elements in the current buffer, given the dataset (current and past),
        the current window index and a random seed to be used for RNG purposes.

        The function should not return data from the current window, only from previous ones. The returned data must cover
        only fake images.

        The function must be deterministic with respect to the inputs. It is expected that, between sliding windows,
        no data will be memorized, so the function must be able to reconstruct the buffer from scratch.
        """

        # Get all generators from past dataset using the new HuggingFace syntax
        generators = past_dataset[self.label_field][:]
        past_ids = past_dataset["ID"][:]

        # Group indices by generator (class), excluding real images (generator == "")
        generator_to_global_ids = defaultdict(list)
        for local_idx, generator in enumerate(generators):
            if generator != "":  # Ignore real images (empty string generator)
                global_id = past_ids[local_idx]
                generator_to_global_ids[generator].append(global_id)

        # If no fake images in past dataset, return empty buffer
        if not generator_to_global_ids:
            return []

        return self._select_balanced_samples(
            generator_to_global_ids, generator_name_to_id
        )

    def _select_balanced_samples(
        self, generator_to_global_ids: dict, generator_name_to_id: Dict[str, int]
    ) -> List[int]:
        """Select balanced samples from scratch for the first buffer."""
        unique_generators = list(generator_to_global_ids.keys())
        n_generators = len(unique_generators)
        samples_per_generator = self.buffer_size // n_generators
        remainder = self.buffer_size % n_generators  # Ignored for the moment

        selected_indices = []

        for i, generator in enumerate(unique_generators):
            available_indices = sorted(generator_to_global_ids[generator])
            random.seed(self.replay_seed + generator_name_to_id[generator])
            random.shuffle(available_indices)

            n_samples_to_select = min(samples_per_generator, len(available_indices))
            selected_generator_indices = available_indices[:n_samples_to_select]
            selected_indices.extend(selected_generator_indices)

        return selected_indices


__all__ = ["ClassBalancedReplayStrategy"]
