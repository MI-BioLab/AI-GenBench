from lightning_data_modules.replay_strategies.replay_strategy import ReplayStrategy
from datasets import Dataset
from typing import Dict, List
import random
from collections import defaultdict


class MostRecentReplayStrategy(ReplayStrategy):
    """
    A replay strategy that maintains a buffer of samples from previous sliding windows.
    This strategy will keep, for each generator, a number of samples which is inversely
    proportional to the moment it was introduced.

    The buffer size is fixed, which means that, as new classes are introduced, the number of samples per class
    will decrease. In particular, the number of classes of a generator will be equal to the number of initially
    selected samples (half of the buffer), and then computed using (1/2)^i, where i is the number of sliding windows
    since the generator was introduced. The buffer size will follow the formula "2 - 2^(-n)", whose limit is 2.
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
        generator_to_global_ids: Dict[str, List[int]] = defaultdict(list)
        for local_idx, generator in enumerate(generators):
            if generator != "":  # Ignore real images (empty string generator)
                global_id = past_ids[local_idx]
                generator_to_global_ids[generator].append(global_id)

        # If no fake images in past dataset, return empty buffer
        if not generator_to_global_ids:
            return []

        return self._select_samples(
            generator_to_global_ids,
            generator_name_to_id,
            sliding_windows_timeline,
            n_th_window,
        )

    def _select_samples(
        self,
        generator_to_global_ids: Dict[str, List[int]],
        generator_name_to_id: Dict[str, int],
        sliding_windows_timeline: List[List[str]],
        sliding_window_idx: int,
    ) -> List[int]:
        assert set(generator_to_global_ids.keys()).issubset(
            set(generator_name_to_id.keys())
        )

        selected_indices = []
        n_elements_per_generator = self._compute_n_elements_per_generator(
            sliding_windows_timeline, sliding_window_idx
        )

        remainder = self._compute_remainder(
            n_elements_per_generator, sliding_windows_timeline, sliding_window_idx
        )

        n_elements_per_generator = self._apply_remainder(
            n_elements_per_generator,
            sliding_windows_timeline,
            sliding_window_idx,
            remainder,
        )

        for generator, generator_samples in n_elements_per_generator.items():
            generator_samples = self._pick_samples_from_generator(
                available_indices=sorted(generator_to_global_ids[generator]),
                n_samples_to_select=generator_samples,
                generator_name_to_id=generator_name_to_id,
                generator=generator,
            )
            selected_indices.extend(generator_samples)

        return selected_indices

    def _compute_n_elements_per_generator(
        self, sliding_windows_timeline: List[List[str]], sliding_window_idx: int
    ) -> Dict[str, int]:
        n_elements_per_generator = dict()
        for window_idx in range(sliding_window_idx):
            current_window_generators = sliding_windows_timeline[window_idx]
            n_current_generators = len(current_window_generators)

            sliding_windows_so_far = sliding_window_idx - window_idx
            n_elements_to_consider = int(
                (self.buffer_size / (2 ** (sliding_windows_so_far - 1)))
                / n_current_generators
            )
            n_elements_per_generator.update(
                {gen: n_elements_to_consider for gen in current_window_generators}
            )

        return n_elements_per_generator

    def _compute_remainder(
        self,
        n_elements_per_generator: Dict[str, int],
        sliding_windows_timeline: List[List[str]],
        sliding_window_idx: int,
    ) -> int:
        remainder = (self.buffer_size * 2) - sum(n_elements_per_generator.values())
        return remainder

    def _apply_remainder(
        self,
        n_elements_per_generator: Dict[str, int],
        sliding_windows_timeline: List[List[str]],
        sliding_window_idx: int,
        remainder: int,
    ) -> Dict[str, int]:
        n_elements_per_generator = n_elements_per_generator.copy()
        if remainder > 0:
            # Distribute the remainder starting from the most recent generators
            for window_idx in range(sliding_window_idx - 1, -1, -1):
                if remainder == 0:
                    break
                current_window_generators = sliding_windows_timeline[window_idx]
                if remainder < len(current_window_generators):
                    break
                for gen in current_window_generators:
                    assert gen in n_elements_per_generator
                    n_elements_per_generator[gen] += 1
                remainder -= len(current_window_generators)
        return n_elements_per_generator

    def _pick_samples_from_generator(
        self,
        available_indices: List[int],
        n_samples_to_select: int,
        generator_name_to_id: Dict[str, int],
        generator: str,
    ) -> List[int]:
        random.seed(self.replay_seed + generator_name_to_id[generator])
        random.shuffle(available_indices)

        if n_samples_to_select > len(available_indices):
            raise ValueError(
                f"Not enough samples for generator {generator}. "
                f"Requested {n_samples_to_select}, but only {len(available_indices)} available."
            )
        selected_indices = available_indices[:n_samples_to_select]
        return selected_indices


__all__ = ["MostRecentReplayStrategy"]
