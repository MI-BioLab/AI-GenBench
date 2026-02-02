from abc import ABC, abstractmethod
from pathlib import Path
import datasets
from typing import Dict, List


class ReplayStrategy(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def make_current_buffer(
        self,
        full_training_dataset: datasets.Dataset,
        past_dataset: datasets.Dataset,
        sliding_windows_timeline: List[List[str]],
        n_th_window: int,
        generator_name_to_id: Dict[str, int],
    ) -> List[int]:
        """
        Returns the indices of the elements in the current buffer, given the dataset (current and past),
        the current window index and a random seed to be used for RNG purposes.

        The function should not return data from the current window, only from previous ones. The returned data must cover
        only fake images.

        The function must be deterministic with respect to the inputs. It is expected that, between sliding windows,
        no data will be memorized, so the function must be able to reconstruct the buffer from scratch.

        The generator_name_to_id dictionary maps generator names (strings) to unique integer IDs. Those can be
        used to ensure different shuffling for different classes, while keeping the same base seed across classes.
        """
        ...


__all__ = ["ReplayStrategy"]


def _test_correctness():
    LOCAL_DATASET_PATH = "..."

    from lightning_data_modules.ai_genbench_pipeline import AIGenBenchPipeline
    from lightning_data_modules.deepfake_datasets import (
        AIGenBenchDatasetLoader,
    )
    from training_utils.sliding_windows_experiment_data import SlidingWindowsDefinition
    from lightning_data_modules.replay_strategies import (
        ClassBalancedReplayStrategy,
        MostRecentReplayStrategy,
        MostRecentHarmonicReplayStrategy,
        WithoutReplay,
    )
    from lightning_data_modules.deepfake_detection_datamodule import (
        dataset_ids_to_indices,
    )
    import time

    strategy = MostRecentHarmonicReplayStrategy(
        replay_seed=42,
        buffer_size=16000,
        label_field="generator",
        use_remainder=False,
    )
    loader = AIGenBenchPipeline(
        dataset_loader=AIGenBenchDatasetLoader(
            Path(LOCAL_DATASET_PATH)
        ),
        train_batch_size=128,
        eval_batch_size=128,
        num_workers=1,
        sliding_windows_definition=SlidingWindowsDefinition(
            n_generators_per_window=4,
            benchmark_type="continual_learning",
            current_window=8,
        ),
        replay_strategy=WithoutReplay(),
    )

    sliding_windows = loader.make_sliding_windows()
    replay_timeline = []
    cl_timeline = []
    for window_idx in range(len(loader.windows_timeline)):
        if window_idx == 0:
            replay_timeline.append([])
        else:
            past_dataset = loader.dataset["train"].select(
                sum(sliding_windows[:window_idx], [])
            )
            current_dataset = loader.dataset["train"].select(
                sliding_windows[window_idx - 1]
            )
            current_buffer_ids = strategy.make_current_buffer(
                full_training_dataset=loader.dataset["train"],
                past_dataset=past_dataset,
                sliding_windows_timeline=loader.windows_timeline,
                n_th_window=window_idx,
                generator_name_to_id=loader.generator_name_to_id,
            )
            replay_timeline.append(current_buffer_ids)
            cl_timeline.append(past_dataset["ID"][:])
            assert set(replay_timeline[-1]).issubset(
                cl_timeline[-1]
            ), f"Replay buffer for window {window_idx} contains indices not present in past windows"
            assert (set(current_buffer_ids) - set(replay_timeline[-2])).issubset(
                current_dataset["ID"][:]
            )

    # Ensure consistency: only previously selected replay samples can be carried on to the next window
    for window_idx in range(1, len(replay_timeline)):
        previous_replay_ids = set(replay_timeline[window_idx - 1])
        current_replay_ids = set(replay_timeline[window_idx])

        # Asssert no duplicates
        assert len(previous_replay_ids) == len(
            replay_timeline[window_idx - 1]
        ), f"Duplicates found in replay buffer for window {window_idx - 1}"
        assert len(current_replay_ids) == len(
            replay_timeline[window_idx]
        ), f"Duplicates found in replay buffer for window {window_idx}"

        # Remove, from current indices, the ones that are also in the current window
        current_window_ids = set(
            loader.dataset["train"].select(sliding_windows[window_idx - 1])["ID"][:]
        )
        current_replay_ids.difference_update(current_window_ids)
        assert current_replay_ids.issubset(
            previous_replay_ids
        ), f"Replay buffer for window {window_idx} contains indices not present in previous buffer"

    # Timing
    replay_timeline = []
    start_time = time.time()
    for window_idx in range(len(sliding_windows)):
        if window_idx == 0:
            replay_timeline.append([])
        else:
            past_dataset = loader.dataset["train"].select(
                sum(sliding_windows[:window_idx], [])
            )
            current_buffer_ids = strategy.make_current_buffer(
                full_training_dataset=loader.dataset["train"],
                past_dataset=past_dataset,
                sliding_windows_timeline=loader.windows_timeline,
                n_th_window=window_idx,
                generator_name_to_id=loader.generator_name_to_id,
            )
            replay_timeline.append(current_buffer_ids)
    end_time = time.time()
    print(f"Done computing the replay timeline in {end_time - start_time:.2f} seconds")
    for window_idx, buffer in enumerate(replay_timeline):
        print(f"Window {window_idx}: {len(buffer)} samples in the replay buffer")

    # Print elements per class in the last buffer
    last_buffer_ids = replay_timeline[-1]
    last_buffer_indices = dataset_ids_to_indices(
        loader.dataset["train"], last_buffer_ids
    )
    last_buffer = loader.dataset["train"].select(last_buffer_indices)
    class_counts: Dict[str, int] = {}
    for gen in last_buffer["generator"]:
        if gen not in class_counts:
            class_counts[gen] = 0
        class_counts[gen] += 1
    print("Class distribution in the last buffer:")
    for gen, count in class_counts.items():
        print(f"  {gen}: {count}")


if __name__ == "__main__":
    _test_correctness()
