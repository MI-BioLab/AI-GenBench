from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Union


@dataclass
class SlidingWindowsDefinition:
    n_generators_per_window: int
    current_window: Optional[int]
    benchmark_type: Literal["continual_learning", "cumulative", "none"]


@dataclass
class SlidingWindowsExperimentInfo:
    sliding_windows_definition: SlidingWindowsDefinition = field(
        default_factory=lambda: SlidingWindowsDefinition(
            n_generators_per_window=0,
            current_window=None,
            benchmark_type="none",
        )
    )
    load_weights_from_previous_window: bool = True
    experiment_id: Optional[Union[int, str]] = None
    experiment_name_prefix: str = "experiment_"
    window_name_prefix: str = "window_"
    logging_initial_step: Optional[int] = None
    experiments_data_folder: Path = field(
        default_factory=lambda: Path("experiments_data")
    )


__all__ = [
    "SlidingWindowsDefinition",
    "SlidingWindowsExperimentInfo",
]
