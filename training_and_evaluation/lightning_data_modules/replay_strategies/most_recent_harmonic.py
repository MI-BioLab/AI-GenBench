from lightning_data_modules.replay_strategies.most_recent import (
    MostRecentReplayStrategy,
)
from typing import Dict, List


class MostRecentHarmonicReplayStrategy(MostRecentReplayStrategy):
    """
    Similar to MostRecentUnboundedReplayStrategy, but the size of the buffer for each generator is computed
    using <buffer_size>*(1/n), where n is the number of sliding windows since the generator was introduced.

    This means that, unlike MostRecentReplayStrategy, the actual buffer size will grow unbounded (harmonic series),
    and the buffer_size constructor parameter will only be used to compute the initial number of samples per generator.
    """

    def __init__(
        self,
        replay_seed: int,
        buffer_size: int,
        label_field: str = "generator",
        use_remainder: bool = True,
    ):
        super().__init__(replay_seed, buffer_size, label_field)
        self.use_remainder = use_remainder

    def _compute_n_elements_per_generator(
        self, sliding_windows_timeline: List[List[str]], sliding_window_idx: int
    ) -> Dict[str, int]:
        n_elements_per_generator = dict()
        for window_idx in range(sliding_window_idx):
            current_window_generators = sliding_windows_timeline[window_idx]
            n_current_generators = len(current_window_generators)

            sliding_windows_so_far = sliding_window_idx - window_idx
            n_elements_to_consider = int(
                (self.buffer_size / sliding_windows_so_far) / n_current_generators
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
        if not self.use_remainder:
            return 0

        n_elements_unrounded = 0.0
        for window_idx in range(sliding_window_idx):
            current_window_generators = sliding_windows_timeline[window_idx]
            n_current_generators = len(current_window_generators)

            sliding_windows_so_far = sliding_window_idx - window_idx
            n_elements_unrounded += self.buffer_size / sliding_windows_so_far

        n_elements_rounded = sum(n_elements_per_generator.values())
        remainder = int(n_elements_unrounded - n_elements_rounded)
        # print(remainder)
        return remainder


__all__ = ["MostRecentHarmonicReplayStrategy"]
