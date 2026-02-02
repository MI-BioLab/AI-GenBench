from typing import List

from lightning_data_modules.replay_strategies.replay_strategy import ReplayStrategy


class WithoutReplay(ReplayStrategy):
    def __init__(self):
        super().__init__()

    def make_current_buffer(self, *args, **kwargs) -> List[int]:
        return []


__all__ = ["WithoutReplay"]
