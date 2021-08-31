from enum import Enum


class HessianMode(Enum):
    Zero = 0
    LowRankQN = 1
    LowRankDiagEVD = 2