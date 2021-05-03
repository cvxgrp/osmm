from enum import Enum


class AlgMode(Enum):
    Bundle = 0
    LowRankQNBundle = 1
    ExactHessian = 2
    LowRankDiagHessian = 3