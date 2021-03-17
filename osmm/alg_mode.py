from enum import Enum


class AlgMode(Enum):
    Exact = 0  # full-rank
    Diag = 1
    Trace = 2
    Hutchinson = 3
    Bundle = 4
    LowRankDiagBundle = 5
    LowRankNewSampBundle = 6
    BFGSBundle = 7  # full-rank
    LowRankQNBundle = 8