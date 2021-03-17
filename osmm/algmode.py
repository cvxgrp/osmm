from enum import Enum


class AlgMode(Enum):
    NoHess = -1
    Exact = 0
    Diag = 1
    Trace = 2
    LowRankDiag = 3
    LowRankNewSamp = 4
    OuterProd = 5  # deprecated
    BFGS = 6
    Hutchinson = 7
    Bundle = 8
    LowRankDiagPlusBundle = 9
    LowRankNewSampPlusBundle = 10
    OuterProdPlusBundle = 11  # deprecated
    BFGSPlusBundle = 12
    LBFGS = 13
    LBFGSPlusBundle = 14
    LowRankQN = 15
    LowRankQNPlusBundle = 16
    LowRankQNPlusBundleNoLambda = 17

    def add_bundle(self):
        if self == AlgMode.NoHess or self == AlgMode.Exact or self == AlgMode.Diag or self == AlgMode.Trace \
                or self == AlgMode.LowRankQN or self == AlgMode.LBFGS or self == AlgMode.LowRankDiag \
                or self == AlgMode.LowRankNewSamp or self == AlgMode.OuterProd or self == AlgMode.BFGS \
                or self == AlgMode.Hutchinson:
            return False
        else:
            return True
