from enum import IntEnum

class CompetitionId(IntEnum):
    """
    Defines the different competition tracks available in the Epochor subnet.
    Validators and miners can use this to tailor their behavior based on the
    active competition.
    """
    UNIVARIATE = 0
    UNIVARIATE_COVARS = 1
    MULTIVARIATE =  2

    def __repr__(self) -> str:
        return f"{self.value}"
