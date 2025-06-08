# The MIT License (MIT)
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

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
