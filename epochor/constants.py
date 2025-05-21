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
    BASELINE = 0
    # Example of how to add more tracks:
    # CUSTOM_TRACK_1 = 1
    # CUSTOM_TRACK_2 = 2

# (Optional) Block-based competition schedule.
# Defines when different competitions (and their parameters, like epsilon for blending)
# become active.
#
# Example:
# COMPETITION_SCHEDULE = {
#    0: {CompetitionId.BASELINE: 1.0},  # Baseline competition active from block 0 with full weight
#    100000: {CompetitionId.BASELINE: 0.5, CompetitionId.CUSTOM_TRACK_1: 0.5}, # Blend at block 100k
#    200000: {CompetitionId.CUSTOM_TRACK_1: 1.0}, # Custom track 1 fully active from block 200k
# }
COMPETITION_SCHEDULE = {}
