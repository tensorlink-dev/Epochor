from typing import Dict, List
from epochor.model.model_constraints import Competition


def get_competition_schedule_for_block(
    block: int,
    schedule_by_block: Dict[int, List[Competition]],
) -> List[Competition]:
    """
    Retrieves the competition schedule for a given block number.

    Args:
        block (int): The current block number.
        schedule_by_block (Dict[int, List[Competition]]): A dictionary where keys are block numbers
            and values are the list of competitions running at that block.

    Returns:
        List[Competition]: The list of competitions for the given block. Returns an empty list if no
                           schedule is found.
    """
    # Find the most recent block number in the schedule that is less than or equal to the current block.
    active_block = max(
        (b for b in schedule_by_block.keys() if b <= block),
        default=None,
    )

    if active_block is None:
        return []

    return schedule_by_block.get(active_block, [])
