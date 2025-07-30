from typing import Dict, List, Optional, Tuple
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


def get_competition_for_block(
    competition_id: str,
    block: int,
    schedule: List[Tuple[int, List[Competition]]],
) -> Optional[Competition]:
    """
    Retrieves a single competition for a given block number and competition id.

    Args:
        competition_id (str): The ID of the competition to retrieve.
        block (int): The current block number.
        schedule (List[Tuple[int, List[Competition]]]): A list of tuples where the first element is the
            block number and the second is the list of competitions running at that block.

    Returns:
        Optional[Competition]: The competition for the given block and ID. Returns None if no
                               schedule is found or the competition is not found.
    """
    schedule_by_block = dict(schedule)
    competitions_for_block = get_competition_schedule_for_block(block, schedule_by_block)
    for comp in competitions_for_block:
        if comp.id == competition_id:
            return comp
    return None
