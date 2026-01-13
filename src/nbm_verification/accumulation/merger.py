"""Functions for merging accumulators."""

from typing import List

from nbm_verification.accumulation.accumulator_base import StatisticsAccumulator


def merge_accumulators(accumulators: List[StatisticsAccumulator]) -> StatisticsAccumulator:
    """
    Merge multiple accumulators into a single accumulator.

    Parameters
    ----------
    accumulators : List[StatisticsAccumulator]
        List of accumulators to merge

    Returns
    -------
    StatisticsAccumulator
        Merged accumulator containing combined statistics

    Raises
    ------
    ValueError
        If accumulators list is empty

    Examples
    --------
    >>> acc1 = GridpointAccumulator(0, 0, 40.0, -100.0, [273.15])
    >>> acc2 = GridpointAccumulator(0, 0, 40.0, -100.0, [273.15])
    >>> merged = merge_accumulators([acc1, acc2])
    """
    if not accumulators:
        raise ValueError("Cannot merge empty list of accumulators")

    # Start with first accumulator
    merged = accumulators[0]

    # Merge the rest
    for acc in accumulators[1:]:
        merged.merge(acc)

    return merged
