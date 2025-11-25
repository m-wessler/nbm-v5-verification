"""Sample count completeness checking."""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def check_sample_completeness(
    n_samples: int,
    min_samples: int,
    entity_type: str,
    entity_id: str,
) -> Dict[str, any]:
    """
    Check if sample count meets minimum threshold.

    Parameters
    ----------
    n_samples : int
        Actual number of samples
    min_samples : int
        Minimum required samples
    entity_type : str
        Type of entity (gridpoint, region, station)
    entity_id : str
        Entity identifier

    Returns
    -------
    Dict[str, any]
        Dictionary with completeness check results

    Examples
    --------
    >>> result = check_sample_completeness(100, 50, "gridpoint", "(10, 20)")
    >>> result['sufficient']
    True
    """
    sufficient = n_samples >= min_samples

    result = {
        "sufficient": sufficient,
        "n_samples": n_samples,
        "min_samples": min_samples,
        "entity_type": entity_type,
        "entity_id": entity_id,
    }

    if not sufficient:
        logger.warning(
            f"{entity_type} {entity_id}: Insufficient samples "
            f"({n_samples} < {min_samples} minimum)"
        )

    return result


def check_gridpoint_completeness(
    gridpoint_metrics: Dict,
    min_samples: int = 5,
) -> List[Dict]:
    """
    Check sample completeness for all gridpoints.

    Parameters
    ----------
    gridpoint_metrics : Dict
        Dictionary of gridpoint metrics keyed by (i, j)
    min_samples : int, optional
        Minimum required samples, by default 5

    Returns
    -------
    List[Dict]
        List of completeness check results for gridpoints with insufficient samples
    """
    insufficient = []

    for (i, j), metrics in gridpoint_metrics.items():
        n_samples = metrics.get("n_samples", 0)

        result = check_sample_completeness(
            n_samples, min_samples, "gridpoint", f"({i}, {j})"
        )

        if not result["sufficient"]:
            insufficient.append(result)

    if len(insufficient) > 0:
        logger.warning(
            f"{len(insufficient)} gridpoints have insufficient samples "
            f"(< {min_samples})"
        )

    return insufficient


def check_regional_completeness(
    regional_metrics: List[Dict],
    min_samples: int = 50,
) -> List[Dict]:
    """
    Check sample completeness for all regions.

    Parameters
    ----------
    regional_metrics : List[Dict]
        List of regional metric dictionaries
    min_samples : int, optional
        Minimum required samples, by default 50

    Returns
    -------
    List[Dict]
        List of completeness check results for regions with insufficient samples
    """
    insufficient = []

    for region in regional_metrics:
        n_samples = region.get("n_samples", 0)
        region_id = region.get("region_id", "unknown")
        region_type = region.get("region_type", "unknown")

        result = check_sample_completeness(
            n_samples, min_samples, f"{region_type}_region", region_id
        )

        if not result["sufficient"]:
            insufficient.append(result)

    if len(insufficient) > 0:
        logger.warning(
            f"{len(insufficient)} regions have insufficient samples (< {min_samples})"
        )

    return insufficient


def check_station_completeness(
    station_metrics: List[Dict],
    min_samples: int = 10,
) -> List[Dict]:
    """
    Check sample completeness for all stations.

    Parameters
    ----------
    station_metrics : List[Dict]
        List of station metric dictionaries
    min_samples : int, optional
        Minimum required samples, by default 10

    Returns
    -------
    List[Dict]
        List of completeness check results for stations with insufficient samples
    """
    insufficient = []

    for station in station_metrics:
        n_samples = station.get("n_samples", 0)
        station_id = station.get("station_id", "unknown")

        result = check_sample_completeness(
            n_samples, min_samples, "station", station_id
        )

        if not result["sufficient"]:
            insufficient.append(result)

    if len(insufficient) > 0:
        logger.warning(
            f"{len(insufficient)} stations have insufficient samples (< {min_samples})"
        )

    return insufficient


def get_completeness_summary(completeness_results: List[Dict]) -> Dict:
    """
    Get summary of completeness check results.

    Parameters
    ----------
    completeness_results : List[Dict]
        List of completeness check results

    Returns
    -------
    Dict
        Summary statistics
    """
    if not completeness_results:
        return {
            "total_entities": 0,
            "insufficient_count": 0,
            "min_samples_found": 0,
            "max_samples_found": 0,
        }

    n_samples_list = [r["n_samples"] for r in completeness_results]

    return {
        "total_entities": len(completeness_results),
        "insufficient_count": len(completeness_results),
        "min_samples_found": min(n_samples_list) if n_samples_list else 0,
        "max_samples_found": max(n_samples_list) if n_samples_list else 0,
        "mean_samples": sum(n_samples_list) / len(n_samples_list) if n_samples_list else 0,
    }
