"""Chunking strategy for spatial and temporal processing."""

import logging
from typing import Dict, List, Optional, Tuple

from nbm_verification.spatial.grid_manager import GridManager
from nbm_verification.temporal.time_manager import TimeManager

logger = logging.getLogger(__name__)


class ChunkingStrategy:
    """
    Determines how to chunk data for memory-efficient processing.

    Combines spatial and temporal chunking to create manageable
    work units that fit in memory.
    """

    def __init__(
        self,
        grid_manager: GridManager,
        time_manager: TimeManager,
        max_chunk_memory_mb: int = 1000,
    ):
        """
        Initialize chunking strategy.

        Parameters
        ----------
        grid_manager : GridManager
            Grid manager with spatial chunking
        time_manager : TimeManager
            Time manager with temporal configuration
        max_chunk_memory_mb : int, optional
            Maximum memory per chunk in MB, by default 1000
        """
        self.grid_manager = grid_manager
        self.time_manager = time_manager
        self.max_chunk_memory_mb = max_chunk_memory_mb

    def get_processing_plan(
        self, spatial_bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> Dict:
        """
        Create complete processing plan.

        Parameters
        ----------
        spatial_bbox : Tuple[float, float, float, float], optional
            Bounding box to limit spatial extent

        Returns
        -------
        Dict
            Processing plan with spatial chunks and temporal groups

        Examples
        --------
        >>> cs = ChunkingStrategy(grid_manager, time_manager)
        >>> plan = cs.get_processing_plan()
        >>> print(plan['num_spatial_chunks'])
        """
        # Get spatial chunks
        spatial_chunks = self.grid_manager.get_chunks(bbox=spatial_bbox)

        # Get temporal groups (init_group, fhour_group combinations)
        temporal_groups = self.time_manager.group_times_for_processing()

        plan = {
            "num_spatial_chunks": len(spatial_chunks),
            "spatial_chunks": spatial_chunks,
            "num_temporal_groups": len(temporal_groups),
            "temporal_groups": temporal_groups,
            "total_tasks": len(spatial_chunks) * len(temporal_groups),
        }

        logger.info(
            f"Processing plan: {plan['num_spatial_chunks']} spatial chunks × "
            f"{plan['num_temporal_groups']} temporal groups = "
            f"{plan['total_tasks']} total tasks"
        )

        return plan

    def estimate_chunk_memory(self, chunk: Dict, num_variables: int = 1) -> float:
        """
        Estimate memory required for a spatial chunk.

        Parameters
        ----------
        chunk : Dict
            Spatial chunk definition
        num_variables : int, optional
            Number of variables to load simultaneously

        Returns
        -------
        float
            Estimated memory in MB
        """
        # Estimate: each gridpoint × num_variables × 2 (forecast + obs) × 8 bytes (float64)
        ni = chunk["ni"]
        nj = chunk["nj"]
        n_gridpoints = ni * nj

        # Bytes per gridpoint for forecast and observation
        bytes_per_gridpoint = num_variables * 2 * 8

        total_bytes = n_gridpoints * bytes_per_gridpoint

        # Convert to MB
        memory_mb = total_bytes / (1024 * 1024)

        return memory_mb

    def adjust_chunk_size(self, current_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Adjust chunk size based on memory constraints.

        Parameters
        ----------
        current_size : Tuple[int, int]
            Current chunk size (ni, nj)

        Returns
        -------
        Tuple[int, int]
            Adjusted chunk size
        """
        # Create a test chunk
        test_chunk = {"ni": current_size[0], "nj": current_size[1]}

        estimated_memory = self.estimate_chunk_memory(test_chunk)

        if estimated_memory <= self.max_chunk_memory_mb:
            return current_size

        # Reduce chunk size proportionally
        scale_factor = (self.max_chunk_memory_mb / estimated_memory) ** 0.5

        new_ni = int(current_size[0] * scale_factor)
        new_nj = int(current_size[1] * scale_factor)

        logger.info(
            f"Adjusted chunk size from {current_size} to ({new_ni}, {new_nj}) "
            f"to fit memory constraint"
        )

        return (new_ni, new_nj)
