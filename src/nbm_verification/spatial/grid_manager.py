"""Grid management and spatial chunking."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from nbm_verification.utils.exceptions import SpatialIndexError

logger = logging.getLogger(__name__)


class GridManager:
    """
    Manages grid definition and spatial chunking strategy.

    Handles grid dimensions, coordinates, and divides the grid into
    manageable chunks for memory-efficient processing.
    """

    def __init__(
        self,
        ni: int,
        nj: int,
        lats: Optional[np.ndarray] = None,
        lons: Optional[np.ndarray] = None,
        chunk_size: Tuple[int, int] = (100, 100),
    ):
        """
        Initialize grid manager.

        Parameters
        ----------
        ni : int
            Number of gridpoints in i-direction (x)
        nj : int
            Number of gridpoints in j-direction (y)
        lats : np.ndarray, optional
            2D array of latitudes
        lons : np.ndarray, optional
            2D array of longitudes
        chunk_size : Tuple[int, int], optional
            Chunk size (i_size, j_size), by default (100, 100)
        """
        self.ni = ni
        self.nj = nj
        self.lats = lats
        self.lons = lons
        self.chunk_size = chunk_size

    def get_chunks(
        self, bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> List[Dict]:
        """
        Generate list of spatial chunks.

        Parameters
        ----------
        bbox : Tuple[float, float, float, float], optional
            Bounding box (min_lat, max_lat, min_lon, max_lon) to limit chunks

        Returns
        -------
        List[Dict]
            List of chunk dictionaries with i_start, i_end, j_start, j_end

        Examples
        --------
        >>> gm = GridManager(2345, 1597, chunk_size=(100, 100))
        >>> chunks = gm.get_chunks()
        >>> len(chunks)
        414
        """
        # Apply bounding box if specified
        if bbox is not None and self.lats is not None and self.lons is not None:
            i_range, j_range = self._get_bbox_indices(bbox)
        else:
            i_range = (0, self.ni)
            j_range = (0, self.nj)

        chunks = []
        chunk_i, chunk_j = self.chunk_size

        # Create chunks
        for j_start in range(j_range[0], j_range[1], chunk_j):
            j_end = min(j_start + chunk_j, j_range[1])

            for i_start in range(i_range[0], i_range[1], chunk_i):
                i_end = min(i_start + chunk_i, i_range[1])

                chunks.append(
                    {
                        "chunk_id": len(chunks),
                        "i_start": i_start,
                        "i_end": i_end,
                        "j_start": j_start,
                        "j_end": j_end,
                        "ni": i_end - i_start,
                        "nj": j_end - j_start,
                    }
                )

        logger.info(f"Created {len(chunks)} spatial chunks")
        return chunks

    def _get_bbox_indices(
        self, bbox: Tuple[float, float, float, float]
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Get grid indices for bounding box.

        Parameters
        ----------
        bbox : Tuple[float, float, float, float]
            Bounding box (min_lat, max_lat, min_lon, max_lon)

        Returns
        -------
        Tuple[Tuple[int, int], Tuple[int, int]]
            i_range and j_range tuples
        """
        min_lat, max_lat, min_lon, max_lon = bbox

        # Find indices where lat/lon are within bbox
        mask = (
            (self.lats >= min_lat)
            & (self.lats <= max_lat)
            & (self.lons >= min_lon)
            & (self.lons <= max_lon)
        )

        j_indices, i_indices = np.where(mask)

        if len(i_indices) == 0:
            raise SpatialIndexError("Bounding box does not intersect grid")

        i_range = (int(np.min(i_indices)), int(np.max(i_indices)) + 1)
        j_range = (int(np.min(j_indices)), int(np.max(j_indices)) + 1)

        return i_range, j_range

    def get_chunk_coords(self, chunk: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get lat/lon coordinates for a chunk.

        Parameters
        ----------
        chunk : Dict
            Chunk dictionary with i_start, i_end, j_start, j_end

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Chunk latitudes and longitudes

        Raises
        ------
        SpatialIndexError
            If coordinates not available
        """
        if self.lats is None or self.lons is None:
            raise SpatialIndexError("Grid coordinates not available")

        i_start = chunk["i_start"]
        i_end = chunk["i_end"]
        j_start = chunk["j_start"]
        j_end = chunk["j_end"]

        chunk_lats = self.lats[j_start:j_end, i_start:i_end]
        chunk_lons = self.lons[j_start:j_end, i_start:i_end]

        return chunk_lats, chunk_lons

    def set_coordinates(self, lats: np.ndarray, lons: np.ndarray) -> None:
        """
        Set grid coordinates.

        Parameters
        ----------
        lats : np.ndarray
            2D array of latitudes
        lons : np.ndarray
            2D array of longitudes
        """
        if lats.shape != (self.nj, self.ni):
            raise ValueError(f"Latitude array shape {lats.shape} does not match grid ({self.nj}, {self.ni})")

        if lons.shape != (self.nj, self.ni):
            raise ValueError(f"Longitude array shape {lons.shape} does not match grid ({self.nj}, {self.ni})")

        self.lats = lats
        self.lons = lons

    def get_total_gridpoints(self) -> int:
        """
        Get total number of gridpoints.

        Returns
        -------
        int
            Total gridpoints
        """
        return self.ni * self.nj

    def get_grid_info(self) -> Dict:
        """
        Get grid information summary.

        Returns
        -------
        Dict
            Dictionary with grid dimensions and chunk info
        """
        return {
            "ni": self.ni,
            "nj": self.nj,
            "total_gridpoints": self.get_total_gridpoints(),
            "chunk_size": self.chunk_size,
            "num_chunks": len(self.get_chunks()),
        }
