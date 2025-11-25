"""Spatial index for efficient gridpoint-to-region lookups."""

import logging
from typing import Dict, List, Optional, Set

import numpy as np

from nbm_verification.spatial.regions import RegionManager

logger = logging.getLogger(__name__)


class SpatialIndex:
    """
    Spatial index mapping gridpoints to regions.

    Pre-computes which gridpoints belong to which regions to avoid
    repeated spatial queries during processing.
    """

    def __init__(self, region_manager: RegionManager):
        """
        Initialize spatial index.

        Parameters
        ----------
        region_manager : RegionManager
            Region manager with loaded shapefiles
        """
        self.region_manager = region_manager
        self.gridpoint_to_regions: Dict[tuple, Dict[str, str]] = {}
        self.region_to_gridpoints: Dict[str, Dict[str, Set[tuple]]] = {}

    def build_index(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        region_types: Optional[List[str]] = None,
    ) -> None:
        """
        Build spatial index for all gridpoints.

        Parameters
        ----------
        lats : np.ndarray
            2D array of gridpoint latitudes
        lons : np.ndarray
            2D array of gridpoint longitudes
        region_types : List[str], optional
            List of region types to index. If None, uses all loaded types.

        Examples
        --------
        >>> rm = RegionManager()
        >>> rm.load_regions("CWA", Path("config/shapefiles/cwa"))
        >>> si = SpatialIndex(rm)
        >>> si.build_index(lats, lons, ["CWA"])
        """
        if region_types is None:
            region_types = self.region_manager.get_loaded_region_types()

        logger.info(f"Building spatial index for {region_types}")

        nj, ni = lats.shape

        # Initialize region_to_gridpoints structure
        for region_type in region_types:
            self.region_to_gridpoints[region_type] = {}
            regions = self.region_manager.list_regions(region_type)
            for region in regions:
                self.region_to_gridpoints[region_type][region["region_id"]] = set()

        # Iterate over all gridpoints
        for j in range(nj):
            for i in range(ni):
                lat = lats[j, i]
                lon = lons[j, i]
                gridpoint = (i, j)

                # Find regions for this gridpoint
                self.gridpoint_to_regions[gridpoint] = {}

                for region_type in region_types:
                    region_info = self.region_manager.get_region_for_point(
                        lat, lon, region_type
                    )

                    if region_info:
                        region_id = region_info["region_id"]
                        self.gridpoint_to_regions[gridpoint][region_type] = region_id

                        # Add to reverse index
                        self.region_to_gridpoints[region_type][region_id].add(gridpoint)

            # Log progress
            if (j + 1) % 100 == 0:
                logger.info(f"Indexed {j + 1}/{nj} rows")

        logger.info("Spatial index built successfully")

    def get_region_for_gridpoint(
        self, i: int, j: int, region_type: str
    ) -> Optional[str]:
        """
        Get region ID for a gridpoint.

        Parameters
        ----------
        i : int
            Gridpoint i-index
        j : int
            Gridpoint j-index
        region_type : str
            Region type

        Returns
        -------
        Optional[str]
            Region ID, or None if gridpoint not in any region
        """
        gridpoint = (i, j)
        if gridpoint not in self.gridpoint_to_regions:
            return None

        return self.gridpoint_to_regions[gridpoint].get(region_type)

    def get_gridpoints_for_region(
        self, region_id: str, region_type: str
    ) -> Set[tuple]:
        """
        Get all gridpoints within a region.

        Parameters
        ----------
        region_id : str
            Region ID
        region_type : str
            Region type

        Returns
        -------
        Set[tuple]
            Set of (i, j) gridpoint tuples
        """
        if region_type not in self.region_to_gridpoints:
            return set()

        return self.region_to_gridpoints[region_type].get(region_id, set())

    def get_all_regions(self, region_type: str) -> List[str]:
        """
        Get list of all region IDs for a region type.

        Parameters
        ----------
        region_type : str
            Region type

        Returns
        -------
        List[str]
            List of region IDs
        """
        if region_type not in self.region_to_gridpoints:
            return []

        return list(self.region_to_gridpoints[region_type].keys())

    def save_index(self, filepath: str) -> None:
        """
        Save spatial index to file for reuse.

        Parameters
        ----------
        filepath : str
            Path to save index
        """
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "gridpoint_to_regions": self.gridpoint_to_regions,
                    "region_to_gridpoints": self.region_to_gridpoints,
                },
                f,
            )

        logger.info(f"Spatial index saved to {filepath}")

    def load_index(self, filepath: str) -> None:
        """
        Load spatial index from file.

        Parameters
        ----------
        filepath : str
            Path to load index from
        """
        import pickle

        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.gridpoint_to_regions = data["gridpoint_to_regions"]
            self.region_to_gridpoints = data["region_to_gridpoints"]

        logger.info(f"Spatial index loaded from {filepath}")
