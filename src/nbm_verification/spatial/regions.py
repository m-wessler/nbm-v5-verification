"""Region management for CWA, RFC, and Public Forecast Zones."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from nbm_verification.utils.exceptions import SpatialIndexError

logger = logging.getLogger(__name__)


class RegionManager:
    """
    Manages regional shapefiles (CWA, RFC, Public Zones).

    Loads shapefiles and provides methods for querying which region(s)
    a gridpoint belongs to.
    """

    def __init__(self):
        """Initialize region manager."""
        self.regions: Dict[str, gpd.GeoDataFrame] = {}
        self.region_types: List[str] = []

    def load_regions(
        self, region_type: str, shapefile_path: Path, id_field: str = "ID"
    ) -> None:
        """
        Load regional shapefile.

        Parameters
        ----------
        region_type : str
            Type of region (e.g., "CWA", "RFC", "Zone")
        shapefile_path : Path
            Path to shapefile directory or .shp file
        id_field : str, optional
            Field name containing region IDs, by default "ID"

        Examples
        --------
        >>> rm = RegionManager()
        >>> rm.load_regions("CWA", Path("config/shapefiles/cwa"))
        """
        logger.info(f"Loading {region_type} regions from {shapefile_path}")

        # Find .shp file if directory provided
        if shapefile_path.is_dir():
            shp_files = list(shapefile_path.glob("*.shp"))
            if not shp_files:
                logger.warning(f"No shapefiles found in {shapefile_path}")
                return
            shapefile_path = shp_files[0]

        try:
            gdf = gpd.read_file(shapefile_path)
            logger.info(f"Loaded {len(gdf)} {region_type} regions")

            # Store the GeoDataFrame
            self.regions[region_type] = gdf
            if region_type not in self.region_types:
                self.region_types.append(region_type)

        except Exception as e:
            logger.error(f"Error loading {region_type} shapefile: {e}")
            raise SpatialIndexError(f"Failed to load {region_type} regions: {e}")

    def get_region_for_point(
        self, lat: float, lon: float, region_type: str
    ) -> Optional[Dict]:
        """
        Find which region contains a given point.

        Parameters
        ----------
        lat : float
            Latitude
        lon : float
            Longitude
        region_type : str
            Type of region to query

        Returns
        -------
        Optional[Dict]
            Region information if point is within a region, None otherwise

        Examples
        --------
        >>> rm = RegionManager()
        >>> rm.load_regions("CWA", Path("config/shapefiles/cwa"))
        >>> region = rm.get_region_for_point(41.5, -88.0, "CWA")
        """
        if region_type not in self.regions:
            logger.warning(f"Region type {region_type} not loaded")
            return None

        gdf = self.regions[region_type]
        point = Point(lon, lat)  # Shapely uses (x, y) = (lon, lat)

        # Find containing region(s)
        containing = gdf[gdf.contains(point)]

        if len(containing) == 0:
            return None

        # Return first match (could have overlaps)
        region = containing.iloc[0]

        return {
            "region_type": region_type,
            "region_id": region.get("ID", region.get("OBJECTID", "unknown")),
            "region_name": region.get("NAME", region.get("name", "unknown")),
        }

    def get_regions_for_points(
        self, lats: np.ndarray, lons: np.ndarray, region_type: str
    ) -> np.ndarray:
        """
        Find regions for multiple points.

        Parameters
        ----------
        lats : np.ndarray
            Array of latitudes
        lons : np.ndarray
            Array of longitudes
        region_type : str
            Type of region to query

        Returns
        -------
        np.ndarray
            Array of region IDs (same shape as lats/lons), None where no region

        Examples
        --------
        >>> rm = RegionManager()
        >>> lats = np.array([41.5, 42.0])
        >>> lons = np.array([-88.0, -87.5])
        >>> regions = rm.get_regions_for_points(lats, lons, "CWA")
        """
        if region_type not in self.regions:
            logger.warning(f"Region type {region_type} not loaded")
            return np.full(lats.shape, None)

        # Flatten for processing
        original_shape = lats.shape
        flat_lats = lats.flatten()
        flat_lons = lons.flatten()

        region_ids = []

        for lat, lon in zip(flat_lats, flat_lons):
            region_info = self.get_region_for_point(lat, lon, region_type)
            region_id = region_info["region_id"] if region_info else None
            region_ids.append(region_id)

        # Reshape to original
        return np.array(region_ids).reshape(original_shape)

    def list_regions(self, region_type: str) -> List[Dict]:
        """
        List all regions of a given type.

        Parameters
        ----------
        region_type : str
            Type of region

        Returns
        -------
        List[Dict]
            List of region information dictionaries
        """
        if region_type not in self.regions:
            logger.warning(f"Region type {region_type} not loaded")
            return []

        gdf = self.regions[region_type]
        regions = []

        for idx, row in gdf.iterrows():
            regions.append(
                {
                    "region_type": region_type,
                    "region_id": row.get("ID", row.get("OBJECTID", f"id_{idx}")),
                    "region_name": row.get("NAME", row.get("name", "unknown")),
                }
            )

        return regions

    def get_loaded_region_types(self) -> List[str]:
        """
        Get list of loaded region types.

        Returns
        -------
        List[str]
            List of region types
        """
        return self.region_types.copy()
