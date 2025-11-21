"""Main verification workflow orchestrator."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from nbm_verification.accumulation.spatial_accumulator import (
    GridpointAccumulator,
    RegionalAccumulator,
)
from nbm_verification.data_access.file_inventory import FileInventory
from nbm_verification.data_access.nbm_reader import NBMReader
from nbm_verification.data_access.synoptic_client import SynopticClient
from nbm_verification.data_access.urma_reader import URMAReader
from nbm_verification.metrics.calculator import MetricsCalculator
from nbm_verification.qc.warnings import WarningManager
from nbm_verification.spatial.grid_manager import GridManager
from nbm_verification.spatial.regions import RegionManager
from nbm_verification.spatial.spatial_index import SpatialIndex
from nbm_verification.spatial.stations import StationManager
from nbm_verification.storage.metadata import MetadataTracker
from nbm_verification.storage.output_writer import OutputWriter
from nbm_verification.temporal.time_manager import TimeManager
from nbm_verification.utils.config_parser import load_config
from nbm_verification.utils.logging import setup_logging
from nbm_verification.workflow.chunking_strategy import ChunkingStrategy
from nbm_verification.workflow.processor import ChunkProcessor

logger = logging.getLogger(__name__)


class VerificationOrchestrator:
    """
    Main orchestrator for the verification workflow.

    Coordinates all components to execute a complete verification run
    from configuration to final output.
    """

    def __init__(self, config_path: Path):
        """
        Initialize verification orchestrator.

        Parameters
        ----------
        config_path : Path
            Path to verification configuration file

        Examples
        --------
        >>> orchestrator = VerificationOrchestrator(Path("config/verification_config.yaml"))
        >>> orchestrator.run()
        """
        # Load configuration
        self.config = load_config(config_path)
        self.run_config = self.config["verification_run"]

        # Setup logging
        log_config = self.run_config.get("logging", {})
        log_file = log_config.get("log_file")
        if log_file:
            log_file = Path(log_file)
        setup_logging(
            log_level=log_config.get("level", "INFO"),
            log_file=log_file,
        )

        logger.info(f"Initialized verification run: {self.run_config['name']}")

        # Initialize components
        self.time_manager = TimeManager(self.run_config["temporal"])
        self.file_inventory = FileInventory()
        self.warning_manager = WarningManager()
        self.metadata_tracker = MetadataTracker(self.run_config["name"])

        # Will be initialized during preprocessing
        self.grid_manager: Optional[GridManager] = None
        self.region_manager: Optional[RegionManager] = None
        self.spatial_index: Optional[SpatialIndex] = None
        self.station_manager: Optional[StationManager] = None
        self.nbm_reader: Optional[NBMReader] = None
        self.urma_reader: Optional[URMAReader] = None
        self.synoptic_client: Optional[SynopticClient] = None

    def run(self) -> None:
        """
        Execute the complete verification workflow.

        This is the main entry point that orchestrates all phases of verification.
        """
        logger.info("=" * 80)
        logger.info(f"Starting verification run: {self.run_config['name']}")
        logger.info("=" * 80)

        try:
            # Phase 1: Preprocessing
            self._preprocessing_phase()

            # Phase 2: Main processing loop
            self._main_processing_loop()

            # Phase 3: Finalization
            self._finalization_phase()

            logger.info("Verification run completed successfully")

        except Exception as e:
            logger.error(f"Verification run failed: {e}", exc_info=True)
            raise

    def _preprocessing_phase(self) -> None:
        """Execute preprocessing phase."""
        logger.info("-" * 80)
        logger.info("Phase 1: Preprocessing")
        logger.info("-" * 80)

        # Initialize data readers
        self._initialize_data_readers()

        # Build file inventory
        self._build_file_inventory()

        # Initialize grid
        self._initialize_grid()

        # Load regions if enabled
        if self.run_config["spatial"].get("regions", {}).get("enabled", False):
            self._load_regions()

        # Load stations if enabled
        if self.run_config["spatial"].get("stations", {}).get("enabled", False):
            self._load_stations()

        logger.info("Preprocessing complete")

    def _initialize_data_readers(self) -> None:
        """Initialize NBM, URMA, and Synoptic data readers."""
        logger.info("Initializing data readers")

        # NBM reader
        nbm_config = self.run_config["data_sources"]["nbm"]
        self.nbm_reader = NBMReader(
            base_path=Path(nbm_config["base_path"]),
            file_pattern=nbm_config.get("file_pattern"),
        )

        # URMA reader
        urma_config = self.run_config["data_sources"]["urma"]
        self.urma_reader = URMAReader(
            source_type=urma_config["type"],
            bucket=urma_config.get("bucket"),
            prefix=urma_config.get("prefix"),
        )

        # Synoptic client (if stations enabled)
        if self.run_config["spatial"].get("stations", {}).get("enabled", False):
            self.synoptic_client = SynopticClient()

    def _build_file_inventory(self) -> None:
        """Build inventory of available data files."""
        logger.info("Building file inventory")

        # Get all init hours and forecast hours
        all_init_hours = []
        for group in self.run_config["temporal"]["init_hours"]:
            all_init_hours.extend(group["hours"])

        all_fhours = []
        for group in self.run_config["temporal"]["forecast_hours"]:
            all_fhours.extend(group["hours"])

        # Scan NBM files
        self.file_inventory.scan_nbm_directory(
            base_path=Path(self.run_config["data_sources"]["nbm"]["base_path"]),
            start_date=self.time_manager.start_date,
            end_date=self.time_manager.end_date,
            init_hours=list(set(all_init_hours)),
            forecast_hours=list(set(all_fhours)),
        )

        # Log file availability
        summary = self.file_inventory.get_missing_files_summary()
        logger.info(f"NBM files: {summary['total_nbm'] - summary['missing_nbm']}/{summary['total_nbm']} available")

        if summary["missing_nbm"] > 0:
            self.warning_manager.add_warning(
                "file_availability",
                f"{summary['missing_nbm']} NBM files missing",
            )

    def _initialize_grid(self) -> None:
        """Initialize grid manager."""
        logger.info("Initializing grid")

        # Get grid info from NBM reader
        grid_info = self.nbm_reader.get_grid_info()

        chunk_config = self.run_config["spatial"].get("chunk_size", {})
        chunk_size = (
            chunk_config.get("x", 100),
            chunk_config.get("y", 100),
        )

        self.grid_manager = GridManager(
            ni=grid_info["ni"],
            nj=grid_info["nj"],
            chunk_size=chunk_size,
        )

        logger.info(f"Grid: {grid_info['ni']} × {grid_info['nj']} gridpoints")

    def _load_regions(self) -> None:
        """Load regional shapefiles."""
        logger.info("Loading regional shapefiles")

        self.region_manager = RegionManager()
        regions_config = self.run_config["spatial"]["regions"]

        for region_type in regions_config.get("types", []):
            shapefile_path = regions_config["shapefile_paths"].get(region_type)
            if shapefile_path:
                try:
                    self.region_manager.load_regions(region_type, Path(shapefile_path))
                except Exception as e:
                    logger.warning(f"Failed to load {region_type} regions: {e}")

    def _load_stations(self) -> None:
        """Load station metadata."""
        logger.info("Loading station metadata")

        stations_config = self.run_config["spatial"]["stations"]
        filters = stations_config.get("filters", {})

        # Get stations from Synoptic
        stations_df = self.synoptic_client.get_stations(
            networks=filters.get("networks"),
            states=filters.get("states"),
        )

        self.station_manager = StationManager()
        self.station_manager.load_stations_from_dataframe(stations_df)

        logger.info(f"Loaded {len(stations_df)} stations")

    def _main_processing_loop(self) -> None:
        """Execute main processing loop."""
        logger.info("-" * 80)
        logger.info("Phase 2: Main Processing")
        logger.info("-" * 80)

        # For each variable
        for variable_config in self.run_config["variables"]:
            self._process_variable(variable_config)

    def _process_variable(self, variable_config: Dict) -> None:
        """
        Process one variable.

        Parameters
        ----------
        variable_config : Dict
            Variable configuration
        """
        variable_name = variable_config["name"]
        logger.info(f"Processing variable: {variable_name}")

        # This would contain the full processing logic
        # For now, just a placeholder
        logger.info(f"Variable {variable_name} processing complete")

    def _finalization_phase(self) -> None:
        """Execute finalization phase."""
        logger.info("-" * 80)
        logger.info("Phase 3: Finalization")
        logger.info("-" * 80)

        # Write metadata
        self.metadata_tracker.save(
            Path(self.run_config["output"]["base_path"]) / self.run_config["name"]
        )

        # Write warnings
        self.warning_manager.save(
            Path(self.run_config["output"]["base_path"])
            / self.run_config["name"]
            / "warnings"
        )

        logger.info("Finalization complete")
