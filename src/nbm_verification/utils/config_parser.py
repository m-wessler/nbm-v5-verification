"""Configuration file parsing for NBM verification system."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from nbm_verification.utils.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and validate verification configuration from YAML file.

    Parameters
    ----------
    config_path : Union[str, Path]
        Path to YAML configuration file

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary

    Raises
    ------
    ConfigurationError
        If configuration file is invalid or missing required fields

    Examples
    --------
    >>> config = load_config("config/verification_config.yaml")
    >>> print(config["verification_run"]["name"])
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Error parsing YAML configuration: {e}")

    # Validate configuration structure
    _validate_config(config)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration structure and required fields.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary to validate

    Raises
    ------
    ConfigurationError
        If required fields are missing or invalid
    """
    # Check for top-level verification_run key
    if "verification_run" not in config:
        raise ConfigurationError("Configuration must contain 'verification_run' key")

    run_config = config["verification_run"]

    # Required fields
    required_fields = ["name", "temporal", "spatial", "variables", "output"]
    for field in required_fields:
        if field not in run_config:
            raise ConfigurationError(f"Missing required field: verification_run.{field}")

    # Validate temporal configuration
    _validate_temporal_config(run_config["temporal"])

    # Validate spatial configuration
    _validate_spatial_config(run_config["spatial"])

    # Validate variables
    _validate_variables_config(run_config["variables"])

    # Validate output configuration
    _validate_output_config(run_config["output"])


def _validate_temporal_config(temporal: Dict[str, Any]) -> None:
    """Validate temporal configuration section."""
    required = ["start_date", "end_date", "init_hours", "forecast_hours"]
    for field in required:
        if field not in temporal:
            raise ConfigurationError(f"Missing required temporal field: {field}")

    # Validate init_hours
    if not isinstance(temporal["init_hours"], list) or len(temporal["init_hours"]) == 0:
        raise ConfigurationError("init_hours must be a non-empty list")

    for group in temporal["init_hours"]:
        if "group_id" not in group or "hours" not in group:
            raise ConfigurationError("Each init_hours group must have 'group_id' and 'hours'")

    # Validate forecast_hours
    if not isinstance(temporal["forecast_hours"], list) or len(temporal["forecast_hours"]) == 0:
        raise ConfigurationError("forecast_hours must be a non-empty list")

    for group in temporal["forecast_hours"]:
        if "group_id" not in group or "hours" not in group:
            raise ConfigurationError(
                "Each forecast_hours group must have 'group_id' and 'hours'"
            )


def _validate_spatial_config(spatial: Dict[str, Any]) -> None:
    """Validate spatial configuration section."""
    # Domain is optional but if present must have type
    if "domain" in spatial and "type" not in spatial["domain"]:
        raise ConfigurationError("spatial.domain must have 'type' field")

    # gridpoint_verification is optional (default True)
    # regions and stations are optional sections


def _validate_variables_config(variables: List[Dict[str, Any]]) -> None:
    """Validate variables configuration."""
    if not isinstance(variables, list) or len(variables) == 0:
        raise ConfigurationError("variables must be a non-empty list")

    for var in variables:
        if "name" not in var:
            raise ConfigurationError("Each variable must have a 'name' field")
        if "metrics" not in var:
            raise ConfigurationError(f"Variable '{var['name']}' must have 'metrics' field")


def _validate_output_config(output: Dict[str, Any]) -> None:
    """Validate output configuration section."""
    if "base_path" not in output:
        raise ConfigurationError("output must have 'base_path' field")


def get_nested_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a value from nested dictionary using dot notation.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    key_path : str
        Dot-separated path to value (e.g., "verification_run.temporal.start_date")
    default : Any, optional
        Default value if key not found, by default None

    Returns
    -------
    Any
        Value at the specified path, or default if not found

    Examples
    --------
    >>> config = {"verification_run": {"name": "test_run"}}
    >>> get_nested_value(config, "verification_run.name")
    'test_run'
    """
    keys = key_path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value
