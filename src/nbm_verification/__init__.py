"""
NBM Verification System

A comprehensive verification framework for evaluating weather forecast accuracy.
Compares NBM forecasts against URMA gridded analyses and Synoptic Data point observations.
"""

__version__ = "0.1.0"
__author__ = "Mike Wessler"

from nbm_verification.utils.logging import setup_logging
from nbm_verification.utils.config_parser import load_config
from nbm_verification.utils.exceptions import (
    NBMVerificationError,
    ConfigurationError,
    DataAccessError,
    ValidationError,
)

__all__ = [
    "setup_logging",
    "load_config",
    "NBMVerificationError",
    "ConfigurationError",
    "DataAccessError",
    "ValidationError",
]
