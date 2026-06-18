"""Custom exception classes for NBM verification system."""


class NBMVerificationError(Exception):
    """Base exception class for NBM verification system."""

    pass


class ConfigurationError(NBMVerificationError):
    """Exception raised for configuration errors."""

    pass


class DataAccessError(NBMVerificationError):
    """Exception raised for data access errors."""

    pass


class ValidationError(NBMVerificationError):
    """Exception raised for data validation errors."""

    pass


class ProcessingError(NBMVerificationError):
    """Exception raised for processing errors."""

    pass


class MetricCalculationError(NBMVerificationError):
    """Exception raised for metric calculation errors."""

    pass


class SpatialIndexError(NBMVerificationError):
    """Exception raised for spatial indexing errors."""

    pass


class TemporalAlignmentError(NBMVerificationError):
    """Exception raised for temporal alignment errors."""

    pass
