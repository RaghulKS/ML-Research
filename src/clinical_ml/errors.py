"""Research-pipeline errors."""


class ClinicalMLError(ValueError):
    """Base error for expected research-pipeline failures."""


class LeakageError(ClinicalMLError):
    """Raised when future information is available to a past prediction."""


class IncompatibleTaskError(ClinicalMLError):
    """Raised when two runs do not share an outcome definition."""


class DatasetLabelError(ClinicalMLError):
    """Raised when synthetic data would be reported as a clinical result."""
