"""MIT SSA forecasting harness. Four peer modules share explicit contracts."""

__version__ = "0.2.0"

from .contracts import ForecastRequest

__all__ = ["ForecastRequest", "__version__"]
