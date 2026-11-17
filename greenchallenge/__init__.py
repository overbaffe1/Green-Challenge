"""Green Challenge 2026 — model serving and execution layer.

The package is intentionally small and dependency-light: NumPy, msgpack and
websockets are already in the contest's baseline image.  Heavy model code
(GreenVLA v1.1, Qwen3.5) stays in the upstream process; this package is the
part a team can change without rebuilding a 20 GB image.
"""

from .boundary import Boundary, BoundaryConfig
from .contract import Calibration
from .feed import FeedConfig, NumericActionFeed
from .layout import MODEL_DIM, RAW_STATE_DIM, WIRE_DIM

__version__ = "0.1.0"

__all__ = [
    "Boundary",
    "BoundaryConfig",
    "Calibration",
    "FeedConfig",
    "NumericActionFeed",
    "MODEL_DIM",
    "RAW_STATE_DIM",
    "WIRE_DIM",
    "__version__",
]
