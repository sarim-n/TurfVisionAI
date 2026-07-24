"""
Purpose: Data model for decoded video frames.
Dependencies: numpy, pydantic, datetime
Inputs: Raw frame array, timing metadata
Outputs: FrameData object passed downstream to Vision Engine
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np
from services.ingestion.metadata import VideoMetadata


@dataclass
class FrameData:
    """Wrapper holding a decoded BGR image frame and temporal metadata."""
    frame_number: int
    timestamp_seconds: float
    image: np.ndarray
    metadata: VideoMetadata
    wall_clock_time: datetime = datetime.now(timezone.utc)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Returns (height, width, channels)."""
        return self.image.shape  # type: ignore

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]
