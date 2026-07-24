"""
Purpose: Video metadata extraction utilities.
Dependencies: cv2, pydantic
Inputs: File path or RTSP stream URL
Outputs: VideoMetadata DTO containing resolution, FPS, total frames, and duration
"""

import os
import cv2
from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    """Metadata describing a video source."""
    source_path: str
    width: int = Field(..., description="Frame width in pixels")
    height: int = Field(..., description="Frame height in pixels")
    fps: float = Field(..., description="Frames per second")
    total_frames: int = Field(..., description="Total frame count")
    duration_seconds: float = Field(..., description="Calculated duration in seconds")
    aspect_ratio: float = Field(..., description="Width to height ratio")


def extract_video_metadata(source: str) -> VideoMetadata:
    """Extracts metadata from a video file or stream source using OpenCV."""
    if not os.path.exists(source) and not source.startswith(("rtsp://", "http://", "https://")):
        raise FileNotFoundError(f"Video source not found: {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video source: {source}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Fallbacks for stream feeds or invalid metadata
        if fps <= 0:
            fps = 30.0
        if total_frames < 0:
            total_frames = 0

        duration_seconds = total_frames / fps if total_frames > 0 else 0.0
        aspect_ratio = width / height if height > 0 else 1.777

        return VideoMetadata(
            source_path=source,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=round(duration_seconds, 2),
            aspect_ratio=round(aspect_ratio, 3),
        )
    finally:
        cap.release()
