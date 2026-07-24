"""
Unit tests for Video Ingestion Service (metadata extraction and frame reader).
"""

import os
import tempfile
import cv2
import numpy as np
import pytest
from services.ingestion.metadata import extract_video_metadata
from services.ingestion.reader import VideoIngestionReader


@pytest.fixture
def temp_synthetic_video():
    """Creates a temporary synthetic 30 FPS, 60-frame (2 second) MP4 video file for testing."""
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "synthetic_match.mp4")

    width, height, fps = 640, 480, 30
    total_frames = 60
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for i in range(total_frames):
        # Create dummy frame with synthetic color variation
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (i * 4 % 255, 128, 64)
        writer.write(frame)

    writer.release()

    yield video_path

    # Cleanup
    if os.path.exists(video_path):
        os.remove(video_path)
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)


def test_extract_video_metadata(temp_synthetic_video):
    metadata = extract_video_metadata(temp_synthetic_video)
    assert metadata.width == 640
    assert metadata.height == 480
    assert abs(metadata.fps - 30.0) < 0.1
    assert metadata.total_frames == 60
    assert abs(metadata.duration_seconds - 2.0) < 0.1


def test_video_ingestion_reader_streaming(temp_synthetic_video):
    with VideoIngestionReader(temp_synthetic_video, target_fps=30.0) as reader:
        frames = list(reader.stream_frames())
        assert len(frames) == 60
        assert frames[0].frame_number == 1
        assert frames[0].image.shape == (480, 640, 3)
        assert frames[-1].frame_number == 60
        assert abs(frames[-1].timestamp_seconds - 2.0) < 0.1


def test_video_ingestion_reader_stride_downsampling(temp_synthetic_video):
    # Stream at target 15 FPS from 30 FPS source -> Should yield 30 frames
    with VideoIngestionReader(temp_synthetic_video, target_fps=15.0) as reader:
        frames = list(reader.stream_frames())
        assert len(frames) == 30
        assert reader.stride == 2
