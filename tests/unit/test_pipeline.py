"""
Unit tests for TurfVision AI Master Pipeline Integrator.
"""

import numpy as np
import pytest
from services.ingestion.frame import FrameData
from services.ingestion.metadata import VideoMetadata
from services.pipeline.models import PipelineConfig
from services.pipeline.pipeline_runner import TurfVisionPipeline


@pytest.fixture
def sample_video_frame():
    metadata = VideoMetadata(
        source_path="synthetic.mp4",
        width=640,
        height=480,
        fps=30.0,
        total_frames=100,
        duration_seconds=3.33,
        aspect_ratio=1.333,
    )
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    return FrameData(frame_number=1, timestamp_seconds=0.033, image=img, metadata=metadata)


def test_pipeline_runner_single_frame(sample_video_frame):
    config = PipelineConfig(match_id="test_m_pipeline", enable_visualization=True)
    pipeline = TurfVisionPipeline(config)

    res = pipeline.run_frame(sample_video_frame)

    assert res.frame_number == 1
    assert res.match_state.match_id == "test_m_pipeline"
    assert res.annotated_image is not None
    assert res.annotated_image.shape == (480, 640, 3)

    report = pipeline.get_analytics_report()
    assert report.match_id == "test_m_pipeline"
