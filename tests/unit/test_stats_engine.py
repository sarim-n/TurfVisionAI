"""
Unit tests for Match Statistics Engine (possession split, player distance run, and 2D heatmaps).
"""

import numpy as np
import pytest
from services.ingestion.frame import FrameData
from services.ingestion.metadata import VideoMetadata
from services.stats_engine.stats_calculator import MatchStatsEngine
from services.vision.homography import PitchHomography
from services.vision.models import TrackedBall, TrackedPlayer
from shared.domain.entities import BoundingBox, Point2D


@pytest.fixture
def sample_frame_metadata():
    return VideoMetadata(
        source_path="test.mp4",
        width=640,
        height=480,
        fps=30.0,
        total_frames=100,
        duration_seconds=3.33,
        aspect_ratio=1.333,
    )


def test_match_stats_engine_distance_and_possession(sample_frame_metadata):
    engine = MatchStatsEngine(match_id="match_test_stats")
    homography = PitchHomography.create_default_calibration(640, 480)
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 1: Player #1 at (100, 100) on left side of pitch (Home side), Ball at (105, 105)
    f1 = FrameData(frame_number=1, timestamp_seconds=0.033, image=img, metadata=sample_frame_metadata)
    p1 = TrackedPlayer(
        track_id=1,
        bbox=BoundingBox(x1=90.0, y1=80.0, x2=110.0, y2=100.0, confidence=0.9),
        ground_position=Point2D(x=100.0, y=100.0),
    )
    ball = TrackedBall(center=Point2D(x=105.0, y=105.0))

    engine.process_frame([p1], ball, homography, f1)

    # Frame 2: Player #1 moves to (130, 100) -> Distance step
    f2 = FrameData(frame_number=2, timestamp_seconds=0.066, image=img, metadata=sample_frame_metadata)
    p1_moved = TrackedPlayer(
        track_id=1,
        bbox=BoundingBox(x1=120.0, y1=80.0, x2=140.0, y2=100.0, confidence=0.9),
        ground_position=Point2D(x=130.0, y=100.0),
    )
    engine.process_frame([p1_moved], ball, homography, f2)

    # Analytics Report Check
    report = engine.get_analytics_report()
    assert report.match_id == "match_test_stats"
    assert report.home_stats.possession_percentage == 100.0
    assert len(report.player_stats) == 1
    assert report.player_stats[0].distance_run_meters > 0.0


def test_generate_heatmap(sample_frame_metadata):
    engine = MatchStatsEngine()
    homography = PitchHomography.create_default_calibration(640, 480)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    f1 = FrameData(frame_number=1, timestamp_seconds=0.033, image=img, metadata=sample_frame_metadata)

    p1 = TrackedPlayer(
        track_id=1,
        bbox=BoundingBox(x1=90.0, y1=80.0, x2=110.0, y2=100.0, confidence=0.9),
        ground_position=Point2D(x=100.0, y=100.0),
    )
    engine.process_frame([p1], None, homography, f1)

    heatmap = engine.generate_heatmap(track_id=1, pitch_grid_size=(105, 68))
    assert heatmap.shape == (68, 105)
    assert np.max(heatmap) == 1.0
