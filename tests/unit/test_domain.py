"""
Unit tests for domain entities and event schemas.
"""

from shared.domain.entities import BoundingBox, Point2D
from shared.schemas.events import EventType, MatchEvent


def test_bounding_box_center():
    bbox = BoundingBox(x1=100.0, y1=200.0, x2=200.0, y2=400.0, confidence=0.95)
    center = bbox.center
    assert center.x == 150.0
    assert center.y == 300.0

    bottom_center = bbox.bottom_center
    assert bottom_center.x == 150.0
    assert bottom_center.y == 400.0


def test_match_event_schema():
    event = MatchEvent(
        event_id="evt_123",
        match_id="match_456",
        event_type=EventType.GOAL,
        video_timestamp_seconds=120.5,
        frame_number=3615,
        location=Point2D(x=52.0, y=34.0),
        details={"scorer_id": 10},
    )
    assert event.event_type == EventType.GOAL
    assert event.details["scorer_id"] == 10
