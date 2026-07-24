"""
Purpose: Player Tracking Engine managing multi-object tracking (MOT) and persistent track IDs.
Dependencies: numpy, services.vision.models, shared.domain.entities, shared.logging
Inputs: List of DetectedObject instances or PlayerDetectionResult per frame
Outputs: PlayerTrackingFrameResult containing active TrackedPlayer instances
"""

import time
from typing import Any, Optional, Union
import numpy as np
from services.vision.models import DetectedObject, PlayerDetectionResult, PlayerTrackingFrameResult, TrackedPlayer
from shared.domain.entities import BoundingBox, Point2D, TrackedObjectType
from shared.logging import setup_logger

logger = setup_logger("player_tracker", service_name="vision")


def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Computes IoU overlap ratio between two bounding boxes."""
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
    area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
    union = area1 + area2 - intersection
    return (intersection / union) if union > 0 else 0.0


class TrackState:
    """Internal state representation for a single tracked player."""

    def __init__(self, track_id: int, initial_bbox: BoundingBox, initial_ground_pos: Point2D):
        self.track_id = track_id
        self.bbox = initial_bbox
        self.ground_position = initial_ground_pos
        self.trajectory_history: list[Point2D] = [initial_ground_pos]
        self.disappeared_frames = 0
        self.total_tracked_frames = 1


class PlayerTracker:
    """Multi-Object Tracker (MOT) assigning persistent track IDs using IoU matching."""

    def __init__(self, max_disappeared: int = 30, iou_threshold: float = 0.3, max_history_len: int = 30):
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.max_history_len = max_history_len
        self.next_track_id = 1
        self.tracks: dict[int, TrackState] = {}

    def _compute_iou_matrix(
        self, existing_boxes: list[BoundingBox], new_boxes: list[BoundingBox]
    ) -> np.ndarray:
        """Computes IoU cost matrix between existing tracks and new detection bounding boxes."""
        matrix = np.zeros((len(existing_boxes), len(new_boxes)), dtype=np.float32)
        for i, b1 in enumerate(existing_boxes):
            for j, b2 in enumerate(new_boxes):
                matrix[i, j] = compute_iou(b1, b2)
        return matrix

    def update(
        self, detections: Union[list[DetectedObject], PlayerDetectionResult], frame: Optional[Any] = None
    ) -> PlayerTrackingFrameResult:
        """Updates internal tracks with new detections for the frame."""
        det_list = detections.detections if isinstance(detections, PlayerDetectionResult) else detections
        player_detections = [d for d in det_list if getattr(d, "object_type", None) == TrackedObjectType.PLAYER]

        if len(self.tracks) == 0:
            # First frame initialization
            for det in player_detections:
                ground_pos = Point2D(x=(det.bbox.x1 + det.bbox.x2) / 2.0, y=det.bbox.y2)
                track = TrackState(
                    track_id=self.next_track_id,
                    initial_bbox=det.bbox,
                    initial_ground_pos=ground_pos,
                )
                self.tracks[self.next_track_id] = track
                self.next_track_id += 1
        else:
            existing_track_ids = list(self.tracks.keys())
            existing_boxes = [self.tracks[tid].bbox for tid in existing_track_ids]
            new_boxes = [det.bbox for det in player_detections]

            if len(new_boxes) > 0 and len(existing_boxes) > 0:
                iou_matrix = self._compute_iou_matrix(existing_boxes, new_boxes)

                # Greedy bipartite matching
                matched_track_indices = set()
                matched_det_indices = set()

                # Sort by highest IoU matches
                flat_matches = []
                for i in range(len(existing_boxes)):
                    for j in range(len(new_boxes)):
                        flat_matches.append((iou_matrix[i, j], i, j))
                flat_matches.sort(key=lambda x: x[0], reverse=True)

                for iou_val, i, j in flat_matches:
                    if iou_val < self.iou_threshold:
                        break
                    if i in matched_track_indices or j in matched_det_indices:
                        continue

                    matched_track_indices.add(i)
                    matched_det_indices.add(j)

                    tid = existing_track_ids[i]
                    det = player_detections[j]
                    ground_pos = Point2D(x=(det.bbox.x1 + det.bbox.x2) / 2.0, y=det.bbox.y2)

                    self.tracks[tid].bbox = det.bbox
                    self.tracks[tid].ground_position = ground_pos
                    self.tracks[tid].trajectory_history.append(ground_pos)
                    if len(self.tracks[tid].trajectory_history) > self.max_history_len:
                        self.tracks[tid].trajectory_history.pop(0)

                    self.tracks[tid].disappeared_frames = 0
                    self.tracks[tid].total_tracked_frames += 1

                # Unmatched existing tracks -> increment disappeared_frames
                for i, tid in enumerate(existing_track_ids):
                    if i not in matched_track_indices:
                        self.tracks[tid].disappeared_frames += 1

                # Unmatched new detections -> assign new track IDs
                for j, det in enumerate(player_detections):
                    if j not in matched_det_indices:
                        ground_pos = Point2D(x=(det.bbox.x1 + det.bbox.x2) / 2.0, y=det.bbox.y2)
                        track = TrackState(
                            track_id=self.next_track_id,
                            initial_bbox=det.bbox,
                            initial_ground_pos=ground_pos,
                        )
                        self.tracks[self.next_track_id] = track
                        self.next_track_id += 1
            else:
                # No new detections -> increment disappeared_frames
                for tid in existing_track_ids:
                    self.tracks[tid].disappeared_frames += 1

        # Purge dead tracks exceeding max_disappeared
        dead_track_ids = [
            tid for tid, track in self.tracks.items() if track.disappeared_frames > self.max_disappeared
        ]
        for tid in dead_track_ids:
            del self.tracks[tid]

        # Convert internal tracks to TrackedPlayer DTOs
        active_tracked_players: list[TrackedPlayer] = []
        for tid, track in self.tracks.items():
            if track.disappeared_frames == 0:
                active_tracked_players.append(
                    TrackedPlayer(
                        track_id=tid,
                        bbox=track.bbox,
                        ground_position=track.ground_position,
                        trajectory_history=list(track.trajectory_history),
                    )
                )

        return PlayerTrackingFrameResult(
            frame_number=0,
            timestamp_seconds=0.0,
            tracked_players=active_tracked_players,
            processing_time_ms=0.0,
        )
