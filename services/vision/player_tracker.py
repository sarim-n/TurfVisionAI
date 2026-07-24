"""
Purpose: Player Tracking Engine for Multi-Object Tracking (MOT) across video frames.
Dependencies: numpy, services.vision.models, shared.domain.entities, shared.logging
Inputs: PlayerDetectionResult and FrameData
Outputs: PlayerTrackingFrameResult containing persistent TrackedPlayer entities
"""

import numpy as np
from services.ingestion.frame import FrameData
from services.vision.models import PlayerDetectionResult, PlayerTrackingFrameResult, TrackedPlayer
from shared.domain.entities import BoundingBox, Point2D
from shared.logging import setup_logger

logger = setup_logger("player_tracker", service_name="vision")


def compute_iou(boxA: BoundingBox, boxB: BoundingBox) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA.x1, boxB.x1)
    yA = max(boxA.y1, boxB.y1)
    xB = min(boxA.x2, boxB.x2)
    yB = min(boxA.y2, boxB.y2)

    inter_width = max(0.0, xB - xA)
    inter_height = max(0.0, yB - yA)
    inter_area = inter_width * inter_height

    boxA_area = (boxA.x2 - boxA.x1) * (boxA.y2 - boxA.y1)
    boxB_area = (boxB.x2 - boxB.x1) * (boxB.y2 - boxB.y1)

    union_area = boxA_area + boxB_area - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


class PlayerTracker:
    """Multi-Object Player Tracker maintaining persistent track IDs and motion trajectory tails."""

    def __init__(self, iou_threshold: float = 0.3, max_history_len: int = 30, max_disappeared: int = 15):
        self.iou_threshold = iou_threshold
        self.max_history_len = max_history_len
        self.max_disappeared = max_disappeared

        self._next_track_id = 1
        self._active_tracks: dict[int, dict] = {}
        # Format: { track_id: {"bbox": BoundingBox, "history": list[Point2D], "disappeared": int, "active_count": int} }

    def reset(self) -> None:
        """Resets tracking state and ID counters."""
        self._next_track_id = 1
        self._active_tracks.clear()
        logger.info("PlayerTracker state reset.")

    def update(
        self, detection_result: PlayerDetectionResult, frame: FrameData
    ) -> PlayerTrackingFrameResult:
        """Updates tracker state with new frame detections and returns persistent TrackedPlayer instances."""
        new_detections = detection_result.detections

        # If no active tracks exist, initialize new tracks for all detections
        if not self._active_tracks:
            for det in new_detections:
                track_id = self._next_track_id
                self._next_track_id += 1
                ground_pt = det.bbox.bottom_center
                self._active_tracks[track_id] = {
                    "bbox": det.bbox,
                    "history": [ground_pt],
                    "disappeared": 0,
                    "active_count": 1,
                    "confidence": det.confidence,
                }

            tracked_players = [
                TrackedPlayer(
                    track_id=tid,
                    bbox=tdata["bbox"],
                    ground_position=tdata["bbox"].bottom_center,
                    trajectory_history=list(tdata["history"]),
                    confidence=tdata["confidence"],
                    active_frames=tdata["active_count"],
                )
                for tid, tdata in self._active_tracks.items()
            ]
            return PlayerTrackingFrameResult(
                frame_number=frame.frame_number,
                timestamp_seconds=frame.timestamp_seconds,
                tracked_players=tracked_players,
            )

        # Match existing tracks with new detections using Greedy IoU Assignment
        track_ids = list(self._active_tracks.keys())
        matched_track_ids = set()
        matched_detection_indices = set()

        if new_detections and track_ids:
            # Build IoU matrix
            iou_matrix = np.zeros((len(track_ids), len(new_detections)), dtype=np.float32)
            for i, tid in enumerate(track_ids):
                for j, det in enumerate(new_detections):
                    iou_matrix[i, j] = compute_iou(self._active_tracks[tid]["bbox"], det.bbox)

            # Greedy matching
            while True:
                if iou_matrix.size == 0:
                    break
                max_val = np.max(iou_matrix)
                if max_val < self.iou_threshold:
                    break
                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                tid = track_ids[i]
                if tid not in matched_track_ids and j not in matched_detection_indices:
                    matched_track_ids.add(tid)
                    matched_detection_indices.add(j)

                    # Update track state
                    det = new_detections[j]
                    ground_pt = det.bbox.bottom_center
                    history = self._active_tracks[tid]["history"]
                    history.append(ground_pt)
                    if len(history) > self.max_history_len:
                        history.pop(0)

                    self._active_tracks[tid]["bbox"] = det.bbox
                    self._active_tracks[tid]["history"] = history
                    self._active_tracks[tid]["disappeared"] = 0
                    self._active_tracks[tid]["active_count"] += 1
                    self._active_tracks[tid]["confidence"] = det.confidence

                iou_matrix[i, :] = -1.0
                iou_matrix[:, j] = -1.0

        # Handle unmatched detections (spawn new track IDs)
        for j, det in enumerate(new_detections):
            if j not in matched_detection_indices:
                track_id = self._next_track_id
                self._next_track_id += 1
                ground_pt = det.bbox.bottom_center
                self._active_tracks[track_id] = {
                    "bbox": det.bbox,
                    "history": [ground_pt],
                    "disappeared": 0,
                    "active_count": 1,
                    "confidence": det.confidence,
                }

        # Handle unmatched tracks (increment disappeared counter and purge expired tracks)
        expired_track_ids = []
        for tid in track_ids:
            if tid not in matched_track_ids:
                self._active_tracks[tid]["disappeared"] += 1
                if self._active_tracks[tid]["disappeared"] > self.max_disappeared:
                    expired_track_ids.append(tid)

        for tid in expired_track_ids:
            del self._active_tracks[tid]

        # Construct final TrackedPlayer list
        tracked_players = [
            TrackedPlayer(
                track_id=tid,
                bbox=tdata["bbox"],
                ground_position=tdata["bbox"].bottom_center,
                trajectory_history=list(tdata["history"]),
                confidence=tdata["confidence"],
                active_frames=tdata["active_count"],
            )
            for tid, tdata in self._active_tracks.items()
            if tdata["disappeared"] == 0  # Only return active visible tracks
        ]

        return PlayerTrackingFrameResult(
            frame_number=frame.frame_number,
            timestamp_seconds=frame.timestamp_seconds,
            tracked_players=tracked_players,
        )
