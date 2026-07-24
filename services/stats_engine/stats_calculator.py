"""
Purpose: Match Statistics Engine aggregating spatial telemetry, possession, and distance metrics.
Dependencies: math, numpy, services.ingestion.frame, services.vision.models, services.vision.homography, services.stats_engine.models, shared.domain.entities, shared.logging
Inputs: TrackedPlayer list, TrackedBall, PitchHomography, and FrameData
Outputs: MatchAnalyticsReport and 2D spatial heatmap density matrices
"""

import math
from typing import Optional
import numpy as np
from services.ingestion.frame import FrameData
from services.stats_engine.models import MatchAnalyticsReport, PlayerStats, TeamStats
from services.vision.homography import PitchHomography
from services.vision.models import TrackedBall, TrackedPlayer
from shared.domain.entities import TeamSide
from shared.logging import setup_logger

logger = setup_logger("stats_engine", service_name="stats_engine")


class MatchStatsEngine:
    """Aggregates player distance, speed, team possession, and spatial heatmaps continuously per frame."""

    def __init__(self, match_id: str = "match_default", possession_proximity_px: float = 60.0):
        self.match_id = match_id
        self.possession_proximity_px = possession_proximity_px

        self._home_possession_frames = 0
        self._away_possession_frames = 0
        self._total_possession_frames = 0

        # Player telemetry: { track_id: {"distance_m": float, "last_m": Point2D, "speeds_kmh": list[float], "team": TeamSide} }
        self._player_telemetry: dict[int, dict] = {}
        # Trajectory history for heatmaps: { track_id: list[(x_m, y_m)] }
        self._pitch_positions_history: dict[int, list[tuple[float, float]]] = {}

    def reset(self) -> None:
        """Resets statistics engine state."""
        self._home_possession_frames = 0
        self._away_possession_frames = 0
        self._total_possession_frames = 0
        self._player_telemetry.clear()
        self._pitch_positions_history.clear()
        logger.info("MatchStatsEngine state reset.")

    def process_frame(
        self,
        tracked_players: list[TrackedPlayer],
        tracked_ball: Optional[TrackedBall],
        homography: Optional[PitchHomography],
        frame: FrameData,
    ) -> None:
        """Processes single frame telemetry updates."""
        dt = 1.0 / frame.metadata.fps

        # 1. Update Player Distance & Speeds
        for player in tracked_players:
            tid = player.track_id
            curr_px = player.ground_position

            # Determine metric position if homography is available
            if homography is not None:
                pitch_pt = homography.pixel_to_pitch(curr_px)
                curr_x_m, curr_y_m = pitch_pt.x_meters, pitch_pt.y_meters
            else:
                curr_x_m, curr_y_m = curr_px.x / 30.0, curr_px.y / 30.0

            if tid not in self._player_telemetry:
                self._player_telemetry[tid] = {
                    "distance_m": 0.0,
                    "last_m": (curr_x_m, curr_y_m),
                    "speeds_kmh": [],
                    "team": TeamSide.UNKNOWN,
                }
                self._pitch_positions_history[tid] = []
            else:
                last_x_m, last_y_m = self._player_telemetry[tid]["last_m"]
                dx = curr_x_m - last_x_m
                dy = curr_y_m - last_y_m
                step_dist_m = math.sqrt(dx * dx + dy * dy)

                # Filter out noisy spatial jumps (> 15 m per frame)
                if step_dist_m <= 15.0:
                    self._player_telemetry[tid]["distance_m"] += step_dist_m
                    speed_m_per_s = step_dist_m / dt if dt > 0 else 0.0
                    speed_kmh = speed_m_per_s * 3.6
                    self._player_telemetry[tid]["speeds_kmh"].append(speed_kmh)

                self._player_telemetry[tid]["last_m"] = (curr_x_m, curr_y_m)

            self._pitch_positions_history[tid].append((curr_x_m, curr_y_m))

        # 2. Update Possession Attribution
        if tracked_ball is not None:
            bx, by = tracked_ball.center.x, tracked_ball.center.y
            closest_player: Optional[TrackedPlayer] = None
            min_dist = float("inf")

            for player in tracked_players:
                dx = player.ground_position.x - bx
                dy = player.ground_position.y - by
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < min_dist and dist <= self.possession_proximity_px:
                    min_dist = dist
                    closest_player = player

            if closest_player is not None:
                # Basic heuristic: Left side of pitch = HOME, Right side = AWAY
                if closest_player.ground_position.x < (frame.width / 2.0):
                    self._home_possession_frames += 1
                else:
                    self._away_possession_frames += 1
                self._total_possession_frames += 1

    def generate_heatmap(
        self,
        track_id: Optional[int] = None,
        pitch_grid_size: tuple[int, int] = (105, 68),
    ) -> np.ndarray:
        """Generates a 2D spatial density heatmap matrix across pitch grid dimensions."""
        grid_w, grid_h = pitch_grid_size
        heatmap = np.zeros((grid_h, grid_w), dtype=np.float32)

        target_tids = [track_id] if track_id in self._pitch_positions_history else list(self._pitch_positions_history.keys())

        for tid in target_tids:
            for xm, ym in self._pitch_positions_history.get(tid, []):
                gx = int(min(grid_w - 1, max(0, xm)))
                gy = int(min(grid_h - 1, max(0, ym)))
                heatmap[gy, gx] += 1.0

        # Normalize density matrix
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap /= max_val

        return heatmap

    def get_analytics_report(self) -> MatchAnalyticsReport:
        """Generates structured MatchAnalyticsReport."""
        if self._total_possession_frames > 0:
            home_poss_pct = round((self._home_possession_frames / self._total_possession_frames) * 100.0, 1)
            away_poss_pct = round(100.0 - home_poss_pct, 1)
        else:
            home_poss_pct, away_poss_pct = 50.0, 50.0

        player_stats_list: list[PlayerStats] = []
        total_home_dist = 0.0
        total_away_dist = 0.0

        for tid, tdata in self._player_telemetry.items():
            dist = round(tdata["distance_m"], 1)
            speeds = tdata["speeds_kmh"]
            avg_speed = round(sum(speeds) / len(speeds), 1) if speeds else 0.0
            max_speed = round(max(speeds), 1) if speeds else 0.0

            player_stats_list.append(
                PlayerStats(
                    track_id=tid,
                    team=tdata["team"],
                    distance_run_meters=dist,
                    avg_speed_kmh=avg_speed,
                    max_speed_kmh=max_speed,
                )
            )

        home_stats = TeamStats(
            team=TeamSide.HOME,
            possession_percentage=home_poss_pct,
            total_distance_run_meters=round(total_home_dist, 1),
        )

        away_stats = TeamStats(
            team=TeamSide.AWAY,
            possession_percentage=away_poss_pct,
            total_distance_run_meters=round(total_away_dist, 1),
        )

        return MatchAnalyticsReport(
            match_id=self.match_id,
            home_stats=home_stats,
            away_stats=away_stats,
            player_stats=player_stats_list,
        )
