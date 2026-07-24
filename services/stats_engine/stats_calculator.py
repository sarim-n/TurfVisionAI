"""
Purpose: Post-match analytics engine calculating total running distance, sprint speeds, team possession percentages, and spatial heatmaps.
Dependencies: numpy, services.vision.models, shared.domain.entities, shared.schemas.events, shared.logging
Inputs: TrackedPlayer trajectories, TrackedBall positions, PitchHomography transforms
Outputs: AnalyticsReport DTO with heatmaps, distance leaderboards, and possession metrics
"""

from typing import Optional, Union
import numpy as np
from pydantic import BaseModel, Field
from services.ingestion.frame import FrameData
from services.vision.homography import PitchHomography
from services.vision.models import TrackedBall, TrackedPlayer
from shared.domain.entities import Point2D, TeamSide
from shared.logging import setup_logger

logger = setup_logger("stats_engine", service_name="stats")


class TeamStats(BaseModel):
    possession_percentage: float = 50.0
    total_distance_meters: float = 0.0


class PlayerStats(BaseModel):
    track_id: int
    team: TeamSide = TeamSide.UNKNOWN
    total_distance_meters: float = 0.0
    top_speed_m_per_sec: float = 0.0
    sprint_count: int = 0
    positions_pitch_meters: list[Point2D] = Field(default_factory=list)

    @property
    def distance_run_meters(self) -> float:
        return self.total_distance_meters


class AnalyticsReport(BaseModel):
    match_id: str
    home_stats: TeamStats = Field(default_factory=TeamStats)
    away_stats: TeamStats = Field(default_factory=TeamStats)
    home_possession_percent: float = 50.0
    away_possession_percent: float = 50.0
    total_match_distance_km: float = 0.0
    player_stats: list[PlayerStats] = Field(default_factory=list)
    heatmap_grid_home: list[list[int]] = Field(default_factory=list)
    heatmap_grid_away: list[list[int]] = Field(default_factory=list)


class MatchStatsEngine:
    """Accumulates player movement trajectories and calculates spatial analytics."""

    def __init__(self, match_id: str = "match_01", pitch_length_m: float = 105.0, pitch_width_m: float = 68.0):
        self.match_id = match_id
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m

        self.player_stats_map: dict[int, PlayerStats] = {}
        self.home_possession_frames = 0
        self.away_possession_frames = 0
        self.total_frames = 0

        # Spatial Heatmap Grids (52x34 grid cells for 105m x 68m pitch)
        self.grid_rows = 34
        self.grid_cols = 52
        self.heatmap_home = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)
        self.heatmap_away = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)

    def process_frame(
        self,
        tracked_players: list[TrackedPlayer],
        tracked_ball: Optional[TrackedBall],
        homography: PitchHomography,
        frame: FrameData,
    ) -> None:
        """Processes player tracking positions and updates distance/speed metrics."""
        self.total_frames += 1

        # Possession logic based on ball location / player proximity
        if tracked_ball is not None:
            ball_pitch = homography.pixel_to_pitch(tracked_ball.center)
            if ball_pitch.x < self.pitch_length_m / 2.0:
                self.home_possession_frames += 1
            else:
                self.away_possession_frames += 1

        for player in tracked_players:
            if player.track_id not in self.player_stats_map:
                self.player_stats_map[player.track_id] = PlayerStats(track_id=player.track_id)

            p_stats = self.player_stats_map[player.track_id]
            pitch_pt = homography.pixel_to_pitch(player.ground_position)

            if p_stats.positions_pitch_meters:
                last_pt = p_stats.positions_pitch_meters[-1]
                dist_m = float(np.hypot(pitch_pt.x - last_pt.x, pitch_pt.y - last_pt.y))

                # Filtering absurd telemetry jumps (> 15m in single frame)
                if dist_m < 15.0:
                    p_stats.total_distance_meters += dist_m
                    dt = 1.0 / (frame.metadata.fps if (frame.metadata and frame.metadata.fps > 0) else 30.0)
                    speed = dist_m / dt
                    if speed > p_stats.top_speed_m_per_sec:
                        p_stats.top_speed_m_per_sec = speed
                    if speed > 7.0:  # Sprint threshold 7m/s (~25km/h)
                        p_stats.sprint_count += 1

            p_stats.positions_pitch_meters.append(pitch_pt)

            # Spatial Heatmap Grid Binning
            c = int(min(self.grid_cols - 1, max(0, pitch_pt.x * (self.grid_cols / self.pitch_length_m))))
            r = int(min(self.grid_rows - 1, max(0, pitch_pt.y * (self.grid_rows / self.pitch_width_m))))
            if p_stats.team == TeamSide.HOME:
                self.heatmap_home[r, c] += 1
            elif p_stats.team == TeamSide.AWAY:
                self.heatmap_away[r, c] += 1

    def generate_match_report(self) -> AnalyticsReport:
        """Generates immutable AnalyticsReport snapshot."""
        total_p_frames = self.home_possession_frames + self.away_possession_frames
        if total_p_frames > 0:
            home_pct = (self.home_possession_frames / total_p_frames) * 100.0
            away_pct = (self.away_possession_frames / total_p_frames) * 100.0
        else:
            home_pct, away_pct = 50.0, 50.0

        p_list = list(self.player_stats_map.values())
        total_dist_m = sum(ps.total_distance_meters for ps in p_list)

        return AnalyticsReport(
            match_id=self.match_id,
            home_stats=TeamStats(possession_percentage=round(home_pct, 1)),
            away_stats=TeamStats(possession_percentage=round(away_pct, 1)),
            home_possession_percent=round(home_pct, 1),
            away_possession_percent=round(away_pct, 1),
            total_match_distance_km=round(total_dist_m / 1000.0, 2),
            player_stats=p_list,
            heatmap_grid_home=self.heatmap_home.tolist(),
            heatmap_grid_away=self.heatmap_away.tolist(),
        )

    # Aliases for API gateway and test suite
    generate_report = generate_match_report
    get_analytics_report = generate_match_report

    def generate_heatmap(self, track_id: Optional[int] = None, pitch_grid_size: tuple[int, int] = (105, 68)) -> np.ndarray:
        """Generates spatial heatmap density grid matrix."""
        grid_w, grid_h = pitch_grid_size
        grid = np.zeros((grid_h, grid_w), dtype=np.float32)

        if track_id is not None and track_id in self.player_stats_map:
            p_stats = self.player_stats_map[track_id]
            for pos in p_stats.positions_pitch_meters:
                c = int(min(grid_w - 1, max(0, pos.x * (grid_w / self.pitch_length_m))))
                r = int(min(grid_h - 1, max(0, pos.y * (grid_h / self.pitch_width_m))))
                grid[r, c] += 1.0
            return grid

        return self.heatmap_home
