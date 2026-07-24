"""
TurfVision AI — Deterministic Event Engine Package
Consumes spatial tracking coordinates and evaluates physics/rules for Goals, Passes, Possession, and Scoreboards.
"""

from services.event_engine.goal_engine import GoalEngine
from services.event_engine.scoreboard import MatchPeriod, ScoreboardEngine

__all__ = ["GoalEngine", "ScoreboardEngine", "MatchPeriod"]
