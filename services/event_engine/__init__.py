"""
TurfVision AI — Deterministic Event Engine Package
Consumes spatial tracking coordinates and evaluates physics/rules for Goals, Passes, Possession, and Shots.
"""

from services.event_engine.goal_engine import GoalEngine

__all__ = ["GoalEngine"]
