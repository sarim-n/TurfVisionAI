"""
TurfVision AI — Master Processing Pipeline Package
Orchestrates Video Ingestion, YOLO Detectors, MOT & Kalman Trackers, Homography, Goal Detector, Scoreboard Engine, and Stats Engine.
"""

from services.pipeline.models import PipelineConfig, PipelineFrameResult
from services.pipeline.pipeline_runner import TurfVisionPipeline

__all__ = ["TurfVisionPipeline", "PipelineConfig", "PipelineFrameResult"]
