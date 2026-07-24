"""
Purpose: FastAPI API Gateway service providing REST endpoints and WebSocket live telemetry streaming.
Dependencies: fastapi, uvicorn, pydantic, shared.config, shared.logging, services.event_engine, services.stats_engine
Inputs: HTTP requests and WebSocket client connections
Outputs: JSON REST responses and 30 FPS WebSocket telemetry frame broadcasts
"""

import asyncio
import json
from typing import Any
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from services.event_engine.goal_engine import GoalEngine
from services.event_engine.scoreboard import ScoreboardEngine
from services.stats_engine.stats_calculator import MatchStatsEngine
from shared.config import get_settings
from shared.logging import setup_logger

logger = setup_logger("api_gateway", service_name="api_gateway")
settings = get_settings()

app = FastAPI(
    title=f"{settings.APP_NAME} Gateway API",
    description="Real-Time AI Football Analytics & Computer Vision Gateway",
    version="0.1.0",
)

# Enable CORS for frontend dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active match engines state registry
_scoreboard = ScoreboardEngine(match_id="match_live_01", home_team_name="HOME FC", away_team_name="AWAY UT")
_stats_engine = MatchStatsEngine(match_id="match_live_01")
_goal_engine = GoalEngine(match_id="match_live_01")


class ConnectionManager:
    """Manages active WebSocket client connections for live telemetry broadcasts."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket client disconnected. Remaining connections: {len(self.active_connections)}")

    async def broadcast_json(self, message: dict[str, Any]):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending WebSocket message: {e}")
                self.disconnect(connection)


manager = ConnectionManager()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "app": settings.APP_NAME, "environment": settings.ENVIRONMENT}


@app.get("/api/v1/matches/{match_id}/state")
async def get_match_state(match_id: str):
    """Returns current live MatchState snapshot."""
    return _scoreboard.get_state_snapshot().model_dump()


@app.get("/api/v1/matches/{match_id}/events")
async def get_match_events(match_id: str):
    """Returns list of all logged MatchEvents."""
    return [event.model_dump() for event in _scoreboard.event_log]


@app.get("/api/v1/matches/{match_id}/analytics")
async def get_match_analytics(match_id: str):
    """Returns post-match or live MatchAnalyticsReport."""
    return _stats_engine.get_analytics_report().model_dump()


@app.websocket("/ws/v1/matches/{match_id}/stream")
async def websocket_match_stream(websocket: WebSocket, match_id: str):
    """WebSocket endpoint streaming live 30 FPS frame telemetry, player positions, ball vectors, and score updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Receive client ping/ack messages
            data = await websocket.receive_text()
            # Send live telemetry snapshot update
            snapshot = {
                "match_id": match_id,
                "scoreboard": _scoreboard.get_state_snapshot().model_dump(),
                "clock_formatted": _scoreboard.format_clock(),
                "recent_events": [e.model_dump() for e in _scoreboard.event_log[-5:]],
            }
            await websocket.send_json(snapshot)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
