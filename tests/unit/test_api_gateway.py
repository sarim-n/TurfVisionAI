"""
Unit tests for FastAPI API Gateway Service.
"""

import pytest
from fastapi.testclient import TestClient
from services.api_gateway.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "TurfVision AI"


def test_get_match_state(client):
    response = client.get("/api/v1/matches/match_live_01/state")
    assert response.status_code == 200
    data = response.json()
    assert data["match_id"] == "match_live_01"
    assert "home_score" in data
    assert "away_score" in data


def test_get_match_events(client):
    response = client.get("/api/v1/matches/match_live_01/events")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_get_match_analytics(client):
    response = client.get("/api/v1/matches/match_live_01/analytics")
    assert response.status_code == 200
    data = response.json()
    assert data["match_id"] == "match_live_01"
    assert "home_stats" in data
    assert "away_stats" in data
