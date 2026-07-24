# TurfVision AI ⚽🤖

> Production-grade AI Football Analytics & Real-Time Computer Vision Platform.

TurfVision AI delivers automated player tracking, ball tracking, goal detection, possession analysis, and post-match analytics for football turfs, sports academies, and broadcaster feeds.

---

## 🏗 System Architecture

The project is organized into decoupled microservices adhering to Clean Architecture principles:

- **`services/ingestion`**: Video file and RTSP stream demuxing pipeline.
- **`services/vision`**: PyTorch & Ultralytics YOLO object detection with ByteTrack tracking.
- **`services/event_engine`**: Deterministic rule-based sports event processor (Goals, Passes, Shots).
- **`services/stats_engine`**: Spatial telemetry and analytical reporting engine.
- **`services/api_gateway`**: FastAPI REST API and WebSocket live streamer.
- **`shared`**: Domain entities, Pydantic DTOs, central configuration, and JSON logging.
- **`frontend`**: React + TypeScript + Vite interactive match dashboard.

---

## 🚀 Quickstart (Local Development)

### 1. Requirements
- Python 3.11+
- Docker & Docker Compose
- Node.js 18+ (for frontend)

### 2. Environment Setup
```bash
cp .env.example .env
```

### 3. Install Python Dependencies
```bash
pip install -e .[dev]
```

### 4. Run Core Services Stack (PostgreSQL & Redis)
```bash
docker-compose -f infrastructure/docker-compose.yml up -d
```

---

## 🧪 Running Tests
```bash
pytest tests/unit
```

---

## 📋 Milestones Roadmap
- [x] **Milestone 1**: Project Architecture & Repository Setup
- [ ] **Milestone 2**: Video Ingestion Engine
- [ ] **Milestone 3**: Player Detection (YOLO)
- [ ] **Milestone 4**: Football Detection
- [ ] **Milestone 5**: Player Tracking (ByteTrack)
- [ ] **Milestone 6**: Ball Tracking & Trajectory Estimation
- [ ] **Milestone 7**: Camera Calibration (Homography / Pitch Projection)
- [ ] **Milestone 8**: Goal Line Spatial Detection
- [ ] **Milestone 9**: Goal Detection Engine
- [ ] **Milestone 10**: Scoreboard Engine
- [ ] **Milestone 11**: Match Statistics Engine
- [ ] **Milestone 12**: Dashboard Frontend
- [ ] **Milestone 13**: Database Persistence & Migrations
- [ ] **Milestone 14**: Optimization & GPU Acceleration
- [ ] **Milestone 15**: Deployment & Cloud Packaging
