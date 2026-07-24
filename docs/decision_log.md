# Architectural Decision Records (ADRs)

## ADR 001: Modular Monorepo Organization
- **Date**: 2026-07-24
- **Status**: Approved
- **Context**: TurfVision AI requires high cohesion between AI tracking output and backend event processing, while allowing services to deploy independently in containers.
- **Decision**: Adopt a Modular Monorepo layout (`services/`, `shared/`, `frontend/`, `infrastructure/`).
- **Consequences**: Enables shared Python domain entities (`shared/domain`) without package distribution overhead; keeps services cleanly partitioned.

## ADR 002: Deterministic Rule-Based Event Engine
- **Date**: 2026-07-24
- **Status**: Approved
- **Context**: Goal detection, possession changes, and passes must be 100% reliable and explainable for sports video analytics.
- **Decision**: Restrict AI/ML strictly to Perception (object detection and spatial tracking). Event detection (goals, passes, shots) is implemented via traditional physics and geometry algorithms.
- **Consequences**: Eliminates black-box ML hallucination in goal/score metrics; allows unit testing of event logic with synthetic trajectory data.

## ADR 003: Asynchronous Database Access with SQLAlchemy 2.0 & PostgreSQL
- **Date**: 2026-07-24
- **Status**: Approved
- **Context**: High-frequency telemetry updates require non-blocking DB persistence.
- **Decision**: Use SQLAlchemy 2.0 with `asyncpg` driver and PostgreSQL 16.
- **Consequences**: High throughput event writes without blocking FastAPI's async event loop.
