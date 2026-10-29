# Production Deployment Guide (Docker & Containerization)

DeepGuard supports containerized production deployment via Docker & Docker Compose.

---

## 1. Docker Compose Deployment

To build and launch the backend service, MongoDB, and frontend container:

```bash
docker-compose up --build -d
```

---

## 2. Docker Health Checks

- Backend API Health Check: `http://localhost:8000/health`
- OpenAPI Specification: `http://localhost:8000/docs`
