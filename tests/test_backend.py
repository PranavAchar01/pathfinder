import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_edge_config():
    cfg = client.get("/api/config").json()
    assert cfg["llm_model"]
    assert cfg["collision_warn_m"] > cfg["collision_stop_m"]


def test_identify_mock():
    r = client.post("/api/identify", json={"query": "what is this", "label_guess": "kettle"})
    assert r.status_code == 200
    body = r.json()
    assert "kettle" in body["query"]
    assert len(body["results"]) >= 1


def test_rsi_telemetry_and_cycle():
    items = [{"label": "scooter", "confidence": 0.3, "known": False, "distance_m": 1.5}]
    r = client.post("/api/rsi/telemetry", json={"items": items})
    assert r.status_code == 200
    assert r.json()["ingested"] == 1

    cycle = client.post("/api/rsi/cycle").json()
    assert "weak_labels" in cycle
