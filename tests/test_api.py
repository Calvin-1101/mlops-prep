"""
this script tests whether our main.py FastAPI server is correct, checking if endpoints
return the correct & expected responses

a fake model is intentionally used because this is part of the CI (continuous integration) pipeline,
where we will use GitHub Actions to run tests on a fresh machine -> GitHub Actions will not have our actual model from MLFLow. 

"""
import numpy as np 
import pytest 
from fastapi.testclient import TestClient

import api.main as main 

VALID_HOUSE = {
    "MedInc": 8.3252, "HouseAge": 41, "AveRooms": 6.98, "AveBedrms": 1.02,
    "Population": 322, "AveOccup": 2.56, "Latitude": 37.88, "Longitude": -122.23,
}

class FakeModel:
    def predict(self, df):
        return np.full(len(df), 2.5)



@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "load_model", lambda: FakeModel())
    with TestClient(main.app) as c:  # `with` runs the startup (lifespan) code
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_ready(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_single(client):
    r = client.post("/predict", json={"features": [VALID_HOUSE]})
    assert r.status_code == 200
    assert r.json()["preds"] == [2.5]


def test_predict_batch(client):
    r = client.post("/predict", json={"features": [VALID_HOUSE] * 3})
    assert r.status_code == 200
    assert len(r.json()["preds"]) == 3


def test_missing_field_rejected(client):
    bad = {k: v for k, v in VALID_HOUSE.items() if k != "MedInc"}
    r = client.post("/predict", json={"features": [bad]})
    assert r.status_code == 422


def test_out_of_range_rejected(client):
    r = client.post("/predict", json={"features": [{**VALID_HOUSE, "Latitude": 51.5}]})
    assert r.status_code == 422


def test_empty_request_rejected(client):
    r = client.post("/predict", json={"features": []})
    assert r.status_code == 422





@pytest.fixture
def broken_client(monkeypatch):
    def fail_to_load():
        raise RuntimeError("MLflow unreachable")

    monkeypatch.setattr(main, "load_model", fail_to_load)
    with TestClient(main.app) as c:
        yield c


def test_ready(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_not_ready_when_model_fails_to_load(broken_client):
    assert broken_client.get("/health").status_code == 200   # alive
    assert broken_client.get("/ready").status_code == 503    # but not ready


def test_predict_returns_503_when_not_ready(broken_client):
    r = broken_client.post("/predict", json={"features": [VALID_HOUSE]})
    assert r.status_code == 503