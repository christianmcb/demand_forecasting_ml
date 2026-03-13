from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_prediction_endpoint():

    sample_request = {
        "store": 1,
        "item": 1,
        "date": "2017-01-01"
    }

    response = client.post("/predict", json=sample_request)

    assert response.status_code == 200
    assert "prediction" in response.json()