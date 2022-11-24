from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_read_main():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"message": "Service is ready to accept requests!"}


def test_correct_predict():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            data='{"data": [{"age":24, "sex":1, "cp":1, "trestbps":100, "chol":228, "fbs":0, "restecg":0, "thalach":100, "exang":0, "oldpeak":228.322, "slope":0, "ca":2, "thal":2}]}'
        )
        print(response.json())
        assert response.status_code == 200
        assert "target" in response.json()
        assert len(response.json()["target"]) == 2


def test_empty_input_predict():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            data='{"data": []}'
        )
        assert response.status_code == 200
        assert "target" in response.json()
        assert len(response.json()["target"]) == 0


def test_incorrect_input_predict():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            data='[{"age":228, "sex":1, "cp":1, "trestbps":100, "chol":228, "fbs":0, "restecg":0, "thalach":100, "exang":0, "oldpeak":228.322, "slope":0, "ca":2, "thal":2}]'
        )
        assert response.status_code == 400
        
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            data='[{"age":asf, "sex":1, "cp":1, "trestbps":100, "chol":228, "fbs":0, "restecg":0, "thalach":100, "exang":0, "oldpeak":228.322, "slope":0, "ca":2, "thal":2}]'
        )
        assert response.status_code == 400