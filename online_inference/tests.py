from fastapi.testclient import TestClient
from app import app


def test_predict():
    with TestClient(app) as client:
        features = ['age', 'sex', 'cp', 'trestbps', 'chol',
                    'fbs', 'restecg', 'thalach', 'exang',
                    'oldpeak', 'slope', 'ca', 'thal']
        data = [59.0, 1.0, 2.0, 150.0, 212.0, 1.0,
                0.0, 157.0, 0.0, 1.6, 0.0, 0.0, 0.0]

        response = client.get(
            "/predict",
            json={"data": [data], "features": features},
        )
        assert response.status_code == 200
        assert response.json() == [{'condition': 0.0}]
