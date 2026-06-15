from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_encode_text():
    response = client.get("/embedding/Hello%20World")
    assert response.status_code == 200
    data = response.json()
    assert "embedding_binary" in data
    assert isinstance(data["embedding_binary"], list)
