from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_vector_search():
    response = client.post(
        "/api/documents/vector/search",
        json={"question": "RAG 有哪些步骤", "top_k": 3, "min_score": 0.0, "mode": "bm25"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data


def test_agent_ticket():
    response = client.post(
        "/api/agent/run",
        json={"question": "有哪些未解决工单？", "top_k": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "ticket_analysis"
