from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_ticket_crud_and_filters():
    create_response = client.post(
        "/api/tickets",
        json={
            "ticket_id": "T-PHASE4-TEST",
            "user_id": "U-PHASE4",
            "issue_type": "测试问题",
            "description": "用于验证 PostgreSQL 持久化层的测试工单",
            "status": "待处理",
            "created_at": "2026-07-13T10:00:00",
            "resolved_hours": 0,
            "satisfaction": 2,
        },
    )
    assert create_response.status_code == 201
    assert create_response.json()["ticket_id"] == "T-PHASE4-TEST"

    list_response = client.get(
        "/api/tickets",
        params={"issue_type": "测试问题", "satisfaction_lte": 2},
    )
    assert list_response.status_code == 200
    assert list_response.json()["total"] == 1

    update_response = client.patch(
        "/api/tickets/T-PHASE4-TEST",
        json={"status": "已解决", "resolved_hours": 1.5, "satisfaction": 5},
    )
    assert update_response.status_code == 200
    assert update_response.json()["status"] == "已解决"

    delete_response = client.delete("/api/tickets/T-PHASE4-TEST")
    assert delete_response.status_code == 204
    assert client.get("/api/tickets/T-PHASE4-TEST").status_code == 404


def test_dashboard_contains_persistent_statistics():
    response = client.get("/api/tickets/dashboard", params={"days": 30})
    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["total_tickets"] >= 12
    assert data["trends"]
    assert "period_comparison" in data


def test_duplicate_ticket_id_returns_conflict():
    payload = {
        "ticket_id": "T-DUPLICATE",
        "user_id": "U-DUP",
        "issue_type": "重复测试",
        "description": "第一次创建",
        "status": "待处理",
        "created_at": "2026-07-13T10:00:00",
        "resolved_hours": 0,
        "satisfaction": 3,
    }
    assert client.post("/api/tickets", json=payload).status_code == 201
    assert client.post("/api/tickets", json=payload).status_code == 409
    assert client.delete("/api/tickets/T-DUPLICATE").status_code == 204


def test_chat_session_and_evaluation_history_are_persisted():
    from app.services.persistence_service import (
        delete_chat_session,
        get_chat_session,
        get_evaluation_run,
        save_chat_exchange,
        save_evaluation_run,
    )

    session_id = save_chat_exchange(
        question="退款规则是什么？",
        answer="请先核对订单信息。",
        sources=["manual.md#chunk-1"],
    )
    conversation = get_chat_session(session_id)
    assert conversation is not None
    assert [item["role"] for item in conversation["messages"]] == ["user", "assistant"]
    assert conversation["messages"][1]["sources"] == ["manual.md#chunk-1"]

    run_id = save_evaluation_run(
        evaluation_type="unit_test",
        configuration={"top_k": 4},
        result={"total": 1, "hit_rate_at_k": 1.0, "cases": [{"question": "test"}]},
    )
    evaluation = get_evaluation_run(run_id)
    assert evaluation is not None
    assert evaluation["metrics"]["hit_rate_at_k"] == 1.0
    assert delete_chat_session(session_id) is True
