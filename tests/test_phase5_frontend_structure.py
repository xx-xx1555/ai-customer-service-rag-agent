import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_frontend_files_are_present_and_valid_python():
    app_file = ROOT / "frontend" / "app.py"
    client_file = ROOT / "frontend" / "api_client.py"
    assert app_file.exists()
    assert client_file.exists()
    ast.parse(app_file.read_text(encoding="utf-8"))
    ast.parse(client_file.read_text(encoding="utf-8"))


def test_compose_contains_postgres_api_qdrant_and_frontend():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    for service in ["postgres:", "qdrant:", "ai-agent-api:", "frontend:"]:
        assert service in compose
