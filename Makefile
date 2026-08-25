.PHONY: up down logs test api frontend clean

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=200

test:
	pytest -q

api:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	API_BASE_URL=http://localhost:8000 streamlit run frontend/app.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
