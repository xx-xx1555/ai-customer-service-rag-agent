import os
import sys
from pathlib import Path

os.environ.setdefault("SKIP_INDEX_ON_STARTUP", "true")
os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("AUTO_CREATE_TABLES", "true")
os.environ.setdefault("AUTO_SEED_TICKETS", "true")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
