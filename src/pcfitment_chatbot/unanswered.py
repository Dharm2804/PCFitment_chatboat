import json
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
UNANSWERED_FILE = BASE_DIR / "data" / "unanswered_questions.json"

_lock = threading.Lock()


def log_unanswered(question: str) -> bool:
    """Save an unanswered question. Returns True if it was new, False if skipped."""
    question = question.strip()
    if not question:
        return False

    with _lock:
        UNANSWERED_FILE.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = json.loads(UNANSWERED_FILE.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                data = []
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        data = [item["question"] if isinstance(item, dict) else item for item in data]

        if question.lower() in (q.lower() for q in data):
            return False   # duplicate

        data.append(question)

        tmp = UNANSWERED_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(UNANSWERED_FILE)
        return True