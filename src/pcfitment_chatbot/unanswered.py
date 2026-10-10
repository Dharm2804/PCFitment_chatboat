import json
import threading
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
UNANSWERED_FILE = BASE_DIR / "data" / "unanswered_questions.json"

_lock = threading.Lock()


def log_unanswered(question: str, userid: int = 1, name: str = "Dharm") -> bool:
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

        # Existing questions (supports old plain-string entries and new dict entries)
        existing = [
            item.get("question", "") if isinstance(item, dict) else item
            for item in data
        ]

        if question.lower() in (q.lower() for q in existing):
            return False   # duplicate

        data.append(
            {
                "userid": userid,
                "name": name,
                "question": question,
                "datetime": datetime.now().isoformat(sep=" ", timespec="seconds"),
                "ismail_send": False,
            }
        )

        tmp = UNANSWERED_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(UNANSWERED_FILE)
        return True
    
def get_pending() -> list:
    """Return all entries where ismail_send is False."""
    with _lock:
        try:
            data = json.loads(UNANSWERED_FILE.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        return [
            item for item in data
            if isinstance(item, dict) and not item.get("ismail_send", False)
        ]


def mark_mail_sent(questions: list) -> None:
    """Set ismail_send = True for the given questions."""
    sent = {q.strip().lower() for q in questions}

    with _lock:
        try:
            data = json.loads(UNANSWERED_FILE.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return

        for item in data:
            if isinstance(item, dict) and item.get("question", "").strip().lower() in sent:
                item["ismail_send"] = True

        tmp = UNANSWERED_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(UNANSWERED_FILE)