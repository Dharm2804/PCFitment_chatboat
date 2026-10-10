import os
import smtplib
import threading
import time
from datetime import datetime
from email.message import EmailMessage

from unanswered import get_pending, mark_mail_sent

_scheduler_started = False


def _send(pending: list) -> bool:
    """Send one email with all pending questions. Returns True on success."""
    host = os.getenv("MAIL_HOST", "smtp.gmail.com")
    port = int(os.getenv("MAIL_PORT", "587"))
    user = os.getenv("MAIL_USER")
    password = os.getenv("MAIL_PASSWORD")
    to = os.getenv("MAIL_TO", user)

    if not (user and password and to):
        print("Mail not configured (MAIL_USER / MAIL_PASSWORD / MAIL_TO missing).")
        return False

    lines = []
    for i, item in enumerate(pending, 1):
        lines.append(
            f"{i}. User ID : {item.get('userid')}\n"
            f"   Name    : {item.get('name')}\n"
            f"   Question: {item.get('question')}\n"
            f"   Asked at: {item.get('datetime')}\n"
        )

    msg = EmailMessage()
    msg["Subject"] = f"PCFitment bot: {len(pending)} unanswered question(s)"
    msg["From"] = user
    msg["To"] = to
    msg.set_content(
        "The chatbot could not answer these questions:\n\n"
        + "\n".join(lines)
        + f"\nSent: {datetime.now():%Y-%m-%d %H:%M:%S}"
    )

    try:
        with smtplib.SMTP(host, port, timeout=15) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)
        print(f"Unanswered-question email sent ({len(pending)} questions).")
        return True
    except Exception as error:
        print(f"Failed to send email: {error}")
        return False


def send_pending_mail() -> None:
    """Send all ismail_send=False questions, then mark them True."""
    pending = get_pending()
    if not pending:
        return
    if _send(pending):
        mark_mail_sent([item["question"] for item in pending])


def _parse_send_time():
    raw = os.getenv("MAIL_SEND_TIME", "21:00")
    try:
        hour, minute = raw.split(":")
        return datetime.now().replace(
            hour=int(hour), minute=int(minute), second=0, microsecond=0
        ).time()
    except ValueError:
        print(f"Invalid MAIL_SEND_TIME '{raw}', using 21:00")
        return datetime.now().replace(hour=21, minute=0, second=0, microsecond=0).time()


def _scheduler_loop():
    last_sent_date = None
    while True:
        now = datetime.now()
        send_time = _parse_send_time()   # re-read so .env changes apply after restart

        # Once per day, at or after the send time
        if now.time() >= send_time and last_sent_date != now.date():
            send_pending_mail()
            last_sent_date = now.date()

        time.sleep(60)


def start_mail_scheduler() -> None:
    """Start the background scheduler once per process."""
    global _scheduler_started
    if _scheduler_started:
        return
    _scheduler_started = True
    threading.Thread(target=_scheduler_loop, daemon=True).start()
    print("Mail scheduler started.")