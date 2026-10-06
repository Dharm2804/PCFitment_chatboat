import os
import smtplib
import threading
from datetime import datetime
from email.message import EmailMessage


def _send(question: str):
    # Read settings at send time, so load_dotenv() has already run
    host = os.getenv("MAIL_HOST", "smtp.gmail.com")
    port = int(os.getenv("MAIL_PORT", "587"))
    user = os.getenv("MAIL_USER")
    password = os.getenv("MAIL_PASSWORD")
    to = os.getenv("MAIL_TO", user)

    if not (user and password and to):
        print("Mail not configured (MAIL_USER / MAIL_PASSWORD / MAIL_TO missing).")
        return

    msg = EmailMessage()
    msg["Subject"] = "PCFitment bot: unanswered question"
    msg["From"] = user
    msg["To"] = to
    msg.set_content(
        "The chatbot could not answer this question:\n\n"
        f"{question}\n\n"
        f"Time: {datetime.now():%Y-%m-%d %H:%M:%S}"
    )

    try:
        with smtplib.SMTP(host, port, timeout=15) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)
        print("Unanswered-question email sent.")
    except Exception as error:
        print(f"Failed to send email: {error}")


def notify_unanswered(question: str):
    """Send the email in the background so the chat doesn't wait."""
    threading.Thread(target=_send, args=(question,), daemon=True).start()