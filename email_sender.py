import os
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

from config import EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECIPIENT

logger = logging.getLogger(__name__)


def send_email_with_attachment(image_path: str, to_email: str | None = None) -> bool:
    """
    Send a security-alert email with an optional screenshot attachment.

    Credentials are read from environment variables (ALERT_SENDER_EMAIL,
    ALERT_EMAIL_PASSWORD, ALERT_TO_EMAIL).  If any are missing the call
    is a no-op and returns False so the caller can continue safely.
    """
    sender    = EMAIL_SENDER
    password  = EMAIL_PASSWORD
    recipient = to_email or EMAIL_RECIPIENT

    if not all([sender, password, recipient]):
        logger.warning(
            "Email alert skipped — credentials not configured. "
            "Set ALERT_SENDER_EMAIL, ALERT_EMAIL_PASSWORD, ALERT_TO_EMAIL in .env"
        )
        return False

    subject = "Security Alert: Unknown Person with Weapon Detected"
    body    = (
        "An unknown person carrying a weapon has been detected by SafeGuard.\n\n"
        "Please review the attached screenshot immediately and take appropriate action."
    )

    msg = MIMEMultipart()
    msg["From"]    = sender
    msg["To"]      = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as fh:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(fh.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={os.path.basename(image_path)}",
        )
        msg.attach(part)
    else:
        logger.warning(f"Screenshot not found at '{image_path}' — sending without attachment.")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.send_message(msg)
        logger.info(f"Alert email sent to {recipient}.")
        return True
    except Exception as exc:
        logger.error(f"Failed to send alert email: {exc}")
        return False
