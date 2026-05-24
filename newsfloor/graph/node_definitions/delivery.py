"""
nodes/delivery.py

Sends the digest email via Google SMTP (Gmail).

No crew, no LLM — this node is pure I/O. The digest has already been
written and approved by the output supervisor. This node's only job is
to get it into the recipient's inbox reliably.

Authentication
──────────────
Gmail requires an App Password, not your regular account password.
Generate one at: https://myaccount.google.com/apppasswords
Set SMTP_PASSWORD in your .env file or Lambda environment variables.

Retry strategy
──────────────
SMTP send failures are almost always transient — throttling, brief
service interruptions, network timeouts. Tenacity handles retries
at the function level with exponential backoff. LangGraph does not
retry this node — by the time we're here the digest is approved and
we want it delivered, not re-evaluated.

Three attempts with backoff:
  Attempt 1: immediate
  Attempt 2: ~2 seconds
  Attempt 3: ~4 seconds

After three failures the error is logged and a DeliveryTaskResult
with sent=False is returned. The graph continues to the trend node
regardless — state should always be written even if delivery failed.
"""

from __future__ import annotations
import logging
import smtplib
import uuid
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import settings
from contracts.nodes import DeliveryTaskInput, DeliveryTaskResult

logger = logging.getLogger(__name__)

_SMTP_HOST = "smtp.gmail.com"
_SMTP_PORT = 465


def run(task_input: DeliveryTaskInput) -> DeliveryTaskResult:
    """
    Sends the digest email via Gmail SMTP and returns a DeliveryTaskResult.
    Never raises — returns sent=False with error details on failure.
    """
    logger.info({
        "node":      "delivery",
        "run_id":    task_input.run_id,
        "topic":     task_input.topic,
        "recipient": task_input.recipient_email,
    })

    try:
        message_id = _send_email(task_input)
        logger.info({
            "node":       "delivery",
            "run_id":     task_input.run_id,
            "message_id": message_id,
            "status":     "sent",
        })
        return DeliveryTaskResult(
            run_id     = task_input.run_id,
            sent       = True,
            message_id = message_id,
        )

    except Exception as e:
        logger.error({
            "node":    "delivery",
            "run_id":  task_input.run_id,
            "error":   str(e),
            "status":  "failed",
        })
        return DeliveryTaskResult(
            run_id = task_input.run_id,
            sent   = False,
            error  = str(e),
        )


# ---------------------------------------------------------------------------
# SMTP send — retried by tenacity, not by LangGraph
# ---------------------------------------------------------------------------

@retry(
    retry     = retry_if_exception_type(smtplib.SMTPException),
    stop      = stop_after_attempt(3),
    wait      = wait_exponential(multiplier=1, min=2, max=8),
    reraise   = True,
)
def _send_email(task_input: DeliveryTaskInput) -> str:
    """
    Sends the email via Gmail SMTP SSL. Returns a generated message ID.
    Retried up to 3 times on SMTPException with exponential backoff.
    reraise=True means the final failure propagates to run() where
    it is caught and returned as a DeliveryTaskResult with sent=False.
    """
    subject = _extract_subject(task_input.digest_html, task_input.topic)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = task_input.sender_email
    msg["To"]      = task_input.recipient_email

    msg.attach(MIMEText(_html_to_text(task_input.digest_html), "plain", "utf-8"))
    msg.attach(MIMEText(task_input.digest_html, "html", "utf-8"))

    with smtplib.SMTP_SSL(_SMTP_HOST, _SMTP_PORT) as server:
        server.login(task_input.sender_email, settings.smtp_app_token)
        server.sendmail(
            task_input.sender_email,
            task_input.recipient_email,
            msg.as_string(),
        )

    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_subject(digest_html: str, topic: str) -> str:
    """
    Extracts the subject line from the digest h1 tag if present,
    otherwise falls back to a formatted topic string.

    The writer is instructed to put the subject in <h1> — this pulls
    it out so the email subject matches the digest headline exactly.
    """
    import re
    match = re.search(r"<h1[^>]*>(.*?)</h1>", digest_html, re.IGNORECASE | re.DOTALL)
    if match:
        # Strip any nested HTML tags from the h1 content
        subject = re.sub(r"<[^>]+>", "", match.group(1)).strip()
        if subject:
            return f"Digest: {subject}"

    # Fallback — shouldn't happen if synthesis is working correctly
    return f"AI Agentic Engineering Digest — {topic.title()}"


def _html_to_text(html: str) -> str:
    """
    Produces a plain text fallback for email clients that don't render HTML.
    Simple tag stripping — not a full HTML-to-text converter, but sufficient
    for the digest structure we produce.
    """
    import re

    # Replace block elements with newlines before stripping tags
    text = re.sub(r"<h[1-6][^>]*>", "\n\n", html, flags=re.IGNORECASE)
    text = re.sub(r"</h[1-6]>",      "\n",   text, flags=re.IGNORECASE)
    text = re.sub(r"<p[^>]*>",       "\n",   text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>",      "\n",   text, flags=re.IGNORECASE)
    text = re.sub(r"<li[^>]*>",      "\n- ", text, flags=re.IGNORECASE)

    # Strip all remaining tags
    text = re.sub(r"<[^>]+>", "", text)

    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}",  " ",    text)

    return text.strip()
