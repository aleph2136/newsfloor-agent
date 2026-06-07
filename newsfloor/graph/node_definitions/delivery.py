"""
nodes/delivery.py

Sends the digest email via Google SMTP (Gmail).

No crew, no LLM — this node is pure I/O. The digest has already been
written and approved by the output supervisor. This node's only job is
to get it into the recipient's inbox reliably.

Email format
────────────
When digest_json is available (the new structured format), the email is
sent as a minimal plain-text summary optimized for mobile reading:

  Daily Digest Takeaway: [Article Title]

  - [Block Title]: [Tier 1 Hook]
    * [Bold Anchor] -> [Bullet Body]
    * ...

  Read full technical deep dives: [article_url]

When digest_json is absent (legacy fallback), the existing HTML-to-text
conversion is used.

Authentication
──────────────
Gmail requires an App Password, not your regular account password.
Generate one at: https://myaccount.google.com/apppasswords
Set SMTP_PASSWORD in your .env file or Lambda environment variables.

Retry strategy
──────────────
SMTP send failures are almost always transient. Tenacity handles retries
at the function level with exponential backoff.

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
import re
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
from contracts.nodes import DeliveryTaskInput, DeliveryTaskResult, DigestStructured

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
        "has_json":  task_input.digest_json is not None,
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
    subject = _extract_subject(task_input.digest_html, task_input.topic, task_input.digest_json)

    if task_input.digest_json is not None:
        plain_body = _json_to_plain_text(task_input.digest_json, task_input.article_url)
    else:
        plain_body = _html_to_text(task_input.digest_html)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = task_input.sender_email
    msg["To"]      = task_input.recipient_email

    msg.attach(MIMEText(plain_body, "plain", "utf-8"))
    msg.attach(MIMEText(task_input.digest_html, "html", "utf-8"))

    server = smtplib.SMTP_SSL(_SMTP_HOST, _SMTP_PORT)
    try:
        server.login(task_input.sender_email, settings.smtp_app_token)
        server.sendmail(
            task_input.sender_email,
            task_input.recipient_email,
            msg.as_string(),
        )
    finally:
        try:
            server.quit()
        except smtplib.SMTPException:
            pass

    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_subject(
    digest_html: str,
    topic: str,
    digest_json: DigestStructured | None = None,
) -> str:
    """
    Extracts the subject line from the structured digest or the HTML h1 tag.
    Falls back to a formatted topic string if neither is available.
    """
    if digest_json is not None:
        title = digest_json.metadata.title.strip()
        if title:
            return f"Digest: {title}"

    match = re.search(r"<h1[^>]*>(.*?)</h1>", digest_html, re.IGNORECASE | re.DOTALL)
    if match:
        subject = re.sub(r"<[^>]+>", "", match.group(1)).strip()
        if subject:
            return f"Digest: {subject}"

    return f"AI Agentic Engineering Digest — {topic.title()}"


def _json_to_plain_text(digest_json: DigestStructured, article_url: str = "") -> str:
    """
    Produces a concise plain-text email from the structured digest JSON.
    Formatted for mobile reading — no markdown, no code blocks, no diagrams.

    Format:
      Daily Digest Takeaway: [Title]

      - [Block Title]: [Tier 1 Hook]
        * [Bold Anchor] -> [Bullet Body]
        ...

      Read full technical deep dives: [url]
    """
    lines: list[str] = [
        f"Daily Digest Takeaway: {digest_json.metadata.title}",
        "",
    ]

    if digest_json.metadata.overall_trend_context:
        lines.append(f"Trend: {digest_json.metadata.overall_trend_context}")
        lines.append("")

    for block in digest_json.content_blocks:
        lines.append(f"- {block.section_title}: {block.tier_1_hook}")
        for bullet in block.tier_2_bullets:
            anchor, body = _split_bold_bullet(bullet)
            if anchor and body:
                lines.append(f"  * [{anchor}] -> {body}")
            else:
                lines.append(f"  * {bullet}")
        lines.append("")

    if article_url:
        lines.append(f"Read full technical deep dives: {article_url}")

    return "\n".join(lines).strip()


def _split_bold_bullet(bullet: str) -> tuple[str, str]:
    """
    Splits a **bold anchor** bullet into (anchor, body) for plain-text formatting.
    Returns ("", "") if the bullet doesn't follow the expected format.
    """
    match = re.match(r"\*\*(.+?)\*\*\s*(.*)", bullet, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", ""


def _html_to_text(html: str) -> str:
    """
    Produces a plain text fallback for email clients that don't render HTML.
    Used when digest_json is not available (legacy path).
    """
    text = re.sub(r"<h[1-6][^>]*>", "\n\n", html, flags=re.IGNORECASE)
    text = re.sub(r"</h[1-6]>",      "\n",   text, flags=re.IGNORECASE)
    text = re.sub(r"<p[^>]*>",       "\n",   text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>",      "\n",   text, flags=re.IGNORECASE)
    text = re.sub(r"<li[^>]*>",      "\n- ", text, flags=re.IGNORECASE)

    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}",  " ",    text)

    return text.strip()
