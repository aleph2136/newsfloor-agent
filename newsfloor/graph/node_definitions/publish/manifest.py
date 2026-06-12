"""
publish/manifest.py

Loads and updates the articles/manifest.json stored in S3.
"""

from __future__ import annotations
import json
import logging

from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def _load_manifest(s3, bucket: str) -> list[dict]:
    try:
        resp = s3.get_object(Bucket=bucket, Key="articles/manifest.json")
        return json.loads(resp["Body"].read().decode("utf-8"))
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("NoSuchKey", "404", "AccessDenied"):
            if code == "AccessDenied":
                logger.warning({
                    "node": "publish",
                    "message": "AccessDenied reading manifest — check s3:GetObject on articles/* in IAM policy",
                })
            return []
        raise
    except Exception as e:
        logger.warning({"node": "publish", "message": f"Manifest load failed: {e} — starting fresh"})
        return []


def _update_manifest(manifest: list[dict], date_str: str, title: str, excerpt: str) -> list[dict]:
    manifest = [e for e in manifest if e.get("date") != date_str]
    manifest.insert(0, {
        "date":    date_str,
        "title":   title,
        "excerpt": excerpt,
        "url":     f"/articles/{date_str}.html",
    })
    manifest.sort(key=lambda e: e.get("date", ""), reverse=True)
    return manifest
