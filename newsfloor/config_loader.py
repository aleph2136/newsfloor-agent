"""
config_loader.py

Loads runtime configuration from newsfloor/config_data/ JSON files.

To customize the digest for a different topic domain, audience, or set of
sources, edit the three JSON files in config_data/ — no Python code changes
required.

  config_data/topics.json   — list of topic strings the strategist rotates through
  config_data/sources.json  — list of RSS/Atom feed URLs to harvest from
  config_data/profile.json  — engineer profile fields (name, focus areas, etc.)

Results are cached via lru_cache: each file is read once per Lambda cold start
and reused across all warm invocations.
"""
from __future__ import annotations
import json
import logging
from functools import lru_cache
from pathlib import Path

from contracts.nodes import EngineerProfile

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent / "config_data"


@lru_cache(maxsize=None)
def load_topics() -> list[str]:
    """Returns the list of available topics from topics.json."""
    path = _CONFIG_DIR / "topics.json"
    topics = json.loads(path.read_text(encoding="utf-8"))
    logger.debug({"config_loader": "topics", "count": len(topics)})
    return topics


@lru_cache(maxsize=None)
def load_sources() -> list[str]:
    """Returns the list of RSS/Atom feed URLs from sources.json."""
    path = _CONFIG_DIR / "sources.json"
    sources = json.loads(path.read_text(encoding="utf-8"))
    logger.debug({"config_loader": "sources", "count": len(sources)})
    return sources


@lru_cache(maxsize=None)
def load_profile() -> EngineerProfile:
    """Returns the engineer profile from profile.json, validated by Pydantic."""
    path = _CONFIG_DIR / "profile.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    profile = EngineerProfile(**data)
    logger.debug({"config_loader": "profile", "name": profile.name})
    return profile
