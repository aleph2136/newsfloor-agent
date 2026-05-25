"""
nodes/trend/signal_analysis.py

LLM reasoning for signal clustering, trend classification,
and reputation anomaly analysis.

All three functions fire conditionally — only when there is something
to reason about. On a run with no new signals and no reputation anomalies
this entire file costs zero tokens.

Public interface
────────────────
  cluster_signals()     Groups semantically equivalent signals
  classify_new_trends() Names and scores genuinely new clusters
"""

from __future__ import annotations
import json
import logging
import re
from datetime import datetime, timezone

from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM

from contracts.primitives import TrendStrength
from contracts.state import TrendRecord, current_week_id
from node_definitions.crew_utils import kickoff_crew

logger = logging.getLogger(__name__)


def cluster_signals(
    llm:             LLM,
    new_signals:     list[str],
    existing_trends: list[TrendRecord],
    topic:           str,
) -> list[dict]:
    """
    Groups semantically equivalent signals and identifies which are
    genuinely new versus matching an existing trend.

    Returns a list of cluster dicts:
      {
        "signals":          list[str],  signals in this cluster
        "is_new":           bool,       True = create a new trend
        "matches_existing": str | None, existing trend name if not new
        "representative":   str,        best signal phrase for the cluster
      }

    Falls back to one-signal-per-cluster if the LLM call fails.
    """
    existing_summary = "\n".join(
        f"- {t.name}: {', '.join(t.key_signals[:3])}"
        for t in existing_trends if not t.archived
    ) or "None yet"

    signals_text = "\n".join(f"- {s}" for s in new_signals)

    analyst = Agent(
        role="Signal Analyst",
        goal=(
            "Cluster semantically related signals together and identify which "
            "clusters represent genuinely new trends versus variations of existing ones."
        ),
        backstory=(
            "You are a precise technical analyst specializing in AI agentic systems. "
            "You have a strong sense for when two differently-worded concepts are "
            "actually the same thing versus genuinely distinct ideas. You never "
            "conflate concepts just because they share keywords."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    task = Task(
        description=f"""
Analyze new signals extracted from today's AI agentic engineering digest.
Cluster semantically equivalent signals and determine which are genuinely new.

TODAY'S TOPIC: {topic}

NEW SIGNALS:
{signals_text}

EXISTING TRENDS (name: key signals):
{existing_summary}

Instructions:
1. Group signals that express the same underlying concept into one cluster.
   Example: "structured agent outputs" and "typed output contracts for agents"
   are the same concept — one cluster, not two trends.

2. For each cluster, determine if it semantically matches an existing trend.
   Matching means the concept is already captured — not just shares a keyword.

3. Identify the most representative signal phrase for each cluster.

Return a JSON array — one object per cluster:
[{{
  "signals":          ["<signal>", ...],
  "is_new":           true or false,
  "matches_existing": "<existing trend name>" or null,
  "representative":   "<most precise signal phrase>"
}}]
        """,
        expected_output="A JSON array of cluster objects.",
        agent=analyst,
        output_json=True,
    )

    crew = Crew(agents=[analyst], tasks=[task], process=Process.sequential, verbose=False)

    try:
        kickoff_crew(crew, "trend.cluster_signals", None, [llm.model])
        result = _parse_json_list(task.output.raw)
        if result:
            return result
    except Exception as e:
        logger.warning({"function": "cluster_signals", "error": str(e)})

    # Fallback — one cluster per signal, all treated as new
    return [
        {"signals": [s], "is_new": True, "matches_existing": None, "representative": s}
        for s in new_signals
    ]


def classify_new_trends(
    llm:          LLM,
    clusters:     list[dict],
    topic:        str,
    existing_ids: set[str],
    run_id:       str,
    now:          str,
) -> list[TrendRecord]:
    """
    Classifies genuinely new signal clusters into properly named TrendRecords.

    The LLM produces canonical names, platform relevance scores, and related
    topic lists. Keyword heuristics are not reliable enough for permanent records.

    Returns a list of TrendRecord objects ready to write to DynamoDB.
    Falls back to keyword-estimated records if the LLM call fails.
    """
    clusters_text = "\n".join(
        f"- Representative: {c.get('representative', c['signals'][0])}\n"
        f"  All signals: {', '.join(c['signals'])}"
        for c in clusters
    )

    classifier = Agent(
        role="Trend Classifier",
        goal=(
            "Produce precise, canonical names and classifications for new AI agentic "
            "engineering trends. Names must be specific, stable, and meaningfully "
            "distinct from each other."
        ),
        backstory=(
            "You are a senior AI systems architect who curates a long-term knowledge "
            "base of emerging trends in agentic engineering. You name things carefully — "
            "a trend name is a permanent record matched against content for months. "
            "Vague names like 'agent improvements' are unacceptable. You always connect "
            "trends to their engineering implications: governance, observability, "
            "reliability, human oversight, or practical value creation."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    task = Task(
        description=f"""
Classify each new signal cluster into a properly named trend record.

TODAY'S TOPIC: {topic}

FOCUS AREAS (for platform_relevance scoring):
  AI agentic architecture, engineering governance, observability,
  reliability, human-in-the-loop design, tools that create real value

NEW SIGNAL CLUSTERS:
{clusters_text}

For each cluster produce:
  trend_id:           URL-safe slug, max 60 chars, lowercase with hyphens
  name:               Canonical human-readable name, title case, 3-7 words
  platform_relevance: Float 0.0-1.0
                        0.9+ = directly about agentic architecture or governance
                        0.7-0.8 = strongly related infrastructure or tooling
                        0.5-0.6 = adjacent topics with indirect relevance
                        below 0.5 = tangential
  related_topics:     1-3 broader topic areas this trend belongs to
  key_signals:        Signal phrases from the cluster, cleaned and normalized

Return a JSON array:
[{{
  "trend_id":           "<slug>",
  "name":               "<Canonical Name>",
  "platform_relevance": <float>,
  "related_topics":     ["<topic>", ...],
  "key_signals":        ["<signal>", ...]
}}]
        """,
        expected_output="A JSON array of trend classification objects.",
        agent=classifier,
        output_json=True,
    )

    crew = Crew(agents=[classifier], tasks=[task], process=Process.sequential, verbose=False)

    try:
        kickoff_crew(crew, "trend.classify_new_trends", run_id, [llm.model])
        items = _parse_json_list(task.output.raw)
    except Exception as e:
        logger.warning({"function": "classify_new_trends", "error": str(e)})
        items = []

    if not items:
        # Fallback — keyword-estimated records
        return [_fallback_trend(c, topic, existing_ids, run_id, now) for c in clusters]

    records = []
    local_ids = set(existing_ids)

    for item in items:
        trend_id = item.get("trend_id", _slugify(item.get("name", "unknown")))
        if trend_id in local_ids:
            trend_id = f"{trend_id}-{run_id[-5:]}"
        local_ids.add(trend_id)

        records.append(TrendRecord(
            trend_id           = trend_id,
            name               = item.get("name", trend_id.replace("-", " ").title()),
            first_observed     = now,
            last_reinforced    = now,
            strength           = 0.25,
            strength_band      = TrendStrength.EMERGING,
            platform_relevance = float(item.get("platform_relevance", 0.5)),
            related_topics     = item.get("related_topics", [topic]),
            key_signals        = item.get("key_signals", []),
            evidence_weeks     = [current_week_id()],
            times_reinforced   = 1,
            archived           = False,
            created_at         = now,
            updated_at         = now,
        ))

    return records



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fallback_trend(
    cluster:      dict,
    topic:        str,
    existing_ids: set[str],
    run_id:       str,
    now:          str,
) -> TrendRecord:
    """
    Keyword-estimated TrendRecord for when LLM classification fails.
    Better than nothing — gives the trend a reasonable starting point.
    """
    signal   = cluster.get("representative", cluster["signals"][0])
    trend_id = _slugify(signal)
    if trend_id in existing_ids:
        trend_id = f"{trend_id}-{run_id[-5:]}"

    focus_keywords = {
        "agent", "agentic", "orchestrat", "govern", "observab",
        "reliab", "supervisor", "pipeline", "tool", "memory",
        "platform", "architect", "structur", "evaluat", "human-in",
    }
    signal_lower      = signal.lower()
    overlap           = sum(1 for kw in focus_keywords if kw in signal_lower)
    platform_relevance = min(1.0, 0.4 + (overlap * 0.15))

    return TrendRecord(
        trend_id           = trend_id,
        name               = signal.title(),
        first_observed     = now,
        last_reinforced    = now,
        strength           = 0.25,
        strength_band      = TrendStrength.EMERGING,
        platform_relevance = platform_relevance,
        related_topics     = [topic],
        key_signals        = cluster["signals"],
        evidence_weeks     = [current_week_id()],
        times_reinforced   = 1,
        archived           = False,
        created_at         = now,
        updated_at         = now,
    )


def _parse_json_list(raw: str) -> list:
    """Parses a JSON array from raw LLM output. Returns empty list on failure."""
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass
    return []


def _slugify(text: str) -> str:
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug[:60]