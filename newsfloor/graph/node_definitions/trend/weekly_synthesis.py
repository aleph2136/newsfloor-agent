"""
nodes/trend/weekly_synthesis.py

Weekly synthesis record assembly and LLM narrative generation.

Called only on Mondays. Reads the last 7 RunRecords, computes signal
patterns deterministically, then uses an LLM agent to produce a
human-readable narrative describing what the week's pattern means.

The narrative is currently logged rather than stored — the WeeklySynthesis
schema can be extended with a narrative field when you want to surface
it in the digest itself.
"""

from __future__ import annotations
import logging
from collections import Counter

from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM

from contracts.state import WeeklySynthesis, current_week_id, ttl_days
from data.db import DynamoDBService

logger = logging.getLogger(__name__)


def write_weekly_synthesis(
    db:  DynamoDBService,
    llm: LLM,
    now: str,
) -> None:
    """
    Assembles and writes a WeeklySynthesis record for the current week.

    Deterministic code collects and computes signal statistics.
    LLM agent produces the pattern narrative.
    """
    recent_runs     = db.get_recent_runs(days=7)
    prior_week_runs = db.get_recent_runs(days=14)

    all_signals: list[str] = []
    topics:      list[str] = []
    run_ids:     list[str] = []

    for run in recent_runs:
        if run.topic:
            topics.append(run.topic)
        all_signals.extend(run.new_signals)
        run_ids.append(run.run_id)

    # Signals appearing 3+ times this week — genuinely recurring
    signal_counts = Counter(all_signals)
    recurring     = [s for s, count in signal_counts.items() if count >= 3]

    # New this week vs prior week
    prior_signals   = set()
    for run in prior_week_runs:
        if run.run_id not in run_ids:
            prior_signals.update(run.new_signals)

    current_signals = set(all_signals)
    emerging        = list(current_signals - prior_signals)[:20]
    fading          = list(prior_signals - current_signals)[:20]

    # LLM narrative
    narrative = _generate_narrative(
        llm       = llm,
        topics    = topics,
        recurring = recurring,
        emerging  = emerging,
        fading    = fading,
    )

    # Log the narrative — extend WeeklySynthesis with a narrative field
    # when you want to store and surface it in future digests
    logger.info({
        "action":           "weekly_narrative",
        "week_id":          current_week_id(),
        "narrative":        narrative,
        "recurring_count":  len(recurring),
        "emerging_count":   len(emerging),
        "fading_count":     len(fading),
    })

    synthesis = WeeklySynthesis(
        week_id                  = current_week_id(),
        topics_covered           = topics,
        recurring_signals        = recurring,
        emerging_concepts        = emerging,
        fading_concepts          = fading,
        source_reputation_deltas = {},
        run_ids_included         = run_ids,
        created_at               = now,
        ttl                      = ttl_days(90),
    )

    db.put_weekly_synthesis(synthesis)


def _generate_narrative(
    llm:       LLM,
    topics:    list[str],
    recurring: list[str],
    emerging:  list[str],
    fading:    list[str],
) -> str:
    """
    Produces a 3-5 sentence pattern narrative for the week.
    Returns an empty string on failure — narrative is non-critical.
    """
    analyst = Agent(
        role="Weekly Pattern Analyst",
        goal=(
            "Interpret a week of AI agentic engineering signals and produce a "
            "concise narrative describing what the pattern suggests about where "
            "the field is moving. Be specific — name the patterns, not just topics."
        ),
        backstory=(
            "You write a weekly synthesis read by practitioners building production "
            "agentic systems. You see patterns others miss and express them precisely. "
            "You connect signal patterns to their engineering implications — governance, "
            "observability, reliability, human oversight, and practical value creation."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    task = Task(
        description=f"""
Produce a weekly pattern narrative for an AI agentic engineering digest.

TOPICS COVERED THIS WEEK:
{chr(10).join(f"- {t}" for t in topics) or "None"}

RECURRING SIGNALS (appeared 3+ times this week):
{chr(10).join(f"- {s}" for s in recurring) or "None"}

EMERGING CONCEPTS (new this week, not seen last week):
{chr(10).join(f"- {s}" for s in emerging[:10]) or "None"}

FADING CONCEPTS (seen last week, absent this week):
{chr(10).join(f"- {s}" for s in fading[:10]) or "None"}

Write a 3-5 sentence narrative that:
- Names the dominant theme or pattern of the week
- Notes what is accelerating and what is losing momentum
- Connects the pattern to what it means for practitioners building
  agentic systems — governance, observability, reliability, or tooling

Return only the narrative — no headings, no bullet points.
        """,
        expected_output="A 3-5 sentence plain text narrative paragraph.",
        agent=analyst,
    )

    crew = Crew(agents=[analyst], tasks=[task], process=Process.sequential, verbose=False)

    try:
        crew.kickoff()
        return task.output.raw.strip()
    except Exception as e:
        logger.warning({"function": "_generate_narrative", "error": str(e)})
        return ""