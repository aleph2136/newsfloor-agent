"""
nodes/scoring.py

Scores each article for relevance and source reputation.

Crew design
───────────
One agent with one responsibility:

  RelevanceAnalyst    Scores each article against the current topic, focus
                      angle, and active trend names. Produces a relevance_score
                      (0.0-1.0) and a one-sentence rationale per article.
                      This is genuine LLM work — evaluating content relevance
                      requires reading comprehension, not just lookup.

Reputation weighting — why no LLM
──────────────────────────────────
Reputation scoring is a deterministic operation: look up the domain score
in the reputation map, apply a weighted formula, compare to threshold.
There is no reasoning involved and no benefit to routing it through a
language model. The previous design used a ReputationAdjuster agent to
do this — it was an expensive calculator. That logic now lives in
_combine_scores() where it belongs.

The reputation map itself is built up over time by the trend node using
a rolling average of article relevance scores per domain. The scoring
node consumes it read-only.

Combined score formula
──────────────────────
  combined = (relevance × RELEVANCE_WEIGHT) + (reputation × REPUTATION_WEIGHT)

  RELEVANCE_WEIGHT  = 0.65  — content quality is the primary signal
  REPUTATION_WEIGHT = 0.35  — source credibility is a secondary modifier

  A highly relevant article from an unknown source (rep=0.5) scores:
    (0.9 × 0.65) + (0.5 × 0.35) = 0.585 + 0.175 = 0.760  → passes at 0.5 threshold

  A low-relevance article from a trusted source (rep=0.9) scores:
    (0.3 × 0.65) + (0.9 × 0.35) = 0.195 + 0.315 = 0.510  → marginal pass

Rework behavior
───────────────
  BELOW_SCORE_THRESHOLD  → lower score_threshold for this pass
  LOW_QUALITY_ARTICLES   → raise score_threshold, stricter filtering
"""

from __future__ import annotations
import json
import logging
import re

from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM

from config import settings
from contracts.nodes import ScoringTaskInput, ScoringTaskResult
from contracts.primitives import ArticleRaw, ArticleScored, RetryReasonCode

logger = logging.getLogger(__name__)

# Default reputation for domains not yet in the reputation map.
# Neutral start — no evidence for or against.
DEFAULT_REPUTATION = 0.5

RELEVANCE_WEIGHT  = 0.65
REPUTATION_WEIGHT = 0.35


def run(task_input: ScoringTaskInput) -> ScoringTaskResult:
    """
    Scores articles for relevance via LLM, then applies reputation
    weighting and threshold in code. Returns a ScoringTaskResult.
    """
    logger.info({
        "node":            "scoring",
        "topic":           task_input.topic,
        "article_count":   len(task_input.articles),
        "score_threshold": task_input.score_threshold,
        "has_retry":       task_input.retry_instruction is not None,
    })

    score_threshold = _apply_retry_adjustments(task_input)

    # --- Step 1: LLM relevance scoring ---
    relevance_scores = _score_relevance(task_input)

    # --- Step 2: Deterministic reputation weighting and threshold ---
    scored_articles = _combine_scores(
        articles          = task_input.articles,
        relevance_scores  = relevance_scores,
        reputation_map    = task_input.source_reputation_map,
        score_threshold   = score_threshold,
    )

    passed   = [a for a in scored_articles if a.passed_threshold]
    filtered = [a for a in scored_articles if not a.passed_threshold]

    logger.info({
        "node":             "scoring",
        "total_scored":     len(scored_articles),
        "passed_threshold": len(passed),
        "filtered_out":     len(filtered),
        "score_threshold":  score_threshold,
    })

    return ScoringTaskResult(
        run_id             = task_input.run_id,
        scored_articles    = scored_articles,
        passed_articles    = passed,
        filtered_articles  = filtered,
        high_quality_count = len(passed),
        low_quality_count  = len(filtered),
    )


# ---------------------------------------------------------------------------
# Step 1 — LLM relevance scoring
# ---------------------------------------------------------------------------

def _score_relevance(task_input: ScoringTaskInput) -> dict[str, dict]:
    """
    Runs the RelevanceAnalyst crew and returns a dict keyed by article_id.
    Each value is {"relevance_score": float, "relevance_rationale": str}.
    Returns an empty dict on failure — _combine_scores handles missing entries
    by applying the default reputation score.
    """
    llm = LLM(model=settings.bedrock_model_haiku)

    relevance_analyst = Agent(
        role="Relevance Analyst",
        goal=(
            "Score each article on how relevant it is to today's topic and focus angle, "
            "considering active trends in the field. Be precise and consistent — "
            "a score of 0.8 means genuinely strong relevance, not loose association."
        ),
        backstory=(
            "You are an expert in AI agentic architecture and engineering with a sharp "
            "eye for distinguishing genuinely relevant technical content from tangentially "
            "related noise. You evaluate articles on substance, not surface keywords."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    article_list = "\n".join(
        f"ID: {a.article_id}\nTitle: {a.title}\nSource: {a.source_domain}\n"
        f"Summary: {a.summary}\n"
        for a in task_input.articles
    )

    relevance_task = Task(
        description=f"""
Score each article below for relevance to today's topic and focus angle.

TOPIC:       {task_input.topic}
FOCUS ANGLE: {task_input.focus_angle}

ACTIVE TRENDS (articles touching these should score higher):
{chr(10).join(f"- {t}" for t in task_input.active_trend_names) or "None yet"}

ARTICLES TO SCORE:
{article_list}

Scoring guide:
  0.0 - 0.3  Irrelevant or only superficially related
  0.4 - 0.6  Moderately relevant, tangential connection
  0.7 - 0.8  Clearly relevant, directly addresses topic or focus angle
  0.9 - 1.0  Highly relevant, addresses topic AND intersects active trends

Return a JSON array where each item has exactly these fields:
  "article_id": "<id>",
  "relevance_score": <float 0.0-1.0>,
  "relevance_rationale": "<one sentence explaining the score>"
        """,
        expected_output=(
            "A JSON array of objects with article_id, relevance_score, "
            "and relevance_rationale fields."
        ),
        agent=relevance_analyst,
        output_json=True,
    )

    crew = Crew(
        agents  = [relevance_analyst],
        tasks   = [relevance_task],
        process = Process.sequential,
        verbose = False,
    )

    try:
        crew.kickoff()
        return _parse_relevance_output(relevance_task.output.raw)
    except Exception as e:
        logger.warning({"node": "scoring", "warning": f"Relevance scoring failed: {e}"})
        return {}


def _parse_relevance_output(raw_output: str) -> dict[str, dict]:
    """
    Parses the RelevanceAnalyst's JSON output into a dict keyed by article_id.
    Returns an empty dict on parse failure — callers apply fallback scores.
    """
    try:
        items = json.loads(raw_output)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw_output, re.DOTALL)
        if not match:
            return {}
        try:
            items = json.loads(match.group())
        except json.JSONDecodeError:
            return {}

    return {
        item["article_id"]: {
            "relevance_score":    float(item.get("relevance_score", DEFAULT_REPUTATION)),
            "relevance_rationale": item.get("relevance_rationale", ""),
        }
        for item in items
        if "article_id" in item
    }


# ---------------------------------------------------------------------------
# Step 2 — Deterministic combination
# ---------------------------------------------------------------------------

def _combine_scores(
    articles:         list[ArticleRaw],
    relevance_scores: dict[str, dict],
    reputation_map:   dict[str, float],
    score_threshold:  float,
) -> list[ArticleScored]:
    """
    Combines LLM relevance scores with reputation map values using the
    weighted formula. All arithmetic — no LLM involved.

    For any article missing from relevance_scores (LLM parse failure),
    applies DEFAULT_REPUTATION as the relevance score so the article
    is not silently dropped.
    """
    scored = []

    for article in articles:
        relevance_data   = relevance_scores.get(article.article_id, {})
        relevance_score  = float(relevance_data.get("relevance_score", DEFAULT_REPUTATION))
        relevance_rationale = relevance_data.get("relevance_rationale", "Fallback — relevance not scored.")

        reputation_score = reputation_map.get(article.source_domain, DEFAULT_REPUTATION)

        combined_score = round(
            (relevance_score * RELEVANCE_WEIGHT) + (reputation_score * REPUTATION_WEIGHT),
            4,
        )
        passed = combined_score >= score_threshold

        # Rationale constructed from actual numbers — honest and auditable
        score_rationale = (
            f"Relevance {relevance_score:.2f} × {RELEVANCE_WEIGHT} + "
            f"Reputation {reputation_score:.2f} × {REPUTATION_WEIGHT} "
            f"= {combined_score:.2f} ({'pass' if passed else 'fail'} at {score_threshold}). "
            f"{relevance_rationale}"
        )

        scored.append(ArticleScored(
            article_id       = article.article_id,
            url              = article.url,
            title            = article.title,
            source_domain    = article.source_domain,
            published_at     = article.published_at,
            summary          = article.summary,
            relevance_score  = relevance_score,
            reputation_score = reputation_score,
            combined_score   = combined_score,
            passed_threshold = passed,
            score_rationale  = score_rationale,
        ))

    return scored


# ---------------------------------------------------------------------------
# Retry adjustments
# ---------------------------------------------------------------------------

def _apply_retry_adjustments(task_input: ScoringTaskInput) -> float:
    """
    Reads the retry_instruction and returns an adjusted score threshold.
    """
    threshold   = task_input.score_threshold
    instruction = task_input.retry_instruction

    if instruction is None:
        return threshold

    reason = instruction.reason_code
    params = instruction.parameter_adjustment

    if reason == RetryReasonCode.BELOW_SCORE_THRESHOLD:
        threshold = params.get("score_threshold", max(0.3, threshold - 0.1))

    elif reason == RetryReasonCode.LOW_QUALITY_ARTICLES:
        threshold = params.get("score_threshold", min(0.8, threshold + 0.1))

    return threshold