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
  combined = (relevance x RELEVANCE_WEIGHT) + (reputation x REPUTATION_WEIGHT) + (recency x RECENCY_WEIGHT)

  RELEVANCE_WEIGHT  = 0.55  — content quality is the primary signal
  REPUTATION_WEIGHT = 0.25  — source credibility is a secondary modifier
  RECENCY_WEIGHT    = 0.20  — publication age biases toward recent articles

  Recency score buckets: 0-7d=1.0, 8-30d=0.75, 31-90d=0.5, 91-180d=0.25, >180d=0.1, no date=0.5

  A highly relevant fresh article (rel=0.9, rep=0.5, recency=1.0):
    (0.9 x 0.55) + (0.5 x 0.25) + (1.0 x 0.20) = 0.495 + 0.125 + 0.200 = 0.820  → passes

  A highly relevant old article (rel=0.9, rep=0.5, recency=0.1):
    (0.9 x 0.55) + (0.5 x 0.25) + (0.1 x 0.20) = 0.495 + 0.125 + 0.020 = 0.640  → still passes

  A marginal old article (rel=0.55, rep=0.5, recency=0.1):
    (0.55 x 0.55) + (0.5 x 0.25) + (0.1 x 0.20) = 0.303 + 0.125 + 0.020 = 0.448  → filtered at 0.5

Rework behavior
───────────────
  BELOW_SCORE_THRESHOLD  → lower score_threshold for this pass
  LOW_QUALITY_ARTICLES   → raise score_threshold, stricter filtering
"""

from __future__ import annotations
import json
import logging
import re
from datetime import datetime, timezone

from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM

from config import settings
from contracts.nodes import ScoringTaskInput, ScoringTaskResult
from contracts.primitives import ArticleRaw, ArticleScored, RetryReasonCode
from node_definitions.crew_utils import kickoff_crew

logger = logging.getLogger(__name__)

# Scoring weights and defaults — configurable via env vars in config.py.
DEFAULT_REPUTATION = settings.scoring_default_reputation
RELEVANCE_WEIGHT   = settings.scoring_relevance_weight
REPUTATION_WEIGHT  = settings.scoring_reputation_weight
RECENCY_WEIGHT     = settings.scoring_recency_weight

# Maximum articles sent to the LLM in a single relevance scoring call.
# At 20+ articles, a single prompt risks context pressure, article conflation,
# and a single malformed JSON entry corrupting the entire response. Batching
# keeps each call focused and degrades partially rather than all-or-nothing.
SCORING_BATCH_SIZE = 10


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
    Scores articles for relevance via the RelevanceAnalyst crew.

    Articles are chunked at SCORING_BATCH_SIZE before being sent to the LLM.
    This keeps each prompt focused and prevents a single malformed JSON entry
    in a large response from corrupting scores for every article in the run.
    Results from all batches are merged before returning.

    Returns a dict keyed by article_id:
      {"relevance_score": float, "relevance_rationale": str}
    Returns an empty dict on total failure — _combine_scores handles missing
    entries by applying the DEFAULT_REPUTATION fallback score.
    """
    articles = task_input.articles
    merged: dict[str, dict] = {}

    for i in range(0, len(articles), SCORING_BATCH_SIZE):
        batch = articles[i : i + SCORING_BATCH_SIZE]
        batch_num = (i // SCORING_BATCH_SIZE) + 1
        logger.info({
            "node":       "scoring",
            "batch":      batch_num,
            "batch_size": len(batch),
        })
        batch_result = _score_batch(batch, task_input)
        merged.update(batch_result)

    return merged


def _score_batch(batch: list, task_input: ScoringTaskInput) -> dict[str, dict]:
    """
    Runs the RelevanceAnalyst crew on a single batch of articles.
    Returns a partial relevance_scores dict for the articles in this batch.
    """
    llm = LLM(model=settings.bedrock_model_scoring)

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
        for a in batch
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

Rationale consistency rule:
Your rationale must match your score. A score of 0.7 or higher means the article
is strongly relevant — your rationale must say so clearly. Do not use hedging
language ("tangentially", "loosely related", "could be relevant") when assigning
a high score. If you find yourself hedging, lower the score to match the rationale.
  Correct for score 0.85: "Directly addresses supervisor node design in LangGraph,
    which is the core of the topic and intersects the active orchestration trend."
  Incorrect for score 0.85: "Touches on agents and mentions orchestration in passing."

Return a JSON array where each item has exactly these fields:
  "article_id": "<id>",
  "relevance_score": <float 0.0-1.0>,
  "relevance_rationale": "<one sentence explaining the score, consistent with the score level>"
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
        kickoff_crew(crew, "scoring", task_input.run_id, [settings.bedrock_model_scoring])
        return _parse_relevance_output(relevance_task.output.raw)
    except Exception as e:
        logger.warning({"node": "scoring", "warning": f"Relevance batch scoring failed: {e}"})
        return {}


def _parse_relevance_output(raw_output: str) -> dict[str, dict]:
    """
    Parses the RelevanceAnalyst's JSON output into a dict keyed by article_id.
    Returns an empty dict on parse failure — callers apply fallback scores.

    Fences are stripped first because LLMs frequently wrap JSON output in
    markdown code blocks. Stripping before the first json.loads attempt means
    a fenced-only response is handled cleanly without falling through to the
    regex fallback, which can match partial content.
    """
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_output.strip(), flags=re.IGNORECASE)
    try:
        items = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
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

def _compute_recency_score(published_at: str, now: datetime | None = None) -> float:
    """
    Returns a recency score in [0.1, 1.0] based on article age.
    Articles with no parseable date score neutrally (0.5).

    Buckets: 0-7d=1.0, 8-30d=0.75, 31-90d=0.5, 91-180d=0.25, >180d=0.1
    """
    if not published_at:
        return 0.5
    try:
        pub_dt = datetime.fromisoformat(published_at)
        if pub_dt.tzinfo is None:
            pub_dt = pub_dt.replace(tzinfo=timezone.utc)
        ref = now or datetime.now(timezone.utc)
        age_days = (ref - pub_dt).days
    except (ValueError, TypeError):
        return 0.5

    if age_days <= 7:
        return 1.0
    elif age_days <= 30:
        return 0.75
    elif age_days <= 90:
        return 0.5
    elif age_days <= 180:
        return 0.25
    else:
        return 0.1


def _combine_scores(
    articles:         list[ArticleRaw],
    relevance_scores: dict[str, dict],
    reputation_map:   dict[str, float],
    score_threshold:  float,
    _now:             datetime | None = None,
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
        recency_score    = _compute_recency_score(article.published_at, _now)

        combined_score = round(
            (relevance_score * RELEVANCE_WEIGHT)
            + (reputation_score * REPUTATION_WEIGHT)
            + (recency_score * RECENCY_WEIGHT),
            4,
        )
        passed = combined_score >= score_threshold

        # Rationale constructed from actual numbers — honest and auditable
        score_rationale = (
            f"Relevance {relevance_score:.2f} × {RELEVANCE_WEIGHT} + "
            f"Reputation {reputation_score:.2f} × {REPUTATION_WEIGHT} + "
            f"Recency {recency_score:.2f} × {RECENCY_WEIGHT} "
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
            recency_score    = recency_score,
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