# tests/tier4/conftest.py
#
# Shared fixtures and utilities for Tier 4 LLM-as-judge tests.
#
# LLM availability
# ─────────────────
# Tier 4 tests require valid AWS credentials with Bedrock access.
# The llm_available() check calls STS to validate credentials without
# triggering any model inference cost. Tests are skipped when it returns False.
#
# Judge model
# ────────────
# The judge calls Claude Haiku via Bedrock directly (not through CrewAI) so
# it stays independent from the pipeline under test. It returns a structured
# JSON verdict — no free-text parsing.

import json
import pytest
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from contracts.primitives import TrendStrength
from contracts.nodes import (
    EngineerProfile,
    FetchTaskResult,
    OrchestratorContext,
    ScoringTaskInput,
    SynthesisTaskInput,
    TopicTaskInput,
    TrendSnapshot,
)
from contracts.primitives import ArticleRaw, ArticleScored

# ---------------------------------------------------------------------------
# LLM availability guard
# ---------------------------------------------------------------------------

def llm_available() -> bool:
    """Returns True if AWS credentials with Bedrock access are configured."""
    try:
        boto3.client("sts", region_name="us-east-1").get_caller_identity()
        return True
    except (NoCredentialsError, ClientError, Exception):
        return False


requires_llm = pytest.mark.skipif(
    not llm_available(),
    reason="AWS credentials not configured — skipping Tier 4 LLM tests",
)


# ---------------------------------------------------------------------------
# Judge helper
# ---------------------------------------------------------------------------

JUDGE_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def judge_output(output: dict | str, criteria: list[dict]) -> dict:
    """
    Calls Claude Haiku via Bedrock to evaluate an output against named criteria.

    Each criterion must have:
      - "name": str  (used as the key in criteria_results)
      - "description": str  (what constitutes a pass)

    Returns:
      {
        "passed": bool,               # True only if all criteria pass
        "criteria_results": {name: bool, ...},
        "rationale": str              # one sentence summary
      }

    Raises on Bedrock errors — tests should catch and fail explicitly.
    """
    if isinstance(output, dict):
        output_text = json.dumps(output, indent=2)
    else:
        output_text = str(output)

    criteria_text = "\n".join(
        f'- "{c["name"]}": {c["description"]}'
        for c in criteria
    )

    prompt = f"""You are a quality evaluator for an AI-powered digest pipeline.
Evaluate the output below against each named criterion and return a JSON verdict.

OUTPUT TO EVALUATE:
{output_text}

CRITERIA (evaluate each as pass/fail):
{criteria_text}

Return ONLY a JSON object with exactly these fields:
{{
  "passed": <true if ALL criteria pass, false otherwise>,
  "criteria_results": {{{", ".join(f'"{c["name"]}": <true|false>' for c in criteria)}}},
  "rationale": "<one sentence summarizing why the output passed or failed overall>"
}}"""

    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    response = client.invoke_model(
        modelId=JUDGE_MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}],
        }),
        contentType="application/json",
        accept="application/json",
    )
    body = json.loads(response["body"].read())
    text = body["content"][0]["text"].strip()

    # Strip markdown fences if the model wrapped its response
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


# ---------------------------------------------------------------------------
# Canonical engineer profile (shared across all tier4 fixtures)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sam_profile() -> EngineerProfile:
    return EngineerProfile(
        name="Sam",
        focus_areas=[
            "AI agentic architecture",
            "multi-agent system design",
            "agent observability and governance",
            "platform engineering",
            "AWS infrastructure",
            "LLM cost and latency optimization",
        ],
        background_summary=(
            "Senior software engineer transitioning into AI agentic architecture. "
            "Deep experience in distributed systems and AWS. Currently building "
            "production multi-agent pipelines and learning LangGraph/CrewAI patterns."
        ),
        experience_level="senior engineer specializing in AI agentic architecture",
    )


# ---------------------------------------------------------------------------
# Canonical active trends (shared across tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def active_trends() -> list[TrendSnapshot]:
    return [
        TrendSnapshot(
            trend_id="supervisor-patterns",
            name="Supervisor Patterns in Multi-Agent Systems",
            strength=0.75,
            strength_band=TrendStrength.STRONG,
            platform_relevance=0.9,
            key_signals=["supervisor node", "conditional routing", "rework loop"],
            last_reinforced="2026-05-20",
        ),
        TrendSnapshot(
            trend_id="structured-outputs",
            name="Structured Outputs and Contract-Driven Agents",
            strength=0.55,
            strength_band=TrendStrength.GROWING,
            platform_relevance=0.85,
            key_signals=["Pydantic", "output validation", "JSON schema enforcement"],
            last_reinforced="2026-05-18",
        ),
    ]


# ---------------------------------------------------------------------------
# Canonical passed articles for synthesis tests
# These are realistic but fixed — any LLM asked to synthesize them should
# be able to produce a coherent digest about multi-agent orchestration.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def canonical_passed_articles() -> list[ArticleScored]:
    def _scored(article_id, url, title, source_domain, summary, rel, rep, combined, passed=True) -> ArticleScored:
        return ArticleScored(
            article_id=article_id,
            url=url,
            title=title,
            source_domain=source_domain,
            published_at="2026-05-01T00:00:00",
            summary=summary,
            relevance_score=rel,
            reputation_score=rep,
            combined_score=combined,
            passed_threshold=passed,
            score_rationale=f"Relevance {rel:.2f} × 0.65 + Reputation {rep:.2f} × 0.35 = {combined:.3f} (pass).",
        )

    return [
        _scored(
            article_id="langchain-supervisor-2026",
            url="https://blog.langchain.dev/supervisor-patterns",
            title="Supervisor Patterns in LangGraph: Coordinating Multi-Agent Workflows",
            source_domain="blog.langchain.dev",
            summary=(
                "This post explores the supervisor pattern in LangGraph where a central node "
                "routes tasks to specialized subagents. Covers conditional edge design, rework "
                "loops, and how to prevent infinite retry cycles using rework count guards."
            ),
            rel=0.95, rep=0.85, combined=0.915,
        ),
        _scored(
            article_id="crewai-process-types-2026",
            url="https://docs.crewai.com/process-types",
            title="CrewAI Process Types: Sequential vs Hierarchical Orchestration",
            source_domain="docs.crewai.com",
            summary=(
                "An overview of how CrewAI supports sequential and hierarchical process types "
                "for coordinating agent execution. Hierarchical mode introduces a manager LLM "
                "that delegates to worker agents and validates their outputs before proceeding."
            ),
            rel=0.88, rep=0.80, combined=0.852,
        ),
        _scored(
            article_id="lilian-agent-memory-2026",
            url="https://lilianweng.github.io/posts/agent-memory",
            title="Memory Mechanisms in Agentic Systems: Short-Term, Long-Term, and Episodic",
            source_domain="lilianweng.github.io",
            summary=(
                "Lilian Weng surveys memory architectures used in LLM agents. Covers in-context "
                "window memory, external vector stores, and episodic memory patterns. Includes "
                "practical implications for multi-agent coordination where agents share state."
            ),
            rel=0.78, rep=0.92, combined=0.829,
        ),
        _scored(
            article_id="anthropic-tool-use-2026",
            url="https://www.anthropic.com/tool-use-patterns",
            title="Tool Use Patterns for Production LLM Agents",
            source_domain="www.anthropic.com",
            summary=(
                "Best practices for designing tool schemas and handling tool call responses "
                "in production agents. Covers error propagation, retry semantics, and how "
                "structured tool outputs reduce downstream parsing failures."
            ),
            rel=0.72, rep=0.90, combined=0.783,
        ),
    ]


# ---------------------------------------------------------------------------
# Canonical raw articles for scoring quality tests
# Grouped into clearly_relevant, clearly_irrelevant, and borderline.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def articles_for_scoring() -> dict[str, list[ArticleRaw]]:
    def _raw(article_id, title, source_domain, summary) -> ArticleRaw:
        return ArticleRaw(
            article_id=article_id,
            url=f"https://{source_domain}/{article_id}",
            title=title,
            source_domain=source_domain,
            published_at="2026-05-01",
            summary=summary,
        )

    return {
        "clearly_relevant": [
            _raw(
                "rel-1", "LangGraph Supervisor Nodes: Routing Agents with Conditional Edges",
                "blog.langchain.dev",
                "Deep dive into building supervisor nodes in LangGraph that route between "
                "specialist agents using conditional edges and rework loops.",
            ),
            _raw(
                "rel-2", "Multi-Agent Coordination: When to Use a Manager vs Peer Model",
                "lilianweng.github.io",
                "Analysis of orchestration topologies for multi-agent systems including "
                "hierarchical manager patterns versus peer-to-peer agent communication.",
            ),
        ],
        "clearly_irrelevant": [
            _raw(
                "irr-1", "2026 FIFA World Cup Group Stage Results",
                "bbc.com",
                "Coverage of the 2026 FIFA World Cup group stage matches, scores, and standings.",
            ),
            _raw(
                "irr-2", "Central Bank Raises Interest Rates for Third Consecutive Quarter",
                "reuters.com",
                "The central bank announced a 0.25 point rate increase citing persistent inflation.",
            ),
        ],
        "borderline": [
            _raw(
                "border-1", "The State of AI Coding Assistants in 2026",
                "techcrunch.com",
                "Overview of AI coding tools including GitHub Copilot and Cursor. Mentions "
                "that some tools now use multi-step agent pipelines to resolve complex tasks.",
            ),
        ],
    }
