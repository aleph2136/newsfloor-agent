from __future__ import annotations
import logging

from contracts.primitives import NodeName, RunStatus, SupervisorDecision, SupervisorRoute
from contracts.nodes import (
    TopicTaskInput,
    InputSupervisorInput,
    OutputSupervisorInput,
)
from state import DigestGraphState
from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# load_context
# Reads DynamoDB, builds OrchestratorContext, writes it into state.
# No LLM call — pure I/O. Runs once at the start of every run.
# ---------------------------------------------------------------------------
def load_context(state: DigestGraphState) -> dict:
    """
    Loads all historical state from DynamoDB and assembles OrchestratorContext.
    This is the only node that reads from DynamoDB at run start.
    The Trend node is the only node that writes to DynamoDB at run end.
    """
    run_id = state["run_id"]
    logger.info({"node": "load_context", "run_id": run_id})
    try:
        from data.load_context import run as load_context_run
        context = load_context_run()
        logger.info({
            "node":                 "load_context",
            "run_id":               run_id,
            "active_trends":        len(context.active_trends),
            "known_sources":        len(context.source_reputation_map),
            "recent_topics":        len(context.recent_topics),
            "recent_run_signals":   len(context.recent_run_signals),
            "recent_weekly_signals": len(context.recent_weekly_signals),
            "status":               "ok",
        })
        return {"context": context}
    except Exception as e:
        logger.error({
            "node":    "load_context",
            "run_id":  run_id,
            "error":   str(e),
            "status":  "degraded — continuing with empty context",
        })
        return {"context": None}


# ---------------------------------------------------------------------------
# topic_node
# Selects today's topic using the CrewAI topic crew.
# ---------------------------------------------------------------------------
 
def topic_node(state: DigestGraphState) -> dict:
    """
    Selects today's topic and focus angle.
    Reads active_retry_instruction if present — adjusts behavior on rework.
    Clears active_retry_instruction after reading it.
    """
    from node_definitions.topic import run as topic_run, AVAILABLE_TOPICS
    logger.info({
        "node":         "topic",
        "run_id":       state["run_id"],
        "rework_count": state["rework_counts"].get(NodeName.TOPIC.value, 0),
    })
 
    retry = state.get("active_retry_instruction")
    context = state["context"]

    task_input = TopicTaskInput(
        run_id              = state["run_id"],
        recent_topics       = context.recent_topics if context else [],
        active_trend_names  = [t.name for t in context.active_trends] if context else [],
        recent_signals      = context.recent_weekly_signals if context else [],
        available_topics    = AVAILABLE_TOPICS,
        retry_instruction   = retry,
    )

    result = topic_run(task_input)
 

    return {
        "topic_result":             result,   
        "active_retry_instruction": None,  
    }


# ---------------------------------------------------------------------------
# fetch_node
# Fetches raw articles from curated sources.
# ---------------------------------------------------------------------------
 
def fetch_node(state: DigestGraphState) -> dict:
    logger.info({"node": "fetch", "run_id": state["run_id"]})
 
    retry = state.get("active_retry_instruction")
    topic_result = state["topic_result"]

    from node_definitions.fetch import run as fetch_run, DEFAULT_SOURCES
    from contracts.nodes import FetchTaskInput
 
    task_input = FetchTaskInput(
        run_id             = state["run_id"],
        topic              = topic_result.topic,
        focus_angle        = topic_result.focus_angle,
        sources            = DEFAULT_SOURCES,
        min_articles       = settings.fetch_min_articles,
        max_articles       = settings.fetch_max_articles,
        retry_instruction  = retry,
    )

    result = fetch_run(task_input)
 
    return {
        "fetch_result":             result,
        "active_retry_instruction": None,
    }


# ---------------------------------------------------------------------------
# scoring_node
# Scores articles for relevance and source reputation.
# ---------------------------------------------------------------------------
 
def scoring_node(state: DigestGraphState) -> dict:
    logger.info({"node": "scoring", "run_id": state["run_id"]})
 
    retry = state.get("active_retry_instruction")
    context = state["context"]
    fetch_result = state["fetch_result"]
    topic_result = state["topic_result"]
 
    from node_definitions.scoring import run as scoring_run
    from contracts.nodes import ScoringTaskInput
 
    task_input = ScoringTaskInput(
        run_id                = state["run_id"],
        topic                 = topic_result.topic,
        focus_angle           = topic_result.focus_angle,
        articles              = fetch_result.articles,
        source_reputation_map = context.source_reputation_map if context else {},
        active_trend_names    = [t.name for t in context.active_trends] if context else [],
        score_threshold       = settings.score_threshold,
        retry_instruction     = retry,
    )
 
    result = scoring_run(task_input)
 
    return {
        "scoring_result":           result,
        "active_retry_instruction": None,
    }


# ---------------------------------------------------------------------------
# input_supervisor
# Evaluates topic + fetch + scoring as a unit.
# Returns SupervisorDecision — LangGraph routes on decision.route.
# Rework sends back to topic_node. Proceed advances to synthesis_node.
# ---------------------------------------------------------------------------
 
def input_supervisor(state: DigestGraphState) -> dict:
    """
    Quality gate over the full input stage.
 
    Evaluates whether the topic is well-chosen, whether enough articles
    were fetched, and whether enough passed scoring to support a meaningful
    digest. All three are evaluated together — a thin scoring result might
    mean the wrong topic was chosen, not just a bad fetch.
 
    Max reworks respected here — if rework_count >= max_reworks the
    supervisor is forced to proceed in degraded mode rather than loop.
    """
    rework_count = state["rework_counts"].get(NodeName.INPUT_SUPERVISOR.value, 0)
 
    logger.info({
        "node":         "input_supervisor",
        "run_id":       state["run_id"],
        "rework_count": rework_count,
    })
 
    supervisor_input = InputSupervisorInput(
        run_id         = state["run_id"],
        topic_result   = state["topic_result"],
        fetch_result   = state["fetch_result"],
        scoring_result = state["scoring_result"],
        rework_count   = rework_count,
    )
 
    from node_definitions.input_supervisor import run as input_supervisor_run
    decision = input_supervisor_run(supervisor_input)
 
    update: dict = {"input_supervisor_decision": decision}
 
    if decision.route == SupervisorRoute.REWORK:
        update["active_retry_instruction"] = decision.retry_instruction
        update["rework_counts"] = {NodeName.INPUT_SUPERVISOR.value: 1}
 
    return update

# ---------------------------------------------------------------------------
# synthesis_node
# Writes the digest and extracts trend signals.
# ---------------------------------------------------------------------------
 
def synthesis_node(state: DigestGraphState) -> dict:
    logger.info({
        "node":         "synthesis",
        "run_id":       state["run_id"],
        "rework_count": state["rework_counts"].get(NodeName.SYNTHESIS.value, 0),
    })
 
    retry = state.get("active_retry_instruction")
    context = state["context"]
    scoring_result = state["scoring_result"]
    topic_result = state["topic_result"]
 
    from node_definitions.synthesis import run as synthesis_run
    from contracts.nodes import SynthesisTaskInput
    from config_loader import load_profile

    task_input = SynthesisTaskInput(
        run_id             = state["run_id"],
        topic              = topic_result.topic,
        focus_angle        = topic_result.focus_angle,
        passed_articles    = scoring_result.passed_articles,
        active_trends      = context.active_trends if context else [],
        recent_run_signals = context.recent_run_signals if context else [],
        engineer_profile   = context.engineer_profile if context else load_profile(),
        retry_instruction  = retry,
    )
 
    result = synthesis_run(task_input)
 
    return {
        "synthesis_result":         result,
        "active_retry_instruction": None,
    }

# ---------------------------------------------------------------------------
# output_supervisor
# Evaluates the synthesis result before delivery.
# Rework sends back to synthesis_node only — articles are not re-fetched.
# ---------------------------------------------------------------------------
 
def output_supervisor(state: DigestGraphState) -> dict:
    """
    Quality gate over the digest output.
 
    Evaluates whether the digest is substantive, on-topic, and appropriately
    personalized to the engineer profile. On rework, only synthesis reruns —
    the fetched and scored articles are reused as-is.
    """
    rework_count = state["rework_counts"].get(NodeName.OUTPUT_SUPERVISOR.value, 0)
 
    logger.info({
        "node":         "output_supervisor",
        "run_id":       state["run_id"],
        "rework_count": rework_count,
    })
 
    topic_result = state.get("topic_result")
    context      = state.get("context")

    if not topic_result or not context:
        logger.warning({
            "node":    "output_supervisor",
            "run_id":  state["run_id"],
            "message": "Missing topic_result or context — forcing proceed in degraded mode",
        })
        return {
            "output_supervisor_decision": SupervisorDecision(
                supervisor   = NodeName.OUTPUT_SUPERVISOR,
                route        = SupervisorRoute.PROCEED,
                rework_count = rework_count,
                rationale    = "Missing upstream state — forced proceed in degraded mode",
            )
        }

    supervisor_input = OutputSupervisorInput(
        run_id           = state["run_id"],
        synthesis_result = state["synthesis_result"],
        topic            = topic_result.topic,
        focus_angle      = topic_result.focus_angle,
        engineer_profile = context.engineer_profile,
        rework_count     = rework_count,
    )
 
    from node_definitions.output_supervisor import run as output_supervisor_run
    decision = output_supervisor_run(supervisor_input)
 
    update: dict = {"output_supervisor_decision": decision}
 
    if decision.route == SupervisorRoute.REWORK:
        update["active_retry_instruction"] = decision.retry_instruction
        update["rework_counts"] = {NodeName.OUTPUT_SUPERVISOR.value: 1}
 
    return update

# ---------------------------------------------------------------------------
# delivery_node
# Sends the digest via Gmail SMTP. Retried by tenacity, not by LangGraph.
# ---------------------------------------------------------------------------

def delivery_node(state: DigestGraphState) -> dict:
    logger.info({"node": "delivery", "run_id": state["run_id"]})

    synthesis_result = state["synthesis_result"]
    topic_result     = state["topic_result"]

    from node_definitions.delivery import run as delivery_run
    from contracts.nodes import DeliveryTaskInput

    task_input = DeliveryTaskInput(
        run_id          = state["run_id"],
        digest_html     = synthesis_result.digest_html,
        topic           = topic_result.topic,
        recipient_email = settings.smtp_recipient_email,
        sender_email    = settings.smtp_sender_email,
    )
 
    result = delivery_run(task_input)

    return {"delivery_result": result}

# ---------------------------------------------------------------------------
# trend_node
# Updates DynamoDB. Always runs — failure here never blocks delivery.
# ---------------------------------------------------------------------------
 
def trend_node(state: DigestGraphState) -> dict:
    """
    Writes the completed run state back to DynamoDB.
    This is the only node that writes to DynamoDB.
 
    Runs after delivery regardless of delivery success — the run record
    and trend state should always be updated even if the email failed.
    Exceptions are caught and logged; they set run_status to DEGRADED
    rather than propagating and crashing the graph.
    """
    logger.info({"node": "trend", "run_id": state["run_id"]})
 
    delivery_result  = state.get("delivery_result")
    synthesis_result = state.get("synthesis_result")
    scoring_result   = state.get("scoring_result")
    topic_result     = state.get("topic_result")
    context          = state.get("context")

    # Trend node always runs — guard against upstream nodes that returned None
    # due to earlier failures. Degrade gracefully rather than crash.
    if not synthesis_result or not scoring_result or not topic_result:
        logger.warning({
            "node":    "trend",
            "message": "One or more upstream results missing — writing degraded run record",
            "run_id":  state["run_id"],
        })
        return {
            "trend_result": None,
            "run_status":   RunStatus.DEGRADED,
        }
 
    from node_definitions.trend import run as trend_run
    from contracts.nodes import TrendTaskInput
 
    # Collect rework counts from both supervisors for the run record
    rework_counts        = state.get("rework_counts", {})
    input_rework_count   = rework_counts.get(NodeName.INPUT_SUPERVISOR.value, 0)
    output_rework_count  = rework_counts.get(NodeName.OUTPUT_SUPERVISOR.value, 0)
 
    task_input = TrendTaskInput(
        run_id                = state["run_id"],
        topic                 = topic_result.topic,
        focus_angle           = topic_result.focus_angle,
        scored_articles       = scoring_result.scored_articles,
        new_signals           = synthesis_result.new_signals,
        trend_confirmations   = synthesis_result.trend_confirmations,
        digest_summary        = synthesis_result.digest_summary,
        existing_trends       = context.active_trends if context else [],
        source_reputation_map = context.source_reputation_map if context else {},
        delivery_sent         = delivery_result.sent if delivery_result else False,
        input_rework_count    = input_rework_count,
        output_rework_count   = output_rework_count,
    )
 
    result = trend_run(task_input)
 
    return {
        "trend_result": result,
        "run_status":   result.run_status,
    }

# ---------------------------------------------------------------------------
# Conditional edge functions
# These are called by LangGraph to determine which edge to follow.
# They read a SupervisorDecision from state and return a string node name.
# LangGraph maps that string to the next node via the conditional_edge map.
# ---------------------------------------------------------------------------
 
def route_input_supervisor(state: DigestGraphState) -> str:
    """
    Called by LangGraph after input_supervisor runs.
    Returns the name of the next node to execute.
    """
    decision = state.get("input_supervisor_decision")
    rework_count = state["rework_counts"].get(NodeName.INPUT_SUPERVISOR.value, 0)
 
    # Force proceed if max reworks exceeded — degraded mode, never infinite loop
    if decision is None or rework_count >= 2:
        if rework_count >= 2:
            logger.warning({
                "node":    "input_supervisor",
                "message": "Max reworks reached — proceeding in degraded mode",
                "run_id":  state["run_id"],
            })
        return NodeName.SYNTHESIS.value
 
    return (
        NodeName.TOPIC.value
        if decision.route == SupervisorRoute.REWORK
        else NodeName.SYNTHESIS.value
    )
 
 
def route_output_supervisor(state: DigestGraphState) -> str:
    """
    Called by LangGraph after output_supervisor runs.
    Returns the name of the next node to execute.
    """
    decision = state.get("output_supervisor_decision")
    rework_count = state["rework_counts"].get(NodeName.OUTPUT_SUPERVISOR.value, 0)
 
    if decision is None or rework_count >= 2:
        if rework_count >= 2:
            logger.warning({
                "node":    "output_supervisor",
                "message": "Max reworks reached — proceeding in degraded mode",
                "run_id":  state["run_id"],
            })
        return NodeName.DELIVERY.value
 
    return (
        NodeName.SYNTHESIS.value
        if decision.route == SupervisorRoute.REWORK
        else NodeName.DELIVERY.value
    )