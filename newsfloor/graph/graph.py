"""
graph/graph.py
 
The LangGraph graph definition for the digest pipeline.
 
This file has one job: wire nodes and edges together.
No business logic lives here — that belongs in graph/nodes.py and nodes/.
 
Reading the graph
─────────────────
The flow is:
 
    START
      └─► load_context
            └─► topic ─► fetch ─► scoring
                                      └─► input_supervisor
                                               ├─ rework ─► topic (loops back)
                                               └─ proceed ─► synthesis
                                                                └─► output_supervisor
                                                                         ├─ rework ─► synthesis (loops back)
                                                                         └─ proceed ─► delivery
                                                                                          └─► trend
                                                                                                └─► END
 
Conditional edges
─────────────────
Two conditional edges exist — one per supervisor. LangGraph calls the
route_ function after the supervisor node runs and uses the returned
string to look up the next node in the routing map.
"""

from __future__ import annotations
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, START, END
from contracts.primitives import NodeName
from state import DigestGraphState
from nodes import (
    load_context,
    topic_node,
    fetch_node,
    scoring_node,
    input_supervisor,
    synthesis_node,
    output_supervisor,
    delivery_node,
    trend_node,
    route_input_supervisor,
    route_output_supervisor,
)

def build_graph() -> CompiledStateGraph:
    """
    Constructs and compiles the digest pipeline graph.
 
    Returns a compiled LangGraph graph ready to invoke.
    Call this once at Lambda cold start and reuse the instance.
 
    Usage:
        graph = build_graph()
        result = graph.invoke({"run_id": "2026-05-19", "rework_counts": {}})
    """
 
    builder = StateGraph(DigestGraphState)
 
    # -------------------------------------------------------------------------
    # Register nodes
    # The string name passed here is what LangGraph uses in edge definitions
    # and what route_ functions return. We use NodeName enum values throughout
    # so the strings are never duplicated.
    # -------------------------------------------------------------------------
    builder.add_node(NodeName.LOAD_CONTEXT.value,      load_context)
    builder.add_node(NodeName.TOPIC.value,             topic_node)
    builder.add_node(NodeName.FETCH.value,             fetch_node)
    builder.add_node(NodeName.SCORING.value,           scoring_node)
    builder.add_node(NodeName.INPUT_SUPERVISOR.value,  input_supervisor)
    builder.add_node(NodeName.SYNTHESIS.value,         synthesis_node)
    builder.add_node(NodeName.OUTPUT_SUPERVISOR.value, output_supervisor)
    builder.add_node(NodeName.DELIVERY.value,          delivery_node)
    builder.add_node(NodeName.TREND.value,             trend_node)
 
    # -------------------------------------------------------------------------
    # Static edges — unconditional transitions
    # -------------------------------------------------------------------------
    builder.add_edge(START,                             NodeName.LOAD_CONTEXT.value)
    builder.add_edge(NodeName.LOAD_CONTEXT.value,       NodeName.TOPIC.value)
    builder.add_edge(NodeName.TOPIC.value,              NodeName.FETCH.value)
    builder.add_edge(NodeName.FETCH.value,              NodeName.SCORING.value)
    builder.add_edge(NodeName.SCORING.value,            NodeName.INPUT_SUPERVISOR.value)
    builder.add_edge(NodeName.SYNTHESIS.value,          NodeName.OUTPUT_SUPERVISOR.value)
    builder.add_edge(NodeName.DELIVERY.value,           NodeName.TREND.value)
    builder.add_edge(NodeName.TREND.value,              END)
 
    # -------------------------------------------------------------------------
    # Conditional edges — supervisor routing
    #
    # add_conditional_edges(source, routing_fn, routing_map)
    #
    #   source      — the node whose output triggers the routing decision
    #   routing_fn  — called with current state, returns a string key
    #   routing_map — maps that string key to the next node name
    #
    # The routing_map explicitly lists every possible outcome so LangGraph
    # can validate the graph is complete and visualize it correctly.
    # -------------------------------------------------------------------------
    builder.add_conditional_edges(
        NodeName.INPUT_SUPERVISOR.value,
        route_input_supervisor,
        {
            NodeName.TOPIC.value:     NodeName.TOPIC.value,      # rework
            NodeName.SYNTHESIS.value: NodeName.SYNTHESIS.value,  # proceed
        }
    )
 
    builder.add_conditional_edges(
        NodeName.OUTPUT_SUPERVISOR.value,
        route_output_supervisor,
        {
            NodeName.SYNTHESIS.value: NodeName.SYNTHESIS.value,  # rework
            NodeName.DELIVERY.value:  NodeName.DELIVERY.value,   # proceed
        }
    )
 
    return builder.compile()
 
 
# Module-level compiled graph instance.
# Lambda imports this once at cold start — subsequent invocations reuse it.
digest_graph = build_graph()