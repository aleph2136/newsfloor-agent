"""
handler.py

Lambda entry point. EventBridge calls lambda_handler once per scheduled run.
"""

import json
import logging
import os
import sys
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# sys.path setup
#
# Graph internals (state.py, nodes.py, graph.py) use flat imports that expect
# their directory to be on sys.path — the same convention as tests/conftest.py.
# Replicate that here so Lambda resolves imports identically to the test suite.
#
# After these inserts, sys.path order is:
#   [0] newsfloor/graph/  — state, nodes, graph, node_definitions
#   [1] newsfloor/        — contracts, data
#   [2] /var/task/        — config.py
# ---------------------------------------------------------------------------
_this_dir  = os.path.dirname(os.path.abspath(__file__))  # .../newsfloor/
_root_dir  = os.path.dirname(_this_dir)                   # .../  (package root)
_graph_dir = os.path.join(_this_dir, "graph")             # .../newsfloor/graph/

for _p in (_root_dir, _this_dir, _graph_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import the module-level compiled graph — reused across warm Lambda invocations.
# build_graph() runs once at cold start; subsequent invocations reuse the compiled instance.
from graph import digest_graph  # noqa: E402


def lambda_handler(event: dict, context) -> dict:
    """
    Entry point for EventBridge scheduled trigger.

    event:   EventBridge event payload (not used — schedule carries no data)
    context: Lambda context object (request ID and remaining time)
    """
    run_id = datetime.utcnow().strftime("%Y-%m-%d")

    logger.info(json.dumps({
        "run_id":        run_id,
        "request_id":    context.aws_request_id,
        "remaining_ms":  context.get_remaining_time_in_millis(),
        "message":       "Digest agent triggered",
    }))

    # Guard against invocations with too little time left — Lambda kills the
    # process hard at timeout, which can leave DynamoDB writes half-done.
    remaining_ms = context.get_remaining_time_in_millis()
    if remaining_ms < 60_000:
        logger.error(json.dumps({
            "run_id":       run_id,
            "remaining_ms": remaining_ms,
            "error":        "Insufficient Lambda time remaining — skipping run",
        }))
        return {
            "statusCode": 202,
            "body": json.dumps({
                "run_id": run_id,
                "status": "skipped — insufficient time remaining",
            }),
        }

    try:
        final_state = digest_graph.invoke({"run_id": run_id, "rework_counts": {}})
        run_status = str(final_state.get("run_status", "unknown"))

        logger.info(json.dumps({
            "run_id":     run_id,
            "run_status": run_status,
            "message":    "Run complete",
        }))

        return {
            "statusCode": 200,
            "body": json.dumps({"run_id": run_id, "run_status": run_status}),
        }

    except Exception as e:
        logger.error(json.dumps({
            "run_id":  run_id,
            "error":   str(e),
            "message": "Pipeline failed with unhandled exception",
        }))
        return {
            "statusCode": 500,
            "body": json.dumps({"run_id": run_id, "status": "error", "error": str(e)}),
        }
