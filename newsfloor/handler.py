"""
handler.py
 
Lambda entry point. EventBridge calls lambda_handler once per scheduled run.
 
"""

import json
import logging
from datetime import datetime
 
logger = logging.getLogger()
logger.setLevel(logging.INFO)
 
 
def lambda_handler(event: dict, context) -> dict:
    """
    Entry point for EventBridge scheduled trigger.
 
    event:   EventBridge event payload (we don't use it — schedule has no data)
    context: Lambda context object (used for logging the request ID)
    """
    run_id = datetime.utcnow().strftime("%Y-%m-%d")
 
    logger.info(json.dumps({
        "run_id":     run_id,
        "request_id": context.aws_request_id,
        "message":    "Digest agent triggered — orchestrator not yet implemented (Phase 2 stub)",
    }))
 
    # Phase 3 replaces this with:
    #   from orchestrator import Orchestrator
    #   result = Orchestrator().run(run_id)
    #   return {"statusCode": 200, "body": result.model_dump_json()}
 
    return {
        "statusCode": 200,
        "body": json.dumps({
            "run_id":  run_id,
            "status":  "stub — pipeline not yet implemented",
        })
    }