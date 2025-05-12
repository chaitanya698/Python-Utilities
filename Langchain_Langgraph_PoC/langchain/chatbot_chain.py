
"""
Composable LangChain “graph” that mirrors the old LangGraph behaviour.
Requires langchain>=0.2.0.
"""
from langchain_core.runnables import RunnableLambda, RunnableBranch
from chatbot_nodes import (
    router_run,
    request_collection_run,
    intent_classification_run,
    confirmation_run,
    execution_run,
    error_run,
)


def _wrap(fn):
    """Let every node update the running state dict in-place‐style."""
    return RunnableLambda(lambda s: {**s, **fn(s)})


router   = _wrap(router_run)
collect  = _wrap(request_collection_run)
classify = _wrap(intent_classification_run)
confirm  = _wrap(confirmation_run)
execute  = _wrap(execution_run)
error    = _wrap(error_run)


classify_flow = (
    collect
    | classify
    | RunnableBranch(                         
        (lambda s: s.get("intent") is None, error),
        RunnableLambda(lambda s: s),
    )
)

confirm_flow = (
    confirm
    | RunnableBranch(
        (lambda s: s.get("confirmed") is None,  error),  
        (lambda s: s.get("confirmed") is False, collect),
        execute,                                         
    )
)

chatbot_chain = (
    router
    | RunnableBranch(
        (lambda s: s.get("awaiting") == "confirm", confirm_flow),
        classify_flow,
    )
)

def build_chain():
    return chatbot_chain
