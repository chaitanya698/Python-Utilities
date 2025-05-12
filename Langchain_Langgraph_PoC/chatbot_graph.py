from langgraph.graph import StateGraph, START, END
from chatbot_state import ChatState
from chatbot_nodes import (
    router_run                as router,
    request_collection_run    as collect,
    intent_classification_run as classify,
    confirmation_run          as confirm,
    execution_run             as execute,
    error_run                 as handle_error,
)


def build_graph():
    sg = StateGraph(ChatState)

    # ── nodes ─────────────────────────────────────────
    sg.add_node("router",   router)
    sg.add_node("collect",  collect)
    sg.add_node("classify", classify)
    sg.add_node("confirm",  confirm)
    sg.add_node("execute",  execute)
    sg.add_node("error",    handle_error)

    # ── entry ─────────────────────────────────────────
    sg.add_edge(START, "router")

    # router → collect  OR  confirm (if awaiting yes/no)
    sg.add_conditional_edges(
        "router",
        lambda s: "confirm" if s.get("awaiting") == "confirm" else "collect",
        {"collect": "collect", "confirm": "confirm"},
    )

    # collect → classify
    sg.add_edge("collect", "classify")

    # classify → error (None intent)  OR  END (ask user & wait)
    sg.add_conditional_edges(
        "classify",
        lambda s: "error" if s.get("intent") is None else "end",
        {"error": "error", "end": END},
    )

    # confirm → execute / collect / error
    sg.add_conditional_edges(
        "confirm",
        lambda s: (
            "error" if s.get("confirmed") is None
            else "execute" if s["confirmed"]
            else "collect"
        ),
        {"execute": "execute", "collect": "collect", "error": "error"},
    )
    


    # execute / error → END (finish this pass)
    sg.add_edge("execute", END)
    sg.add_edge("error",   END)

    return sg.compile()
