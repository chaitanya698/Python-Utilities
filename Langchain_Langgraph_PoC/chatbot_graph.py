from langgraph.graph import StateGraph, START, END

from chatbot_state import ChatState
from chatbot_nodes import (
    router_run                as router,
    request_collection_run    as collect,
    intent_classification_run as classify,
    confirmation_run          as confirm,
    execution_run             as execute,
    error_run                 as handle_error,
    exit_run                  as exit_chat,
    show_history_run          as show_history,
)


def build_graph():
    sg = StateGraph(ChatState)

    sg.add_node("history", show_history)     
    sg.add_node("router",   router)
    sg.add_node("collect",  collect)
    sg.add_node("classify", classify)
    sg.add_node("confirm",  confirm)
    sg.add_node("execute",  execute)
    sg.add_node("error",    handle_error)
    sg.add_node("exit",     exit_chat)

    # ── entry edge ───────────────────────────────
    sg.add_edge(START, "router")

    # ── router fan-out ───────────────────────────
    def next_from_router(state: ChatState):
        if state.get("exit_requested"):
            return "exit"
        if state.get("history_requested"):       # ← new branch
            return "history"
        if state.get("awaiting") == "confirm":
            return "confirm"
        return "collect"

    sg.add_conditional_edges(
        "router",
        next_from_router,
        path_map=["exit", "history", "confirm", "collect"],  # ← only mapping for names we return
    )

    # ── normal flow ──────────────────────────────
    sg.add_edge("collect", "classify")

    def next_from_classify(state: ChatState):
        # if we couldn’t find an intent, send the user an error
        return "error" if state.get("intent") is None else END

    sg.add_conditional_edges(
        "classify",
        next_from_classify,
        path_map=["error", END],   # list form is fine even when one entry is END
    )

    def next_from_confirm(state: ChatState):
        if state.get("confirmed") is None:
            return "error"
        return "execute" if state["confirmed"] else "collect"

    sg.add_conditional_edges(
        "confirm",
        next_from_confirm,
        path_map=["execute", "collect", "error"],
    )

    # ── simple edges ─────────────────────────────
    sg.add_edge("history", END) 
    sg.add_edge("execute", END)
    sg.add_edge("error",   END)
    sg.add_edge("exit",    END)

    return sg.compile()
