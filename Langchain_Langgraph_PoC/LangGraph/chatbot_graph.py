
from __future__ import annotations         

from pathlib import Path
from typing import Literal, Union

from langgraph.graph import StateGraph, START, END
from langgraph.pregel import RetryPolicy

from chatbot_state import ChatState
from chatbot_nodes import (
    router_run, greeting_run, request_collection_run,
    intent_classification_run, confirmation_run,
    form_transfer_run, execution_run,
    error_run, fallback_run, show_history_run, exit_run,
)


def build_graph():
    sg = StateGraph(ChatState)

    sg.add_node("router",   router_run)
    sg.add_node("collect",  request_collection_run)
    sg.add_node("classify", intent_classification_run,
                retry=RetryPolicy(max_attempts=3))
    sg.add_node("confirm",  confirmation_run)
    sg.add_node("form_transfer", form_transfer_run)
    sg.add_node("execute",  execution_run,
                retry=RetryPolicy(max_attempts=3))
    sg.add_node("error",    error_run)
    sg.add_node("fallback", fallback_run)
    sg.add_node("history",  show_history_run)
    sg.add_node("exit",     exit_run)

    sg.add_edge(START, "router")

    def next_from_router(state: ChatState):
        if state.get("exit_requested"):
            return "exit"
        if state.get("history_requested"):
            return "history"
        if state.get("awaiting") == "confirm":
            return "confirm"
        if state.get("awaiting") == "multistep_form":
            return "form_transfer"
        return "collect"

    sg.add_conditional_edges(
        "router", next_from_router,
        path_map=["exit", "history", "confirm", "form_transfer", "collect"],
    )

    sg.add_edge("collect", "classify")

    def next_from_classify(state: ChatState):
        if state.get("intent") is None:
            return "error"
        if state.get("awaiting") == "multistep_form":
            return "form_transfer"
        return END

    sg.add_conditional_edges(
        "classify", next_from_classify, path_map=["error", "form_transfer", END]
    )

    def next_from_form(state: ChatState):
        if state.get("awaiting") == "multistep_form":
            return "form_transfer"
        return "confirm"

    sg.add_conditional_edges(
        "form_transfer", next_from_form, path_map=["form_transfer", "confirm"],
    )

    def next_from_confirm(state: ChatState):
        if state.get("confirmed") is None:
            return "error"
        if state["confirmed"]:
            return "execute"
        return "collect"

    sg.add_conditional_edges(
        "confirm", next_from_confirm, path_map=["execute", "collect", "error"]
    )

    sg.add_edge("error",    "fallback")
    sg.add_edge("fallback", "router")

    sg.add_edge("history", END)
    sg.add_edge("execute", END)
    sg.add_edge("exit",    END)

    return sg.compile()


def create_visual_graph(fmt: Literal["png", "ascii", "mermaid"] = "png",
    filename: Union[str, Path] = "chatbot_workflow",
) -> Path:
    """Compile the graph and save it as SVG or PNG."""
    g = build_graph()
    graph_obj = g.get_graph()

    filename = Path(filename)
    if fmt == "png":
        out_path = filename.with_suffix(".png")
        png = graph_obj.draw_mermaid_png()      # <- still supported
        out_path.write_bytes(png)
    elif fmt == "ascii":
        out_path = filename.with_suffix(".txt")
        out_path.write_text(graph_obj.draw_ascii())
    elif fmt == "mermaid":
        out_path = filename.with_suffix(".mmd")  # Mermaid source
        out_path.write_text(graph_obj.draw_mermaid())
    elif fmt == "png":
        out_path = filename.with_suffix(".png")
        out_path.write_bytes(graph_obj.draw_mermaid_png())
    else:
        raise ValueError("fmt must be 'png' | 'ascii' | 'mermaid'")

    return out_path.resolve()

def save_svg(path: Union[str, Path] = "chatbot_workflow.svg") -> Path:
    """
    Back-compat wrapper so old code that imports `save_svg`
    keeps working.  Simply calls `create_visual_graph(fmt="svg")`.
    """
    return create_visual_graph(fmt="svg", filename=path)


if __name__ == "__main__":
    import argparse, sys

    p = argparse.ArgumentParser(description="Export chatbot workflow diagram.")
    p.add_argument("--fmt", choices=("svg", "png"), default="svg")
    p.add_argument("--output", default="chatbot_workflow",
                   help="Base filename (extension auto-added).")
    args = p.parse_args()

    path = create_visual_graph(fmt=args.fmt, filename=args.output)
    print("Graph written to", path)
    sys.exit(0)
