from __future__ import annotations

import json
from pathlib import Path
import click

from chatbot_graph import build_graph, create_visual_graph
from chatbot_state import ChatState
from chatbot_nodes import greeting_run


HISTORY_FILE = Path("./LangGraph/chat_history.json")


def _load_history() -> list[str]:
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text())
    return []


def _save_history(messages: list[str]) -> None:
    HISTORY_FILE.write_text(json.dumps(messages, ensure_ascii=False, indent=2))


def _session_was_closed(messages: list[str]) -> bool:
    """
    Return True iff the last assistant message says “Good-bye!”,
    which means the exit node already ended that session.
    """
    if not messages:
        return False
    last_assistant = next(
        (m for m in reversed(messages) if m.startswith("assistant:")), ""
    )
    return "Good-bye!" in last_assistant


graph = build_graph()


@click.command()
@click.option(
    "--save-graph",
    type=click.Path(writable=True, dir_okay=False, path_type=Path),
    help="Render the workflow diagram (PNG) to the given file and exit.",
)
def chat(save_graph: Path | None) -> None:
    """
    Start an interactive chat session, or export the graph if --save-graph is used.
    """

    if save_graph:
        fp = create_visual_graph(fmt="png", filename=save_graph)
        print(f"Workflow diagram written to {fp}")
        return

    messages = _load_history()
    state: ChatState

    if messages and not _session_was_closed(messages):

        print("🔄  Restored previous session.")
        state = {"messages": messages}
        last_bot = next(
            (m for m in reversed(messages) if m.startswith("assistant:")), None
        )
        if last_bot:
            print(f"Bot: {last_bot.split(': ', 1)[1]}")
    else:

        state = {"messages": []}
        state.update(greeting_run(state))
        print(f"Bot: {state['assistant_response']}")

    while not state.get("done"):
        user = input("You: ")
        state["user_input"] = user
        state = graph.invoke(state)
        print(f"Bot: {state['assistant_response']}")
        _save_history(state["messages"])


if __name__ == "__main__":
    chat()
