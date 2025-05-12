import json
from pathlib import Path
import click

from chatbot_graph import build_graph
from chatbot_state import ChatState
from chatbot_nodes import greeting_run

HISTORY_FILE = Path("chat_history.json")


def _load_history():
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text())
    return []


def _save_history(messages):
    HISTORY_FILE.write_text(json.dumps(messages, ensure_ascii=False, indent=2))


graph = build_graph()


@click.command()
def chat():
    
    messages = _load_history()
    if messages:
        print("🔄  Restored previous session.")
        state: ChatState = {"messages": messages}
        
        last = next((m for m in reversed(messages) if m.startswith("assistant:")), None)
        if last:
            print(f"Bot: {last.split(': ', 1)[1]}")
    else:
        state: ChatState = {"messages": []}
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
