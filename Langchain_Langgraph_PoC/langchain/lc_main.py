
"""
Command-line driver for the Acme Bank virtual assistant.

Features
--------
✔ Persistent chat history on disk (chat_memory.json)
✔ Character-by-character streaming of normal bot replies
✔ Meta-commands:
    • history / /history / show history  → prints transcript instantly
    • exit / quit                       → saves & exits
"""
from __future__ import annotations

import time
import click

from chatbot_state import ChatState
from chatbot_chain import build_chain
from chatbot_nodes import greeting_run
import chatbot_memory as mem





def stream_print(text: str, delay: float = 0.02) -> None:
    """Print *text* one character at a time for a 'typing' effect."""
    for ch in text:
        print(ch, end="", flush=True)
        time.sleep(delay)
    print()  



@click.command()
@click.option(
    "--clear-history",
    is_flag=True,
    help="Start a fresh session and wipe any stored chat transcript.",
)
def chat(clear_history: bool) -> None:
    
    if clear_history:
        mem.clear()
        messages: list[str] = []
    else:
        messages = mem.load()

    chain = build_chain()
    state: ChatState = {"messages": messages}

    
    if not messages:
        state.update(greeting_run(state))
        stream_print(state["assistant_response"])
    else:
        print("Restored previous session. Continue…")

    
    while True:
        raw = input("You: ")
        user = raw.strip()

        
        if not user:
            continue
        user_lower = user.lower()

        
        if user_lower in {"exit", "quit"}:
            stream_print("Bot: Goodbye! (I’ll remember this chat.)")
            mem.save(state["messages"])
            break

        if user_lower in {"history", "/history", "show history"}:
            hist = "\n".join(state["messages"])
            if not hist:
                hist = "-- (no messages yet) --"
            print("Chat history so far:\n" + hist)
            continue
        

        
        state["user_input"] = raw             
        state = chain.invoke(state)

        stream_print(f"Bot: {state['assistant_response']}")

        
        mem.save(state["messages"])



if __name__ == "__main__":
    chat()
