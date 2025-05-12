import uuid, datetime as dt
from pathlib import Path
import json

from chatbot_state import ChatState
from rules import classify




HISTORY_FILE = Path("chat_history.json")


def _append(msgs, who, text):
    return msgs + [f"{who}: {text}"]


def _save_history(messages):
    """
    Persist the full chat log to disk so that next login
    can restore the conversation seamlessly.
    """
    HISTORY_FILE.write_text(json.dumps(messages, ensure_ascii=False, indent=2))




def router_run(state: ChatState) -> ChatState:
    """
    Decide whether the user wants to leave.  
    If so, flag `exit=True` so that the graph can branch to the exit node.
    """
    text = state.get("user_input", "").strip().lower()
    if text in {"exit", "quit"}:
        return {"exit": True}
    return {}


def greeting_run(state: ChatState) -> ChatState:
    msg = (
        "Welcome to Complaints Capture virtual assistant! "
        "Tell me what you’d like help with "
        "(e.g., lost card, account balance, transfer…)."
    )
    return {
        "assistant_response": msg,
        "messages": _append(state.get("messages", []), "assistant", msg),
    }




def request_collection_run(state: ChatState) -> ChatState:
    req = state["user_input"].strip()
    msg = "Thanks! Let me look into that…"
    return {
        "assistant_response": msg,
        "messages": _append(_append(state["messages"], "user", req),
                            "assistant", msg),
    }


def intent_classification_run(state: ChatState) -> ChatState:
    intent = classify(state["user_input"])
    if intent is None:
        err = "I’m not sure I understood. Could you rephrase?"
        return {
            "intent": None,
            "assistant_response": err,
            "messages": _append(state["messages"], "assistant", err),
            "error_msg": err,
        }

    msg = (f"It sounds like **{intent.replace('_', ' ')}**. "
           "Is that correct? (yes/no)")
    return {
        "intent": intent,
        "assistant_response": msg,
        "messages": _append(state["messages"], "assistant", msg),
        "awaiting": "confirm",
    }


def confirmation_run(state: ChatState) -> ChatState:
    ans = state["user_input"].strip().lower()
    msgs = _append(state["messages"], "user", ans)

    if ans in {"yes", "y"}:
        return {"confirmed": True, "awaiting": None, "messages": msgs}
    if ans in {"no", "n"}:
        retry = "Okay, please tell me again what you need help with."
        return {
            "confirmed": False,
            "assistant_response": retry,
            "messages": _append(msgs, "assistant", retry),
            "awaiting": None,
        }

    oops = "Just reply with yes or no, please."
    return {
        "confirmed": None,
        "assistant_response": oops,
        "messages": _append(msgs, "assistant", oops),
        "error_msg": oops,
        "awaiting": None,
    }


def execution_run(state: ChatState) -> ChatState:
    ticket = f"BNK-{uuid.uuid4().hex[:6].upper()}"
    ts = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    msg = (f"Request **{ticket}** logged on {ts}. "
           "Anything else I can help with?")
    return {
        "result": ticket,
        "assistant_response": msg,
        "messages": _append(state["messages"], "assistant", msg),
    }




def error_run(state: ChatState) -> ChatState:
    msg = state.get("error_msg") or "Sorry, something went wrong."
    return {
        "assistant_response": msg,
        "messages": _append(state["messages"], "assistant", msg),
    }


def exit_run(state: ChatState) -> ChatState:
    """
    Save history and end the session.
    """
    bye = "Good-bye! I’ve saved our conversation. See you next time."
    msgs = _append(state["messages"], "assistant", bye)

    
    _save_history(msgs)

    return {
        "assistant_response": bye,
        "messages": msgs,
        "done": True,          
    }

def router_run(state: ChatState) -> ChatState:
    text = state.get("user_input", "").strip().lower()
    if text in {"exit", "quit"}:
        return {"exit_requested": True}   
    return {}

def router_run(state: ChatState) -> ChatState:
    text = state.get("user_input", "").strip().lower()

    if text in {"exit", "quit"}:
        return {"exit_requested": True}

    if text in {"history", "show history", "chat history"}:
        return {"history_requested": True}

    return {}

def show_history_run(state: ChatState) -> ChatState:          # ← NEW NODE
    """
    Return the full chat log (or the last N turns) so the user can review it.
    """
    lines = state.get("messages", [])
    pretty = "\n".join(lines[-20:]) or "No previous conversation found."
    msg = f"Here’s our recent chat:\n\n{pretty}"
    return {
        "assistant_response": msg,
        "messages": _append(state["messages"], "assistant", msg),
    }
