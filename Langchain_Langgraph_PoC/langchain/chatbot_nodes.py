# chatbot_nodes.py
import uuid, datetime as dt
from chatbot_state import ChatState
from rules import classify          # your rule-based intent classifier


def _append(msgs, who, text):
    return msgs + [f"{who}: {text}"]


def router_run(state: ChatState) -> ChatState:
    return {}                       # real routing handled by edges


def greeting_run(state: ChatState) -> ChatState:
    msg = (
        "Welcome to Complaints Capture virtual assistant!\n"
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
