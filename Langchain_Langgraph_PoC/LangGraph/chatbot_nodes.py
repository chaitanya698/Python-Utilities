import uuid, datetime as dt, json
from pathlib import Path

from chatbot_state import ChatState
from rules import classify

HISTORY_FILE = Path("chat_history.json")
MAX_RETRIES = 3


def _append(msgs, who, text):
    return msgs + [f"{who}: {text}"]


def _save_history(messages):
    HISTORY_FILE.write_text(json.dumps(messages, ensure_ascii=False, indent=2))


def router_run(state: ChatState) -> ChatState:
    """Look for high-level commands first."""
    text = state.get("user_input", "").strip().lower()

    if text in {"exit", "quit"}:
        return {"exit_requested": True}

    if text in {"history", "show history", "chat history"}:
        return {"history_requested": True}

    if state.get("awaiting") == "multistep_form":
        return {}

    return {}


def greeting_run(state: ChatState) -> ChatState:
    msg = (
        "Hello.!, this is complaints capture virtual assistant!\n"
        "Tell me what you’d like help with (e.g., *lost card*, "
        "*account balance*, *transfer* …)."
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
        "messages": _append(_append(state["messages"], "user", req), "assistant", msg),
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

    if intent == "transfer":
        msg = "Sure – let’s set up a transfer.\nWhat **amount** would you like to send?"
        return {
            "intent": intent,
            "form_name": "transfer",
            "form_step": 0,
            "form_data": {},
            "awaiting": "multistep_form",
            "assistant_response": msg,
            "messages": _append(state["messages"], "assistant", msg),
        }

    msg = f"It sounds like **{intent.replace('_', ' ')}**. " "Is that correct? (yes/no)"
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

    oops = "Just reply with *yes* or *no*, please."
    return {
        "confirmed": None,
        "assistant_response": oops,
        "messages": _append(msgs, "assistant", oops),
        "error_msg": oops,
    }


def form_transfer_run(state: ChatState) -> ChatState:
    step = state.get("form_step", 0)
    data = state.get("form_data", {})
    reply = state["user_input"].strip()

    msgs = _append(state["messages"], "user", reply)

    if step == 0:
        data["amount"] = reply
        ask = "Great. Which account number should we credit?"
        return {
            "form_step": 1,
            "form_data": data,
            "assistant_response": ask,
            "messages": _append(msgs, "assistant", ask),
            "awaiting": "multistep_form",
        }

    if step == 1:
        data["account"] = reply

        confirm_msg = (
            f"Transfer ₹{data['amount']} to **{data['account']}**. " "Proceed? (yes/no)"
        )
        return {
            "form_step": 2,
            "form_data": data,
            "assistant_response": confirm_msg,
            "messages": _append(msgs, "assistant", confirm_msg),
            "awaiting": "confirm",
            "confirmed": None,
        }

    err = "Unexpected form state."
    return {
        "assistant_response": err,
        "messages": _append(msgs, "assistant", err),
        "error_msg": err,
    }


def execution_run(state: ChatState) -> ChatState:
    ticket = f"BNK-{uuid.uuid4().hex[:6].upper()}"
    ts = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    msg = f"Request **{ticket}** logged on {ts}.\n" "Anything else I can help with?"
    return {
        "result": ticket,
        "assistant_response": msg,
        "messages": _append(state["messages"], "assistant", msg),
    }


def show_history_run(state: ChatState) -> ChatState:
    lines = state.get("messages", [])
    pretty = "\n".join(lines[-20:]) or "No previous conversation found."
    msg = f"Here’s our recent chat:\n\n{pretty}"
    return {
        "assistant_response": msg,
        "messages": _append(state["messages"], "assistant", msg),
    }


def fallback_run(state: ChatState) -> ChatState:

    retries = state.get("retries", 0) + 1
    if retries >= MAX_RETRIES:
        bye = (
            "Something kept failing. Let’s start over later. "
            "Your session is being closed."
        )
        msgs = _append(state["messages"], "assistant", bye)
        _save_history(msgs)
        return {"assistant_response": bye, "messages": msgs, "done": True}

    msg = "Let’s try that again — could you rephrase?"
    return {
        "assistant_response": msg,
        "messages": _append(state["messages"], "assistant", msg),
        "retries": retries,
    }


def error_run(state: ChatState) -> ChatState:
    msg = state.get("error_msg") or "Sorry, something went wrong."
    return {
        "assistant_response": msg,
        "messages": _append(state["messages"], "assistant", msg),
    }


def exit_run(state: ChatState) -> ChatState:
    bye = "Good-bye! I’ve saved our conversation. See you next time."
    msgs = _append(state["messages"], "assistant", bye)
    _save_history(msgs)
    return {"assistant_response": bye, "messages": msgs, "done": True}
