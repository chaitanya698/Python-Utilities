"""
Very small helper to persist/retrieve the rolling chat log.
Writes to ./chat_memory.json next to the program.
"""
from __future__ import annotations
import json, typing as t
from pathlib import Path

_MEM_FILE = Path("chat_memory.json")


def load() -> list[str]:
    if _MEM_FILE.exists():
        try:
            data = json.loads(_MEM_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except Exception:
            pass                      
    return []


def save(messages: t.Sequence[str]) -> None:
    try:
        _MEM_FILE.write_text(
            json.dumps(list(messages), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        
        pass


def clear() -> None:
    _MEM_FILE.unlink(missing_ok=True)
