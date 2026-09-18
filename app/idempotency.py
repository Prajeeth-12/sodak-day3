"""Stable fingerprints for side effects. Hashing, used for exactly-once."""
import hashlib
import json
import re
from datetime import date


def _normalize(value):
    """Whole floats become ints, recursively."""
    if isinstance(value, float) and value == int(value):
        return int(value)
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    return value


def canonical_json(value) -> str:  # DONE (2.2): sorted keys, no spaces, 12.0→12
    return json.dumps(_normalize(value), sort_keys=True, separators=(",", ":"))


def idempotency_key(run_id: str, step_seq: int, tool_name: str, args: dict) -> str:  # DONE (2.2): SHA-256 of [run, step, tool, args]
    return hashlib.sha256(canonical_json([run_id, step_seq, tool_name, args]).encode()).hexdigest()


def notification_dedupe_key(roll_no: str, message: str, day: date) -> str:  # DONE (3.3): SHA-256 of [student, message, date]
    collapsed = re.sub(r"\s+", " ", message).strip()
    return hashlib.sha256(canonical_json([roll_no, collapsed, day.isoformat()]).encode()).hexdigest()
