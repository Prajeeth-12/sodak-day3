"""Where the two databases live, and the model to use. (Given.)"""
import os
from pathlib import Path

from app.memory import RunStore
from app.placement_db import PlacementDb


def _load_env() -> None:
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("'\"")
            if k and k not in os.environ:
                os.environ[k] = v


_load_env()

AGENT_DB = os.environ.get("AGENT_DB", "agent.db")
PLACEMENT_DB = os.environ.get("PLACEMENT_DB", "placement.db")
PROVIDER = os.environ.get("PROVIDER", "nvidia" if os.environ.get("NVIDIA_API_KEY") else "gemini")
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "meta/llama-3.2-11b-vision-instruct")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
MODEL = NVIDIA_MODEL if (PROVIDER == "nvidia" or os.environ.get("NVIDIA_API_KEY")) else GEMINI_MODEL


def open_stores() -> tuple[RunStore, PlacementDb]:
    store, placement = RunStore(AGENT_DB), PlacementDb(PLACEMENT_DB)
    store.migrate()
    placement.migrate()
    return store, placement


def make_provider(mock: bool, slow: float = 0.0):
    if mock:
        from app.providers import booking_mock

        return booking_mock(slow)
    if PROVIDER == "nvidia" or os.environ.get("NVIDIA_API_KEY"):
        from app.providers import NvidiaProvider

        return NvidiaProvider(model=NVIDIA_MODEL, api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)
    from app.providers import GeminiProvider

    return GeminiProvider(GEMINI_MODEL)

