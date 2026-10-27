"""SPOT-E: test-time entropy shaping with frozen VLMs."""
from .config import SPOTEConfig
from .engine import SPOTE

__all__ = ["SPOTE", "SPOTEConfig"]
