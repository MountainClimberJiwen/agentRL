"""Prompt registry, assembly, and rendering for agentRL.

All prompt evolution happens in TEXT space — no model weights are touched.
"""

from pathlib import Path

from agentrl.prompts.assembler import PromptAssembler
from agentrl.prompts.registry import PromptRegistry, RetrievalStrategy, SelectedEvidence

PROMPTS_DIR = Path(__file__).parent

__all__ = [
    "PROMPTS_DIR",
    "PromptRegistry",
    "PromptAssembler",
    "RetrievalStrategy",
    "SelectedEvidence",
]
