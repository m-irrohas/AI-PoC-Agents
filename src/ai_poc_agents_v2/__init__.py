"""AI-PoC-Agents-v2: Automated PoC Framework with Multi-Agent System."""

__version__ = "2.0.0"
__author__ = "AI-PoC-Agents Team"
__description__ = "Automated Proof of Concept Framework using Multi-Agent System"

from .core.state import PoCState, PoCProject, create_initial_state
from .workflow.orchestrator import PoCWorkflow
from .agents import ProblemAgent, PoCAgent, EvaluationAgent

__all__ = [
    "PoCState",
    "PoCProject", 
    "create_initial_state",
    "PoCWorkflow",
    "ProblemAgent",
    "PoCAgent",
    "EvaluationAgent"
]