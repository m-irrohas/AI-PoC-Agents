"""Workflow condition functions for AI-PoC-Agents-v2."""

from typing import Dict, Any, Literal
from ..core.state import PoCState, get_next_phase, should_repeat_phase as core_should_repeat


def should_continue_phase(state: PoCState) -> Literal["continue", "repeat", "next_phase", "end"]:
    """Determine if current phase should continue, repeat, or advance."""
    
    current_phase = state["current_phase"]
    phase_score = state["phase_scores"].get(current_phase, 0.0)
    score_threshold = state["score_threshold"]
    max_iterations = state["max_iterations"]
    current_iteration = state["iteration"]
    
    # Check if workflow should end
    if not state.get("should_continue", True):
        return "end"
    
    if state.get("error_message"):
        return "end"
    
    # Check if phase needs to repeat due to low score
    if phase_score > 0 and phase_score < score_threshold and current_iteration < max_iterations:
        return "repeat"
    
    # Check if phase is complete (has a score and meets threshold)
    if phase_score >= score_threshold:
        next_phase = get_next_phase(current_phase)
        if next_phase:
            return "next_phase"
        else:
            return "end"  # All phases complete
    
    # Continue with current phase if no score yet or first iteration
    return "continue"


def route_next_agent(state: PoCState) -> str:
    """Route to the appropriate agent based on current phase."""
    
    current_phase = state["current_phase"]
    
    # Map phases to agents
    phase_to_agent = {
        "problem_identification": "problem_agent",
        "idea_generation": "problem_agent", 
        "idea_selection": "problem_agent",
        "poc_design": "poc_agent",
        "poc_implementation": "poc_agent",
        "poc_execution": "poc_agent",
        "result_evaluation": "evaluation_agent",
        "reflection": "evaluation_agent",
        "reporting": "evaluation_agent"
    }
    
    agent = phase_to_agent.get(current_phase, "end")
    
    if agent == "end" or not state.get("should_continue", True):
        return "end"
    
    return agent


def evaluate_phase_completion(state: PoCState) -> Literal["repeat", "next_phase", "end"]:
    """Evaluate if phase is complete and determine next action."""
    
    current_phase = state["current_phase"]
    phase_score = state["phase_scores"].get(current_phase, 0.0)
    score_threshold = state["score_threshold"]
    
    # Check if phase meets quality threshold
    if phase_score >= score_threshold:
        next_phase = get_next_phase(current_phase)
        if next_phase:
            return "next_phase"
        else:
            return "end"
    
    # Check if we should repeat
    if core_should_repeat(state):
        return "repeat"
    
    # If can't repeat and doesn't meet threshold, still advance
    # (this handles cases where max iterations reached)
    next_phase = get_next_phase(current_phase)
    if next_phase:
        return "next_phase"
    else:
        return "end"


def handle_error(state: PoCState) -> Dict[str, Any]:
    """Handle workflow errors and cleanup."""
    
    error_msg = state.get("error_message", "Unknown error occurred")
    
    # Log error
    state["logs"].append(f"WORKFLOW ERROR: {error_msg}")
    state["should_continue"] = False
    
    return state