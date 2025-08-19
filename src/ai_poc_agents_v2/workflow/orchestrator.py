"""Main workflow orchestrator for AI-PoC-Agents-v2."""

from typing import Dict, Any, Generator
import logging
from pathlib import Path
from datetime import datetime

# Note: LangGraph imports would be here in production
# For now, we'll create a simplified workflow system
# from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import MemorySaver

from ..core.state import PoCState, get_next_phase, get_phase_agents, update_phase_score
from ..core.config import Config
from ..agents import ProblemAgent, PoCAgent, EvaluationAgent
from .conditions import should_continue_phase, route_next_agent, evaluate_phase_completion

logger = logging.getLogger(__name__)


class PoCWorkflow:
    """Main workflow orchestrator for AI-PoC-Agents-v2."""
    
    def __init__(self, config: Config):
        self.config = config
        self.agents = self._create_agents()
        
    def _create_agents(self) -> Dict[str, Any]:
        """Create all agent instances."""
        return {
            "problem_agent": ProblemAgent("problem_identifier", self.config),  # Handles all problem phases
            "poc_agent": PoCAgent("poc_designer", self.config),  # Handles all PoC phases
            "evaluation_agent": EvaluationAgent("result_evaluator", self.config)  # Handles all evaluation phases
        }
    
    def run(self, state: PoCState, thread_id: str = "default") -> PoCState:
        """Run the complete PoC workflow."""
        
        project_name = state["project"].theme
        logger.info(f"Starting AI-PoC-Agents-v2 workflow for: {project_name}")
        
        try:
            # Main workflow loop
            max_steps = 50  # Safety limit
            step_count = 0
            
            while state.get("should_continue", True) and step_count < max_steps:
                step_count += 1
                logger.info(f"Workflow step {step_count}: Phase {state['current_phase']} (iteration {state['iteration']})")
                
                # Execute current phase
                state = self._execute_current_phase(state)
                
                if not state.get("should_continue", True):
                    break
                
                # Determine next action
                action = should_continue_phase(state)
                logger.info(f"Phase controller action: {action}")
                
                if action == "next_phase":
                    state = self._advance_to_next_phase(state)
                elif action == "repeat":
                    state = self._repeat_current_phase(state)
                elif action == "end":
                    state["should_continue"] = False
                elif action == "continue":
                    # Continue with current phase - no state change needed
                    pass
                
                # Update timestamp
                state["updated_at"] = datetime.now()
            
            if step_count >= max_steps:
                logger.warning(f"Workflow stopped at step limit: {max_steps}")
                state["logs"].append(f"WARNING: Workflow reached step limit of {max_steps}")
            
            logger.info("AI-PoC-Agents-v2 workflow completed")
            return state
            
        except Exception as e:
            logger.error(f"Workflow failed: {str(e)}")
            state["error_message"] = str(e)
            state["should_continue"] = False
            return state
    
    def _execute_current_phase(self, state: PoCState) -> PoCState:
        """Execute the current phase with appropriate agent."""
        
        current_phase = state["current_phase"]
        
        # Route to appropriate agent
        agent_name = route_next_agent(state)
        
        if agent_name == "end":
            state["should_continue"] = False
            return state
        
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        # Update current agent in state
        agent_type_map = {
            "problem_agent": "problem_identifier",
            "poc_agent": "poc_designer",
            "evaluation_agent": "result_evaluator"
        }
        
        state["current_agent"] = agent_type_map.get(agent_name, agent_name)
        
        # Execute agent
        agent = self.agents[agent_name]
        state = agent(state)
        
        # Calculate phase score from latest result
        if state["phase_results"]:
            latest_result = state["phase_results"][-1]
            if latest_result.phase == current_phase:
                update_phase_score(state, latest_result.score)
        
        return state
    
    def _advance_to_next_phase(self, state: PoCState) -> PoCState:
        """Advance workflow to next phase."""
        
        current_phase = state["current_phase"]
        next_phase = get_next_phase(current_phase)
        
        if next_phase:
            logger.info(f"Advancing from {current_phase} to {next_phase}")
            
            # Mark current phase as completed
            if current_phase not in state["completed_phases"]:
                state["completed_phases"].append(current_phase)
            
            # Update to next phase
            state["current_phase"] = next_phase
            state["iteration"] = 0
            
            # Update agent type based on new phase
            agent_type_map = {
                "problem_identification": "problem_identifier",
                "idea_generation": "idea_generator", 
                "idea_selection": "idea_selector",
                "poc_design": "poc_designer",
                "poc_implementation": "poc_implementer",
                "poc_execution": "poc_executor",
                "result_evaluation": "result_evaluator",
                "reflection": "reflector",
                "reporting": "reporter"
            }
            
            state["current_agent"] = agent_type_map.get(next_phase, "unknown")
            
        else:
            logger.info("All phases completed")
            state["should_continue"] = False
        
        return state
    
    def _repeat_current_phase(self, state: PoCState) -> PoCState:
        """Repeat current phase with incremented iteration."""
        
        current_phase = state["current_phase"]
        new_iteration = state["iteration"] + 1
        
        logger.info(f"Repeating phase {current_phase}, iteration {new_iteration}")
        
        state["iteration"] = new_iteration
        
        return state
    
    def stream(self, state: PoCState, config: Dict[str, Any] = None) -> Generator[PoCState, None, None]:
        """Stream workflow execution for monitoring."""
        
        project_name = state["project"].theme
        logger.info(f"Streaming AI-PoC-Agents-v2 workflow for: {project_name}")
        
        max_steps = 50
        step_count = 0
        
        yield state  # Initial state
        
        try:
            while state.get("should_continue", True) and step_count < max_steps:
                step_count += 1
                
                # Execute current phase
                state = self._execute_current_phase(state)
                yield state  # Yield after each phase execution
                
                if not state.get("should_continue", True):
                    break
                
                # Determine next action
                action = should_continue_phase(state)
                
                if action == "next_phase":
                    state = self._advance_to_next_phase(state)
                elif action == "repeat":
                    state = self._repeat_current_phase(state)
                elif action == "end":
                    state["should_continue"] = False
                
                # Update timestamp
                state["updated_at"] = datetime.now()
                
                yield state  # Yield after state updates
            
            logger.info("Workflow streaming completed")
            
        except Exception as e:
            logger.error(f"Workflow streaming failed: {str(e)}")
            state["error_message"] = str(e)
            state["should_continue"] = False
            yield state
    
    def get_workflow_summary(self, state: PoCState) -> Dict[str, Any]:
        """Get summary of workflow execution."""
        
        return {
            "project_theme": state["project"].theme,
            "current_phase": state["current_phase"],
            "completed_phases": state["completed_phases"],
            "overall_score": state["overall_score"],
            "phase_scores": state["phase_scores"],
            "total_iterations": sum(1 for result in state["phase_results"]),
            "artifacts_count": len(state["artifacts"]),
            "execution_time": (state["updated_at"] - state["started_at"]).total_seconds() if "updated_at" in state else 0,
            "should_continue": state["should_continue"],
            "error_message": state.get("error_message")
        }
    
    def save_state(self, state: PoCState, path: Path) -> None:
        """Save workflow state to file."""
        
        import json
        
        # Convert state to serializable format
        serializable_state = dict(state)
        
        # Handle non-serializable objects
        serializable_state["project"] = state["project"].__dict__
        serializable_state["started_at"] = state["started_at"].isoformat()
        serializable_state["updated_at"] = state["updated_at"].isoformat()
        
        if state.get("selected_idea"):
            idea = state["selected_idea"]
            serializable_state["selected_idea"] = idea.__dict__ if hasattr(idea, '__dict__') else idea
        
        if state.get("implementation"):
            impl = state["implementation"]
            serializable_state["implementation"] = impl.__dict__ if hasattr(impl, '__dict__') else impl
        
        if state.get("evaluation_results"):
            eval_results = state["evaluation_results"]
            serializable_state["evaluation_results"] = eval_results.__dict__ if hasattr(eval_results, '__dict__') else eval_results
        
        # Convert phase results
        serializable_state["phase_results"] = [
            {
                "phase": result.phase,
                "agent": result.agent,
                "iteration": result.iteration,
                "success": result.success,
                "score": result.score,
                "output": result.output,
                "feedback": result.feedback,
                "artifacts": result.artifacts,
                "execution_time": result.execution_time,
                "timestamp": result.timestamp.isoformat()
            }
            for result in state["phase_results"]
        ]
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(serializable_state, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Workflow state saved to: {path}")


def create_poc_workflow(config: Config) -> PoCWorkflow:
    """Create and return a configured AI-PoC-Agents-v2 workflow."""
    return PoCWorkflow(config)