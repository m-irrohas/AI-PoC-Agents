"""Workflow Orchestrator for AI-PoC-Agents-v2 specialized agents."""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from pathlib import Path

from ..core.state import PoCState
from ..core.config import Config
from .problem_identification_agent import ProblemIdentificationAgent
from .search_problem_agent import SearchProblemAgent
from .idea_generation_agent import IdeaGenerationAgent
from .idea_reflection_agent import IdeaReflectionAgent
from .poc_design_agent import PoCDesignAgent
from .implementation_agent import ImplementationAgent
from .execute_agent import ExecuteAgent
from .reflection_agent import ReflectionAgent
from .reporting_agent import ReportingAgent

logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """Orchestrates the execution of specialized PoC development agents."""
    
    def __init__(self, config: Config):
        """Initialize the workflow orchestrator with specialized agents."""
        self.config = config
        self.agents = self._initialize_agents()
        
        # Define the workflow phases and their corresponding agents
        self.workflow_phases = [
            ("problem_identification", "problem_identification"),
            ("problem_search", "search_problem"), 
            ("idea_generation", "idea_generation"),
            ("idea_reflection", "idea_reflection"),
            ("poc_design", "poc_design"),
            ("implementation", "implementation"),
            ("execution", "execute"),
            ("reflection", "reflection"),
            ("reporting", "reporting")
        ]
        
        # Track execution history
        self.execution_history = []
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all specialized agents."""
        return {
            "problem_identification": ProblemIdentificationAgent(self.config),
            "search_problem": SearchProblemAgent(self.config),
            "idea_generation": IdeaGenerationAgent(self.config),
            "idea_reflection": IdeaReflectionAgent(self.config),
            "poc_design": PoCDesignAgent(self.config),
            "implementation": ImplementationAgent(self.config),
            "execute": ExecuteAgent(self.config),
            "reflection": ReflectionAgent(self.config),
            "reporting": ReportingAgent(self.config)
        }
    
    def execute_workflow(self, state: PoCState, start_from_phase: Optional[str] = None) -> PoCState:
        """Execute the complete PoC development workflow."""
        
        self.logger.info("Starting PoC development workflow with specialized agents")
        self._log_workflow_start(state)
        
        # Load existing results if starting from a later phase
        if start_from_phase:
            self._load_existing_results(state)
        
        # Determine starting phase
        start_index = 0
        if start_from_phase:
            for i, (phase_name, _) in enumerate(self.workflow_phases):
                if phase_name == start_from_phase:
                    start_index = i
                    break
        
        # Execute workflow phases
        for i, (phase_name, agent_name) in enumerate(self.workflow_phases[start_index:], start_index):
            if not state.get("should_continue", True):
                self.logger.warning(f"Workflow stopped at phase {phase_name} due to error or user request")
                break
            
            self.logger.info(f"Executing phase {i+1}/{len(self.workflow_phases)}: {phase_name}")
            
            # Validate agent exists
            if agent_name not in self.agents:
                self.logger.error(f"Agent {agent_name} not found for phase {phase_name}")
                state["error_message"] = f"Agent {agent_name} not found"
                state["should_continue"] = False
                break
            
            # Update state with current phase
            state["current_phase"] = phase_name
            
            # Execute agent
            agent = self.agents[agent_name]
            result = agent(state)
            
            # Record agent execution in history
            self.execution_history.append(agent_name)
            
            # Check for execution errors that need implementation fix (feedback loop)
            if phase_name == "execution" and result.get("needs_implementation_fix", False):
                feedback_success = self._handle_execution_feedback(state, result, i)
                if feedback_success:
                    continue  # Skip to next phase after successful fix
            
            # Check for reflection issues that need idea change (feedback loop)
            if phase_name == "reflection" and result.get("needs_idea_change", False):
                feedback_success = self._handle_reflection_feedback(state, result, i)
                if feedback_success:
                    continue  # Skip to next phase after successful idea change
            
            # Check if agent execution was successful
            if "error_message" in state and state["error_message"]:
                self.logger.error(f"Agent {agent_name} failed: {state['error_message']}")
                break
            
            # Log phase completion
            self._log_phase_completion(phase_name, state)
            
            # Check for critical failures
            if self._should_stop_workflow(state, phase_name):
                self.logger.error(f"Critical failure in {phase_name}, stopping workflow")
                break
            
            # Optional: Check quality gates
            if self._has_quality_gate(phase_name):
                if not self._passes_quality_gate(state, phase_name):
                    self.logger.warning(f"Quality gate failed for {phase_name}")
                    # Could implement retry logic here if needed
        
        # Finalize workflow
        state["workflow_completed"] = state.get("should_continue", True)
        state["completed_at"] = datetime.now()
        
        self._log_workflow_completion(state)
        
        # Print execution history
        self._print_execution_history()
        
        return state
    
    def execute_single_phase(self, state: PoCState, phase_name: str) -> PoCState:
        """Execute a single phase of the workflow."""
        
        # Find agent for the phase
        agent_name = None
        for p_name, a_name in self.workflow_phases:
            if p_name == phase_name:
                agent_name = a_name
                break
        
        if not agent_name:
            raise ValueError(f"Unknown phase: {phase_name}")
        
        self.logger.info(f"Executing single phase: {phase_name}")
        
        # Load existing results for dependencies
        self._load_existing_results(state)
        
        # Update state and execute
        state["current_phase"] = phase_name
        agent = self.agents[agent_name]
        state = agent(state)
        
        # Record agent execution in history
        self.execution_history.append(agent_name)
        
        return state
    
    def get_workflow_status(self, state: PoCState) -> Dict[str, Any]:
        """Get comprehensive workflow status and progress."""
        
        completed_phases = [result.phase for result in state.get("phase_results", [])]
        current_phase = state.get("current_phase", "")
        
        phase_status = []
        for phase_name, agent_name in self.workflow_phases:
            status = {
                "phase": phase_name,
                "agent": agent_name,
                "status": "completed" if phase_name in completed_phases else (
                    "in_progress" if phase_name == current_phase else "pending"
                ),
                "score": None,
                "artifacts": []
            }
            
            # Find phase result for score and artifacts
            for result in state.get("phase_results", []):
                if result.phase == phase_name:
                    status["score"] = result.score
                    status["artifacts"] = result.artifacts
                    break
            
            phase_status.append(status)
        
        return {
            "total_phases": len(self.workflow_phases),
            "completed_phases": len(completed_phases),
            "current_phase": current_phase,
            "workflow_completed": state.get("workflow_completed", False),
            "overall_score": state.get("overall_score", 0.0),
            "phase_status": phase_status,
            "should_continue": state.get("should_continue", True),
            "error_message": state.get("error_message", ""),
            "total_artifacts": len(state.get("artifacts", [])),
            "workspace_path": state.get("workspace_path", "")
        }
    
    def retry_failed_phase(self, state: PoCState, phase_name: str, max_retries: int = 2) -> PoCState:
        """Retry a failed phase with optional iteration."""
        
        self.logger.info(f"Retrying phase: {phase_name}")
        
        # Increment iteration counter for this phase
        iteration_key = f"{phase_name}_iteration"
        current_iteration = state.get(iteration_key, 0)
        
        if current_iteration >= max_retries:
            self.logger.error(f"Max retries ({max_retries}) exceeded for phase {phase_name}")
            state["error_message"] = f"Phase {phase_name} failed after {max_retries} retries"
            return state
        
        state[iteration_key] = current_iteration + 1
        state["iteration"] = state[iteration_key]  # Set global iteration for artifact naming
        
        # Clear previous error state
        state["should_continue"] = True
        if "error_message" in state:
            del state["error_message"]
        
        # Execute the phase
        return self.execute_single_phase(state, phase_name)
    
    def get_phase_dependencies(self, phase_name: str) -> List[str]:
        """Get the phases that must complete before the given phase."""
        dependencies = []
        
        for p_name, _ in self.workflow_phases:
            if p_name == phase_name:
                break
            dependencies.append(p_name)
        
        return dependencies
    
    def validate_phase_readiness(self, state: PoCState, phase_name: str) -> Dict[str, Any]:
        """Validate that all dependencies for a phase are satisfied."""
        
        dependencies = self.get_phase_dependencies(phase_name)
        completed_phases = [result.phase for result in state.get("phase_results", [])]
        
        missing_dependencies = [dep for dep in dependencies if dep not in completed_phases]
        
        # Check specific requirements for each phase
        requirements_met = True
        missing_requirements = []
        
        if phase_name == "problem_search" and "problem_identification" not in completed_phases:
            requirements_met = False
            missing_requirements.append("Problem analysis results")
        
        if phase_name == "idea_generation" and not state.get("problem_search"):
            requirements_met = False
            missing_requirements.append("Problem search results")
        
        if phase_name == "idea_reflection" and not state.get("ideas"):
            requirements_met = False
            missing_requirements.append("Generated ideas")
        
        if phase_name == "poc_design" and not state.get("idea_evaluation"):
            requirements_met = False
            missing_requirements.append("Idea evaluation results")
        
        if phase_name == "implementation" and not state.get("technical_specification"):
            requirements_met = False
            missing_requirements.append("Technical design specification")
        
        if phase_name == "execution" and not state.get("implementation_results"):
            requirements_met = False
            missing_requirements.append("Implementation code")
        
        return {
            "ready": len(missing_dependencies) == 0 and requirements_met,
            "missing_dependencies": missing_dependencies,
            "missing_requirements": missing_requirements,
            "can_proceed": len(missing_dependencies) == 0  # Can proceed even with missing reqs
        }
    
    def _log_workflow_start(self, state: PoCState) -> None:
        """Log workflow start with project information."""
        project = state["project"]
        print("\033[93m" + "="*80)  # Yellow color
        print("🚀 STARTING POC DEVELOPMENT WORKFLOW")
        print("="*80)
        print(f"📋 Project: {project.theme}")
        print(f"📝 Description: {project.description}")
        print(f"🎯 Task Type: {getattr(project, 'task_type', 'Not identified')}")
        print(f"📅 Timeline: {project.timeline_days} days")
        print(f"📁 Workspace: {state.get('workspace_path', 'Not set')}")
        print(f"🔄 Total Phases: {len(self.workflow_phases)}")
        print("="*80 + "\033[0m")  # Reset color
    
    def _log_phase_completion(self, phase_name: str, state: PoCState) -> None:
        """Log phase completion with results."""
        # Find the most recent phase result
        phase_result = None
        for result in reversed(state.get("phase_results", [])):
            if result.phase == phase_name:
                phase_result = result
                break
        
        if phase_result:
            status_emoji = "✅" if phase_result.success else "❌"
            score_color = "\033[92m" if phase_result.score > 0.7 else "\033[93m" if phase_result.score > 0.5 else "\033[91m"
            
            print(f"{status_emoji} {phase_name.upper()} COMPLETED")
            print(f"   Score: {score_color}{phase_result.score:.3f}\033[0m | Time: {phase_result.execution_time:.1f}s | Artifacts: {len(phase_result.artifacts)}")
            if phase_result.feedback:
                print(f"   {phase_result.feedback.strip()}")
    
    def _log_workflow_completion(self, state: PoCState) -> None:
        """Log workflow completion summary."""
        completed = state.get("workflow_completed", False)
        status_emoji = "🎉" if completed else "⚠️"
        status_color = "\033[92m" if completed else "\033[91m"
        
        print(status_color + "="*80)
        print(f"{status_emoji} WORKFLOW {'COMPLETED' if completed else 'STOPPED'}")
        print("="*80)
        
        # Summary statistics
        phase_results = state.get("phase_results", [])
        if phase_results:
            avg_score = sum(r.score for r in phase_results) / len(phase_results)
            total_time = sum(r.execution_time for r in phase_results)
            total_artifacts = sum(len(r.artifacts) for r in phase_results)
            
            print(f"📊 Phases Completed: {len(phase_results)}/{len(self.workflow_phases)}")
            print(f"📈 Average Score: {avg_score:.3f}")
            print(f"⏱️ Total Time: {total_time:.1f}s")
            print(f"📁 Total Artifacts: {total_artifacts}")
            print(f"🗂️ Workspace: {state.get('workspace_path', 'Not set')}")
        
        if not completed and state.get("error_message"):
            print(f"❌ Error: {state['error_message']}")
        
        print("="*80 + "\033[0m")  # Reset color
    
    def _should_stop_workflow(self, state: PoCState, phase_name: str) -> bool:
        """Check if workflow should stop due to critical failures."""
        
        # Check for explicit stop conditions
        if not state.get("should_continue", True):
            return True
        
        # Check for critical phase failures
        critical_phases = ["problem_identification", "search_problem", "idea_generation"]
        if phase_name in critical_phases:
            phase_results = [r for r in state.get("phase_results", []) if r.phase == phase_name]
            if phase_results and not phase_results[-1].success:
                return True
        
        return False
    
    def _has_quality_gate(self, phase_name: str) -> bool:
        """Check if phase has a quality gate."""
        quality_gate_phases = ["problem_identification", "idea_reflection", "poc_design", "implementation"]
        return phase_name in quality_gate_phases
    
    def _passes_quality_gate(self, state: PoCState, phase_name: str) -> bool:
        """Check if phase passes its quality gate."""
        phase_results = [r for r in state.get("phase_results", []) if r.phase == phase_name]
        if not phase_results:
            return False
        
        latest_result = phase_results[-1]
        
        # Define minimum scores for quality gates
        minimum_scores = {
            "problem_identification": 0.6,
            "idea_reflection": 0.5,
            "poc_design": 0.7,
            "implementation": 0.6
        }
        
        minimum_score = minimum_scores.get(phase_name, 0.5)
        return latest_result.score >= minimum_score
    
    def _load_existing_results(self, state: PoCState) -> None:
        """Load existing results from previous phases."""
        workspace_path = Path(state["workspace_path"])
        
        # Phase mapping for agent memory keys
        phase_memory_map = {
            "problem_identification": "problem_identification",
            "problem_search": "search_problem",
            "idea_generation": "idea_generation",
            "idea_reflection": "idea_reflection",
            "poc_design": "poc_design",
            "implementation": "implementation",
            "execution": "execute",
            "reflection": "reflection",
            "reporting": "reporting"
        }
        
        for phase_name, agent_name in self.workflow_phases:
            phase_dir = workspace_path / phase_name
            if phase_dir.exists():
                # Look for the most recent iteration file
                iteration_files = list(phase_dir.glob(f"{phase_name}_iteration_*.json"))
                if iteration_files:
                    # Sort by iteration number and get the latest
                    latest_file = max(iteration_files, key=lambda x: int(x.stem.split('_')[-1]))
                    
                    # Check if file exists and is readable
                    if not latest_file.exists():
                        self.logger.warning(f"File {latest_file} does not exist")
                        continue
                    
                    if not latest_file.is_file():
                        self.logger.warning(f"{latest_file} is not a file")
                        continue
                    
                    # Load JSON data
                    import json
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if not content.strip():
                        self.logger.warning(f"File {latest_file} is empty")
                        continue
                    
                    data = json.loads(content)
                    
                    if not isinstance(data, dict):
                        self.logger.warning(f"File {latest_file} does not contain valid JSON object")
                        continue
                    
                    # Load into agent memory
                    memory_key = phase_memory_map.get(phase_name, agent_name)
                    if "agent_memory" not in state:
                        state["agent_memory"] = {}
                    if memory_key not in state["agent_memory"]:
                        state["agent_memory"][memory_key] = {}
                    
                    # Load specific data based on phase
                    if phase_name == "problem_identification":
                        state["agent_memory"][memory_key]["problem_analysis"] = data
                    elif phase_name == "problem_search":
                        state["agent_memory"][memory_key]["search_results"] = data
                        state["problem_search"] = data
                    elif phase_name == "idea_generation":
                        state["agent_memory"][memory_key]["generated_ideas"] = data.get("ideas", [])
                        # Convert dict ideas to PoCIdea objects
                        from ..core.state import PoCIdea
                        ideas_data = data.get("ideas", [])
                        poc_ideas = []
                        for idea_dict in ideas_data:
                            poc_idea = PoCIdea(
                                id=idea_dict.get("id", ""),
                                title=idea_dict.get("title", ""),
                                description=idea_dict.get("description", ""),
                                approach=idea_dict.get("technical_approach", ""),
                                technologies=idea_dict.get("technology_stack", []),
                                implementation_complexity=idea_dict.get("implementation_complexity", 3),
                                expected_impact=idea_dict.get("expected_impact", 3),
                                innovation_score=idea_dict.get("innovation_score", 0.5),
                                feasibility_score=idea_dict.get("feasibility_score", 0.5),
                                total_score=idea_dict.get("total_score", 0.5),
                                pros=idea_dict.get("pros", []),
                                cons=idea_dict.get("cons", []),
                                required_skills=idea_dict.get("recommended_technologies", []),
                                estimated_effort_hours=idea_dict.get("estimated_effort_hours", 24),
                                risk_factors=idea_dict.get("risk_mitigation", [])
                            )
                            poc_ideas.append(poc_idea)
                        state["ideas"] = poc_ideas
                    elif phase_name == "idea_reflection":
                        state["agent_memory"][memory_key]["idea_evaluation"] = data.get("evaluation_results", {})
                        state["agent_memory"][memory_key]["implementation_readiness"] = data.get("implementation_readiness", {})
                        # Find and set the selected idea if available
                        if "ideas" in state:
                            # Select highest-scoring idea from existing ideas
                            ideas = state["ideas"]
                            if ideas:
                                selected_idea = max(ideas, key=lambda x: x.total_score)
                                state["selected_idea"] = selected_idea
                                self.logger.info(f"Auto-selected highest scoring idea: {selected_idea.title} (score: {selected_idea.total_score:.2f})")
                    elif phase_name == "poc_design":
                        state["agent_memory"][memory_key]["design_specification"] = data
                        # Reconstruct PoCImplementation object from design data
                        from ..core.state import PoCImplementation
                        if "selected_idea" in state:
                            implementation = PoCImplementation(idea_id=state["selected_idea"].id)
                            # Load design data into implementation
                            spec_data = data.get("technical_specification", {})
                            implementation.architecture = spec_data.get("architecture", {})
                            implementation.tech_stack = spec_data.get("technology_stack", [])
                            implementation.environment_config = spec_data.get("environment_configuration", {})
                            implementation.test_cases = spec_data.get("test_cases", [])
                            implementation.dependencies = spec_data.get("dependencies", [])
                            state["implementation"] = implementation
                    elif phase_name == "implementation":
                        state["agent_memory"][memory_key]["implementation_results"] = data
                        # Load generated code files info if available
                        if "implementation_summary" in data:
                            summary = data["implementation_summary"]
                            state["agent_memory"][memory_key]["generated_files"] = summary.get("code_files", {})
                            state["agent_memory"][memory_key]["total_lines"] = data.get("code_generation_context", {}).get("total_lines", 0)
                        # Also check for the actual generated code directory and load files
                        code_dir = workspace_path / "generated_code" / f"iteration_{state.get('iteration', 0)}"
                        if code_dir.exists() and "implementation" in state:
                            # Load actual code files into implementation.code_files
                            implementation = state["implementation"]
                            
                            # Load Python files
                            for code_file in code_dir.glob("*.py"):
                                try:
                                    with open(code_file, 'r', encoding='utf-8') as f:
                                        implementation.code_files[code_file.name] = f.read()
                                        self.logger.info(f"Loaded code file: {code_file.name}")
                                except Exception as e:
                                    self.logger.warning(f"Failed to load code file {code_file}: {e}")
                            
                            # Also load requirements.txt if exists
                            requirements_file = code_dir / "requirements.txt"
                            if requirements_file.exists():
                                try:
                                    with open(requirements_file, 'r', encoding='utf-8') as f:
                                        implementation.code_files["requirements.txt"] = f.read()
                                        self.logger.info("Loaded requirements.txt")
                                except Exception as e:
                                    self.logger.warning(f"Failed to load requirements.txt: {e}")
                            
                            # Load other common files
                            for filename in ["README.md", "config.py", "setup.py", "Dockerfile"]:
                                file_path = code_dir / filename
                                if file_path.exists():
                                    try:
                                        with open(file_path, 'r', encoding='utf-8') as f:
                                            implementation.code_files[filename] = f.read()
                                            self.logger.info(f"Loaded {filename}")
                                    except Exception as e:
                                        self.logger.warning(f"Failed to load {filename}: {e}")
                            
                            state["implementation_code_path"] = str(code_dir)
                            self.logger.info(f"Loaded {len(implementation.code_files)} total files from {code_dir}")
                    elif phase_name == "execution":
                        state["agent_memory"][memory_key]["execution_results"] = data
                        state["agent_memory"][memory_key]["test_results"] = data.get("test_results", {})
                        state["agent_memory"][memory_key]["performance_metrics"] = data.get("performance_metrics", {})
                        # Load evaluation results into state for reflection agent
                        if "evaluation" in data:
                            from ..core.state import EvaluationResult
                            eval_data = data["evaluation"]
                            evaluation_result = EvaluationResult(
                                overall_score=eval_data.get("overall_score", 0.0),
                                technical_score=eval_data.get("technical_score", 0.0),
                                business_score=eval_data.get("business_score", 0.0),
                                innovation_score=eval_data.get("innovation_score", 0.0),
                                success_criteria_met=eval_data.get("success_criteria_met", []),
                                quantitative_metrics=eval_data.get("quantitative_metrics", {}),
                                qualitative_feedback=eval_data.get("qualitative_feedback", ""),
                                strengths=eval_data.get("strengths", []),
                                weaknesses=eval_data.get("weaknesses", []),
                                improvement_suggestions=eval_data.get("improvement_suggestions", []),
                                next_steps=eval_data.get("next_steps", []),
                                lessons_learned=eval_data.get("lessons_learned", [])
                            )
                            state["evaluation_results"] = evaluation_result
                            self.logger.info("Loaded evaluation results from execution data")
                    elif phase_name == "reflection":
                        state["agent_memory"][memory_key]["reflection_analysis"] = data
                        state["agent_memory"][memory_key]["lessons_learned"] = data.get("lessons_learned", [])
                        state["agent_memory"][memory_key]["improvement_recommendations"] = data.get("improvement_recommendations", [])
                    elif phase_name == "reporting":
                        state["agent_memory"][memory_key]["reports"] = data
                        state["agent_memory"][memory_key]["generated_documents"] = data.get("generated_documents", [])
                    
                    self.logger.info(f"Loaded existing results for {phase_name} from {latest_file}")
    
    def set_local_code_paths(self, paths: List[str]) -> None:
        """Set local code search paths for SearchProblemAgent."""
        if "search_problem" in self.agents:
            self.agents["search_problem"].set_local_code_paths(paths)
    
    def _handle_execution_feedback(self, state: PoCState, execution_result: Dict[str, Any], 
                                 current_phase_index: int) -> bool:
        """Handle feedback loop when execution fails and needs implementation fix."""
        
        # Check feedback loop limits to prevent infinite loops
        feedback_count = state.get("feedback_loop_count", 0)
        max_feedback_loops = 3  # Maximum number of feedback loops
        
        if feedback_count >= max_feedback_loops:
            self.logger.warning(f"Maximum feedback loops ({max_feedback_loops}) reached, continuing to next phase")
            return False
        
        # Increment feedback loop counter
        state["feedback_loop_count"] = feedback_count + 1
        
        self.logger.info(f"Execution failed with fixable errors, starting feedback loop #{feedback_count + 1}")
        
        # Prepare improvement instructions for implementation agent
        fix_instructions = execution_result.get("fix_instructions", [])
        error_analysis = execution_result.get("output", {}).get("error_analysis", {})
        
        # Store feedback information in state for implementation agent
        state["execution_feedback"] = {
            "fix_instructions": fix_instructions,
            "error_analysis": error_analysis,
            "execution_errors": execution_result.get("output", {}).get("execution_success", True),
            "feedback_iteration": feedback_count + 1
        }
        
        # Find implementation phase index
        implementation_phase_index = None
        for i, (phase_name, agent_name) in enumerate(self.workflow_phases):
            if phase_name == "implementation":
                implementation_phase_index = i
                break
        
        if implementation_phase_index is None:
            self.logger.error("Implementation phase not found in workflow")
            return False
        
        # Re-execute implementation phase with feedback
        self.logger.info("Re-executing implementation phase with error feedback")
        
        # Update state to implementation phase
        state["current_phase"] = "implementation"
        
        # Execute implementation agent with feedback
        implementation_agent = self.agents["implementation"]
        implementation_result = implementation_agent(state)
        
        # Record agent execution in history
        self.execution_history.append("implementation")
        
        if not implementation_result.get("success", False):
            self.logger.error("Implementation agent failed to fix issues")
            return False
        
        # Re-execute execution phase
        self.logger.info("Re-executing execution phase after implementation fix")
        
        # Update state back to execution phase
        state["current_phase"] = "execution"
        
        # Execute execution agent again
        execution_agent = self.agents["execute"]
        new_execution_result = execution_agent(state)
        
        # Record agent execution in history
        self.execution_history.append("execute")
        
        # Check if the fix worked
        if new_execution_result.get("success", False) or not new_execution_result.get("needs_implementation_fix", False):
            self.logger.info(f"Feedback loop #{feedback_count + 1} successful - execution fixed")
            # Clear feedback data
            if "execution_feedback" in state:
                del state["execution_feedback"]
            return True
        else:
            self.logger.warning(f"Feedback loop #{feedback_count + 1} did not resolve execution issues")
            return False
    
    def _handle_reflection_feedback(self, state: PoCState, reflection_result: Dict[str, Any], 
                                  current_phase_index: int) -> bool:
        """Handle feedback loop when reflection suggests trying a different idea."""
        
        # Check feedback loop limits to prevent infinite loops
        idea_change_count = state.get("idea_change_count", 0)
        max_idea_changes = 2  # Maximum number of idea changes
        
        if idea_change_count >= max_idea_changes:
            self.logger.warning(f"Maximum idea changes ({max_idea_changes}) reached, continuing to next phase")
            return False
        
        # Increment idea change counter
        state["idea_change_count"] = idea_change_count + 1
        
        self.logger.info(f"Reflection suggests trying different idea, starting idea change #{idea_change_count + 1}")
        
        # Prepare improvement instructions for idea reflection agent
        change_reasons = reflection_result.get("idea_change_reasons", [])
        reflection_analysis = reflection_result.get("output", {})
        
        # Store feedback information in state for idea reflection agent
        state["idea_change_feedback"] = {
            "change_reasons": change_reasons,
            "reflection_analysis": reflection_analysis,
            "previous_idea_issues": reflection_analysis.get("reflection_summary", {}),
            "change_iteration": idea_change_count + 1
        }
        
        # Find idea_reflection phase index
        idea_reflection_phase_index = None
        for i, (phase_name, agent_name) in enumerate(self.workflow_phases):
            if phase_name == "idea_reflection":
                idea_reflection_phase_index = i
                break
        
        if idea_reflection_phase_index is None:
            self.logger.error("Idea reflection phase not found in workflow")
            return False
        
        # Re-execute from idea_reflection through implementation to reflection
        self.logger.info("Re-executing from idea_reflection with alternative idea selection")
        
        # Re-execute idea_reflection phase
        state["current_phase"] = "idea_reflection"
        idea_reflection_agent = self.agents["idea_reflection"]
        idea_reflection_result = idea_reflection_agent(state)
        
        # Record agent execution in history
        self.execution_history.append("idea_reflection")
        
        if not idea_reflection_result.get("success", False):
            self.logger.error("Idea reflection agent failed to select alternative idea")
            return False
        
        # Re-execute subsequent phases: poc_design -> implementation -> execution -> reflection
        phases_to_rerun = ["poc_design", "implementation", "execution", "reflection"]
        
        for phase_name in phases_to_rerun:
            self.logger.info(f"Re-executing {phase_name} with new idea")
            state["current_phase"] = phase_name
            
            agent = self.agents[phase_name]
            phase_result = agent(state)
            
            # Record agent execution in history
            self.execution_history.append(phase_name)
            
            if not phase_result.get("success", False):
                self.logger.error(f"{phase_name} failed during idea change iteration")
                return False
        
        # Check if the new idea worked better
        new_reflection_result = state["agent_memory"].get("reflector", {})
        new_overall_success = new_reflection_result.get("overall_assessment", 0.0)
        
        if new_overall_success > reflection_analysis.get("overall_success", 0.0):
            self.logger.info(f"Idea change #{idea_change_count + 1} improved results")
            # Clear feedback data
            if "idea_change_feedback" in state:
                del state["idea_change_feedback"]
            return True
        else:
            self.logger.warning(f"Idea change #{idea_change_count + 1} did not improve results significantly")
            return False
    
    def _print_execution_history(self) -> None:
        """Print the execution history showing agent execution order."""
        if not self.execution_history:
            print("No agents executed.")
            return
        
        print("\\033[94m" + "="*60)  # Blue color
        print("📈 AGENT EXECUTION HISTORY")
        print("="*60)
        print(f"Total Executions: {len(self.execution_history)}")
        print("Execution Order:")
        
        for i, agent_name in enumerate(self.execution_history, 1):
            print(f"  {i:2d}. {agent_name}")
        
        print("="*60 + "\\033[0m")  # Reset color