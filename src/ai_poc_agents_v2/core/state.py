"""State management for AI-PoC-Agents-v2 workflow."""

from typing import Dict, List, Any, Optional, Literal
from typing_extensions import TypedDict
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import uuid


PoCPhase = Literal[
    "problem_identification",
    "idea_generation", 
    "idea_selection",
    "poc_design",
    "poc_implementation", 
    "poc_execution",
    "result_evaluation",
    "reflection",
    "reporting"
]

AgentType = Literal[
    "problem_identifier",
    "idea_generator", 
    "idea_selector",
    "poc_designer",
    "poc_implementer",
    "poc_executor", 
    "result_evaluator",
    "reflector",
    "reporter"
]

@dataclass
class PoCProject:
    """PoC Project information."""
    theme: str
    description: str = ""
    domain: str = ""
    requirements: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    target_users: List[str] = field(default_factory=list)
    timeline_days: int = 7
    budget_limit: Optional[float] = None
    technology_preferences: List[str] = field(default_factory=list)

@dataclass
class PoCIdea:
    """PoC Idea with evaluation metrics."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    approach: str = ""
    technologies: List[str] = field(default_factory=list)
    implementation_complexity: int = 1  # 1-5 scale
    expected_impact: int = 1  # 1-5 scale
    feasibility_score: float = 0.0  # 0.0-1.0
    innovation_score: float = 0.0  # 0.0-1.0
    total_score: float = 0.0  # 0.0-1.0
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    required_skills: List[str] = field(default_factory=list)
    estimated_effort_hours: int = 8
    risk_factors: List[str] = field(default_factory=list)

@dataclass 
class PoCImplementation:
    """PoC Implementation details."""
    idea_id: str
    architecture: Dict[str, Any] = field(default_factory=dict)
    tech_stack: List[str] = field(default_factory=list)
    code_files: Dict[str, str] = field(default_factory=dict)  # filename -> content
    environment_config: Dict[str, Any] = field(default_factory=dict)
    test_cases: List[Dict[str, Any]] = field(default_factory=list)
    deployment_instructions: str = ""
    dependencies: List[str] = field(default_factory=list)
    docker_config: Optional[str] = None
    execution_logs: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class EvaluationResult:
    """PoC Evaluation results."""
    overall_score: float = 0.0  # 0.0-1.0
    technical_score: float = 0.0
    business_score: float = 0.0
    innovation_score: float = 0.0
    success_criteria_met: List[bool] = field(default_factory=list)
    quantitative_metrics: Dict[str, float] = field(default_factory=dict)
    qualitative_feedback: str = ""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)

@dataclass
class PhaseResult:
    """Result from a phase execution."""
    phase: PoCPhase
    agent: AgentType
    iteration: int
    success: bool
    score: float
    output: Dict[str, Any]
    feedback: str
    artifacts: List[str]
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)

class PoCState(TypedDict):
    """Central state for AI-PoC-Agents-v2 workflow."""
    
    # Project information
    project: PoCProject
    
    # Current workflow state
    current_phase: PoCPhase
    current_agent: AgentType
    iteration: int
    
    # Core workflow data
    ideas: List[PoCIdea]
    selected_idea: Optional[PoCIdea] 
    implementation: Optional[PoCImplementation]
    evaluation_results: Optional[EvaluationResult]
    
    # Execution history
    phase_results: List[PhaseResult]
    agent_memory: Dict[AgentType, Dict[str, Any]]
    
    # Quality metrics
    overall_score: float
    phase_scores: Dict[PoCPhase, float]
    
    # Configuration
    max_iterations: int
    score_threshold: float
    model_config: Dict[str, str]
    
    # Output tracking
    artifacts: List[str]
    logs: List[str]
    workspace_path: str
    sample_data_path: str
    
    # Control flags
    should_continue: bool
    error_message: Optional[str]
    completed_phases: List[PoCPhase]
    
    # Timestamps
    started_at: datetime
    updated_at: datetime


def create_initial_state(
    project: PoCProject,
    workspace_path: str,
    model_config: Optional[Dict[str, str]] = None,
    max_iterations: int = 3,
    score_threshold: float = 0.7,
    sample_data_path: str = ""
) -> PoCState:
    """Create initial state for a PoC project."""
    
    now = datetime.now()
    
    return PoCState(
        project=project,
        current_phase="problem_identification",
        current_agent="problem_identifier", 
        iteration=0,
        ideas=[],
        selected_idea=None,
        implementation=None,
        evaluation_results=None,
        phase_results=[],
        agent_memory={},
        overall_score=0.0,
        phase_scores={},
        max_iterations=max_iterations,
        score_threshold=score_threshold,
        model_config=model_config or {
            "problem_identifier": "gpt-4o",
            "idea_generator": "gpt-4o",
            "idea_selector": "gpt-4o-mini",
            "poc_designer": "gpt-4o", 
            "poc_implementer": "gpt-4o",
            "poc_executor": "gpt-4o-mini",
            "result_evaluator": "gpt-4o-mini",
            "reflector": "gpt-4o-mini",
            "reporter": "gpt-4o-mini"
        },
        artifacts=[],
        logs=[],
        workspace_path=workspace_path,
        sample_data_path=sample_data_path,
        should_continue=True,
        error_message=None,
        completed_phases=[],
        started_at=now,
        updated_at=now
    )


def get_next_phase(current_phase: PoCPhase) -> Optional[PoCPhase]:
    """Get the next phase in the PoC workflow."""
    phase_order: List[PoCPhase] = [
        "problem_identification",
        "idea_generation",
        "idea_selection", 
        "poc_design",
        "poc_implementation",
        "poc_execution",
        "result_evaluation",
        "reflection",
        "reporting"
    ]
    
    try:
        current_index = phase_order.index(current_phase)
        if current_index < len(phase_order) - 1:
            return phase_order[current_index + 1]
        return None
    except ValueError:
        return None


def get_phase_agents(phase: PoCPhase) -> List[AgentType]:
    """Get the agents that should execute in a given phase."""
    phase_agents: Dict[PoCPhase, List[AgentType]] = {
        "problem_identification": ["problem_identifier"],
        "idea_generation": ["idea_generator"],
        "idea_selection": ["idea_selector"],
        "poc_design": ["poc_designer"],
        "poc_implementation": ["poc_implementer"], 
        "poc_execution": ["poc_executor"],
        "result_evaluation": ["result_evaluator"],
        "reflection": ["reflector"],
        "reporting": ["reporter"]
    }
    
    return phase_agents.get(phase, [])


def should_repeat_phase(state: PoCState) -> bool:
    """Determine if the current phase should be repeated."""
    current_phase = state["current_phase"]
    phase_score = state["phase_scores"].get(current_phase, 0.0)
    score_threshold = state["score_threshold"] 
    max_iterations = state["max_iterations"]
    current_iteration = state["iteration"]
    
    should_repeat = (
        phase_score < score_threshold and
        current_iteration < max_iterations
    )
    
    return should_repeat


def update_phase_score(state: PoCState, score: float) -> None:
    """Update the score for the current phase."""
    current_phase = state["current_phase"]
    state["phase_scores"][current_phase] = score
    
    # Update overall score as average of completed phases
    completed_scores = list(state["phase_scores"].values())
    if completed_scores:
        state["overall_score"] = sum(completed_scores) / len(completed_scores)
    
    state["updated_at"] = datetime.now()


def get_workspace_path(state: PoCState, subdirectory: Optional[str] = None) -> Path:
    """Get workspace path for storing artifacts."""
    base_path = Path(state["workspace_path"])
    
    if subdirectory:
        return base_path / subdirectory
    
    return base_path


def add_artifact(state: PoCState, artifact_path: str) -> None:
    """Add an artifact to the state."""
    state["artifacts"].append(artifact_path)
    state["updated_at"] = datetime.now()


def log_message(state: PoCState, message: str, agent: Optional[AgentType] = None) -> None:
    """Add a log message to the state."""
    timestamp = datetime.now().isoformat()
    if agent:
        log_entry = f"[{timestamp}] {agent}: {message}"
    else:
        log_entry = f"[{timestamp}] {message}"
    
    state["logs"].append(log_entry)
    state["updated_at"] = datetime.now()