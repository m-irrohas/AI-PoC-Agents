"""Configuration management for AI-PoC-Agents-v2."""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import os


@dataclass
class ModelConfig:
    """Model configuration."""
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    temperature: float = 0.1
    max_tokens: int = 4000
    
    # Model assignments for different agents
    problem_identifier: str = "gpt-4o-mini"
    idea_generator: str = "gpt-4o-mini" 
    idea_selector: str = "gpt-4o-mini"
    poc_designer: str = "gpt-4o-mini"
    poc_implementer: str = "gpt-4o-mini"
    poc_executor: str = "gpt-4o-mini"
    result_evaluator: str = "gpt-4o-mini"
    reflector: str = "gpt-4o-mini"
    reporter: str = "gpt-4o-mini"


@dataclass
class WorkflowConfig:
    """Workflow configuration."""
    max_iterations: int = 3
    score_threshold: float = 0.7
    timeout_minutes: int = 30
    enable_parallel_execution: bool = False
    save_intermediate_results: bool = True
    workspace_cleanup: bool = False


@dataclass
class ExecutionConfig:
    """Execution environment configuration."""
    docker_enabled: bool = True
    container_timeout_minutes: int = 10
    memory_limit_mb: int = 1024
    cpu_limit: float = 1.0
    network_enabled: bool = False
    allowed_languages: list = field(default_factory=lambda: ["python", "javascript", "bash"])
    package_managers: list = field(default_factory=lambda: ["pip", "npm", "apt"])


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    
    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        model_config = ModelConfig(**data.get("model", {}))
        workflow_config = WorkflowConfig(**data.get("workflow", {}))
        execution_config = ExecutionConfig(**data.get("execution", {}))
        
        return cls(
            model=model_config,
            workflow=workflow_config, 
            execution=execution_config
        )
    
    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        data = {
            "model": self.model.__dict__,
            "workflow": self.workflow.__dict__,
            "execution": self.execution.__dict__
        }
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def get_model_for_agent(self, agent_type: str) -> str:
        """Get the model configuration for a specific agent."""
        return getattr(self.model, agent_type, "gpt-4o-mini")
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        if not self.model.api_key:
            raise ValueError("OpenAI API key is required")
        
        if self.workflow.score_threshold < 0 or self.workflow.score_threshold > 1:
            raise ValueError("Score threshold must be between 0 and 1")
        
        if self.workflow.max_iterations < 1:
            raise ValueError("Max iterations must be at least 1")
        
        return True