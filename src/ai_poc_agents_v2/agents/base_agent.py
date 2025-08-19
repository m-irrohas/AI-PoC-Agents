"""Base agent class for AI-PoC-Agents-v2."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
import json
import time
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from ..core.state import PoCState, PhaseResult, AgentType
from ..core.config import Config

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all AI-PoC-Agents-v2 agents."""
    
    def __init__(self, agent_type: AgentType, config: Config):
        self.agent_type = agent_type
        self.config = config
        self.llm = self._create_llm()
        self.logger = logging.getLogger(f"{__name__}.{agent_type}")
    
    def _create_llm(self) -> ChatOpenAI:
        """Create LLM instance for this agent."""
        model_name = self.config.get_model_for_agent(self.agent_type)
        
        return ChatOpenAI(
            model=model_name,
            api_key=self.config.model.api_key,
            base_url=self.config.model.base_url,
            temperature=self.config.model.temperature,
            max_tokens=self.config.model.max_tokens,
        )
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        pass
    
    @abstractmethod
    def execute_phase(self, state: PoCState) -> Dict[str, Any]:
        """Execute the agent's main logic for the current phase."""
        pass
    
    def __call__(self, state: PoCState) -> PoCState:
        """Execute agent and update state."""
        start_time = time.time()
        
        self.logger.info(f"Executing {self.agent_type} in phase {state['current_phase']}")
        
        # Execute agent logic
        result = self.execute_phase(state)
        
        execution_time = time.time() - start_time
        
        # Create phase result
        phase_result = PhaseResult(
            phase=state["current_phase"],
            agent=self.agent_type,
            iteration=state["iteration"],
            success=result.get("success", True),
            score=result.get("score", 0.7),
            output=result.get("output", {}),
            feedback=result.get("feedback", ""),
            artifacts=result.get("artifacts", []),
            execution_time=execution_time,
            timestamp=datetime.now()
        )
        
        # Update state
        state["phase_results"].append(phase_result)
        
        # Update agent memory
        if self.agent_type not in state["agent_memory"]:
            state["agent_memory"][self.agent_type] = {}
        
        state["agent_memory"][self.agent_type].update(result.get("memory", {}))
        
        # Add artifacts and logs
        state["artifacts"].extend(result.get("artifacts", []))
        
        log_message = result.get("log", f"Executed {self.agent_type} successfully in {execution_time:.2f}s")
        state["logs"].append(f"{self.agent_type}: {log_message}")
        
        self.logger.info(f"{self.agent_type} completed with score {phase_result.score:.3f} in {execution_time:.2f}s")
        
        state["updated_at"] = datetime.now()
        return state
    
    def _generate_response(
        self, 
        system_prompt: str, 
        user_prompt: str,
        context: Optional[Dict[str, Any]] = None,
        response_format: str = "text"
    ) -> str:
        """Generate response using the LLM."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def _generate_structured_response(
        self, 
        system_prompt: str, 
        user_prompt: str,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate structured response in JSON format."""
        
        structured_prompt = f"""
{user_prompt}

Please respond in valid JSON format following this schema:
{json.dumps(schema, indent=2)}

Ensure your response is valid JSON that can be parsed.
"""
        
        response = self._generate_response(system_prompt, structured_prompt)
        
        # Try to extract JSON from response
        if "```json" in response:
            json_start = response.find("```json") + 7
            json_end = response.find("```", json_start)
            json_content = response[json_start:json_end].strip()
        else:
            json_content = response.strip()
        
        # Attempt to parse JSON, fall back to default if fails
        try:
            if json_content:
                parsed_json = json.loads(json_content)
                return parsed_json
            else:
                self.logger.error("Empty JSON content")
                return self._get_default_response(schema)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            self.logger.error(f"Response content: {response[:500]}...")  # Limit log size
            # Return a default structure based on schema
            return self._get_default_response(schema)
    
    def _get_default_response(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate default response structure based on schema."""
        default_response = {}
        
        for key, value_type in schema.items():
            if value_type == "string":
                default_response[key] = ""
            elif value_type == "list":
                default_response[key] = []
            elif value_type == "dict":
                default_response[key] = {}
            elif value_type == "number":
                default_response[key] = 0.0
            elif value_type == "boolean":
                default_response[key] = False
            else:
                default_response[key] = None
        
        return default_response
    
    def _get_project_context(self, state: PoCState) -> str:
        """Get project context for prompts."""
        project = state["project"]
        
        context = f"""
PoC Project Information:
- Theme: {project.theme}
- Description: {project.description}
- Domain: {project.domain}
- Requirements: {', '.join(project.requirements) if project.requirements else 'None specified'}
- Constraints: {', '.join(project.constraints) if project.constraints else 'None specified'}
- Success Criteria: {', '.join(project.success_criteria) if project.success_criteria else 'None specified'}
- Target Users: {', '.join(project.target_users) if project.target_users else 'Not specified'}
- Timeline: {project.timeline_days} days
- Technology Preferences: {', '.join(project.technology_preferences) if project.technology_preferences else 'No preferences'}

Current Phase: {state["current_phase"]}
Iteration: {state["iteration"]}
Overall Score: {state["overall_score"]:.3f}
"""
        
        return context
    
    def _get_previous_results(self, state: PoCState, same_agent_only: bool = False) -> str:
        """Get results from previous phases/iterations."""
        results = []
        
        for result in state["phase_results"]:
            if same_agent_only and result.agent != self.agent_type:
                continue
                
            results.append(
                f"Phase: {result.phase} | Agent: {result.agent} | "
                f"Iteration: {result.iteration} | Score: {result.score:.3f}\n"
                f"Feedback: {result.feedback}\n"
                f"Artifacts: {', '.join(result.artifacts) if result.artifacts else 'None'}\n"
            )
        
        return "Previous Results:\n" + "\n".join(results) if results else "No previous results."
    
    def _save_artifact(self, content: str, filename: str, state: PoCState, subdirectory: Optional[str] = None) -> str:
        """Save content as artifact and return the path."""
        workspace_path = Path(state["workspace_path"])
        
        if subdirectory:
            output_dir = workspace_path / subdirectory
        else:
            output_dir = workspace_path / state["current_phase"]
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        artifact_path = str(output_path)
        self.logger.info(f"Saved artifact: {artifact_path}")
        
        return artifact_path
    
    def _save_json_artifact(self, data: Dict[str, Any], filename: str, state: PoCState, subdirectory: Optional[str] = None) -> str:
        """Save JSON data as artifact and return the path."""
        content = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        return self._save_artifact(content, filename, state, subdirectory)
    
    def _get_agent_memory(self, state: PoCState) -> Dict[str, Any]:
        """Get this agent's memory."""
        return state["agent_memory"].get(self.agent_type, {})
    
    def _update_agent_memory(self, state: PoCState, updates: Dict[str, Any]) -> None:
        """Update this agent's memory."""
        if self.agent_type not in state["agent_memory"]:
            state["agent_memory"][self.agent_type] = {}
        
        state["agent_memory"][self.agent_type].update(updates)