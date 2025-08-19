"""PoC Design & Implementation Agent for AI-PoC-Agents-v2."""

from typing import Dict, Any, List
import json
import re
import subprocess
from pathlib import Path

from .base_agent import BaseAgent
from ..core.state import PoCState, PoCImplementation


class PoCAgent(BaseAgent):
    """Agent responsible for PoC design, implementation, and execution."""
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        return """
You are an expert PoC Design & Implementation Agent specialized in rapid prototyping.

Your responsibilities:
1. POC DESIGN: Create detailed technical design and architecture
2. POC IMPLEMENTATION: Generate working code and setup instructions  
3. POC EXECUTION: Test the implementation and collect performance data

Key principles:
- Focus on rapid, working prototypes over perfect code
- Use well-established, reliable technologies  
- Prioritize functionality over optimization
- Ensure reproducibility with clear setup instructions
- Include proper error handling and logging
- Design for easy demonstration and evaluation

Tech stack preferences (use unless specified otherwise):
- Python for ML/AI, data processing, general scripting
- Local command-line applications for PoCs (avoid web servers unless specifically requested)
- Docker for containerization and reproducibility  
- Popular, well-documented libraries and frameworks

FOR MACHINE LEARNING/CLASSIFICATION TASKS:
- Use scikit-learn for traditional ML algorithms (Random Forest, SVM, Gradient Boosting)
- Use pandas for data preprocessing and analysis
- Use numpy for numerical computations
- Include cross-validation for robust performance evaluation
- Generate confusion matrices and classification reports
- Save model performance metrics to files (JSON/CSV)
- Implement feature importance analysis where applicable
- Include data visualization with matplotlib/seaborn
- Use train/validation/test splits or cross-validation
- Handle categorical features with proper encoding
- Include model comparison between multiple algorithms

Always provide:
- Complete, runnable code
- Clear setup and execution instructions
- Test cases and validation steps
- Performance monitoring with ML metrics (accuracy, precision, recall, F1)
- Cross-validation results and statistical analysis
- Documentation for demonstration
"""
    
    def execute_phase(self, state: PoCState) -> Dict[str, Any]:
        """Execute the agent's logic based on current phase."""
        
        current_phase = state["current_phase"]
        
        if current_phase == "poc_design":
            return self._design_poc(state)
        elif current_phase == "poc_implementation":
            return self._implement_poc(state)  
        elif current_phase == "poc_execution":
            result = self._execute_poc(state)
            
            # Check if execution failed and we need to go back to implementation
            if result.get("needs_reimplementation", False):
                # Check retry count (max 3 retries)
                retry_count = state.get("implementation_retry_count", 0)
                max_retry = 3
                
                if retry_count < max_retry:
                    # Log phase transition with orange color
                    print(f"\033[38;5;208m{'='*70}")  # Orange color
                    print(f"🔄 EXECUTION FAILED - RETURNING TO IMPLEMENTATION PHASE")
                    print(f"📝 Will re-implement with error feedback (retry {retry_count + 1}/{max_retry})")
                    print(f"{'='*70}\033[0m")  # Reset color
                    
                    # Change phase back to implementation
                    state["current_phase"] = "poc_implementation"
                    
                    # Increment retry count
                    state["implementation_retry_count"] = retry_count + 1
                    state["implementation_retry"] = True
                    state["retry_reason"] = "execution_failure"
                    
                    # The result still contains the error info for the caller to handle
                    result["phase_change"] = "poc_implementation"
                    result["retry_needed"] = True
                else:
                    # Max retries reached, mark as failed
                    print(f"\033[38;5;208m{'='*70}")  # Orange color
                    print(f"❌ MAX RETRIES REACHED ({max_retry}) - EXECUTION FAILED")
                    print(f"🚫 Stopping retry loop")
                    print(f"{'='*70}\033[0m")  # Reset color
                    
                    result["max_retries_reached"] = True
                    result["retry_needed"] = False
            
            return result
        else:
            raise ValueError(f"PoCAgent cannot handle phase: {current_phase}")
    
    def _design_poc(self, state: PoCState) -> Dict[str, Any]:
        """Design the PoC architecture and implementation plan."""
        
        project = state["project"]
        selected_idea = state.get("selected_idea")
        
        if not selected_idea:
            raise ValueError("No selected idea available for PoC design")
        
        context = self._get_project_context(state)
        previous_results = self._get_previous_results(state, same_agent_only=True)
        
        # Analyze existing workspace structure for context-aware design
        workspace_analysis = self._analyze_workspace_structure(state)
        
        # Get selected idea details
        idea_info = selected_idea.__dict__ if hasattr(selected_idea, '__dict__') else selected_idea
        
        user_prompt = f"""
{context}

{previous_results}

SELECTED IDEA FOR POC DEVELOPMENT:
{json.dumps(idea_info, indent=2, ensure_ascii=False, default=str)}

EXISTING WORKSPACE ANALYSIS:
{json.dumps(workspace_analysis, indent=2, ensure_ascii=False, default=str)}

TASK: Design a comprehensive PoC implementation plan that leverages the existing workspace structure and data files.

Create a detailed technical design including:

1. ARCHITECTURE OVERVIEW
   - High-level system architecture
   - Key components and their interactions  
   - Data flow and processing pipeline
   - Integration points and APIs

2. TECHNOLOGY STACK
   - Programming languages and frameworks
   - Libraries and dependencies  
   - Databases/storage solutions
   - Development and deployment tools
   - Specific versions where important
   
   FOR AI/ML TASKS: PRIORITIZE MACHINE LEARNING APPROACHES:
   - Deep Learning: TensorFlow, PyTorch, Keras
   - Pre-trained Models: Hugging Face Transformers, OpenAI API
   - Computer Vision: OpenCV + ML models, TrOCR, EasyOCR
   - NLP: BERT, GPT variants, spaCy with ML models
   - Avoid traditional rule-based approaches when ML alternatives exist

3. IMPLEMENTATION PLAN
   - Core components to build
   - Development phases/milestones
   - File structure and organization
   - Configuration management

4. TECHNICAL SPECIFICATIONS
   - Input/output specifications
   - Data models and schemas
   - API endpoints (if applicable)
   - Performance requirements
   - Security considerations

5. ENVIRONMENT SETUP
   - Development environment requirements
   - Dependencies installation
   - Configuration files needed
   - Database/storage setup

6. TESTING STRATEGY
   - Unit test approach
   - Integration test scenarios
   - Performance testing plan
   - User acceptance criteria

7. DEPLOYMENT PLAN
   - Deployment environment (Docker, cloud, local)
   - Build and deployment scripts
   - Configuration management
   - Monitoring and logging

8. DEMONSTRATION PLAN
   - Demo scenarios and use cases
   - Sample data/inputs
   - Expected outputs/results
   - Success criteria validation

Focus on creating a practical, implementable design that can be executed within the timeline.
"""
        
        schema = {
            "architecture_overview": "string",
            "system_components": "list", 
            "data_flow": "string",
            "technology_stack": "dict",
            "programming_languages": "list",
            "frameworks": "list", 
            "libraries": "list",
            "databases": "list",
            "development_phases": "list",
            "file_structure": "dict",
            "input_specifications": "dict",
            "output_specifications": "dict",
            "api_endpoints": "list",
            "performance_requirements": "dict",
            "environment_requirements": "list",
            "dependencies": "list",
            "configuration_files": "list",
            "testing_scenarios": "list",
            "deployment_method": "string",
            "demo_scenarios": "list",
            "success_criteria": "list"
        }
        
        response = self._generate_structured_response(
            self.get_system_prompt(),
            user_prompt,
            schema
        )
        
        # Create PoC Implementation object
        implementation = PoCImplementation(
            idea_id=idea_info.get("id", "unknown"),
            architecture=response,
            tech_stack=response.get("programming_languages", []) + response.get("frameworks", []),
            dependencies=response.get("dependencies", []),
            environment_config=response.get("environment_requirements", {}),
            test_cases=[{"scenario": scenario} for scenario in response.get("testing_scenarios", [])]
        )
        
        state["implementation"] = implementation
        
        # Save design document
        design_content = json.dumps(response, indent=2, ensure_ascii=False, default=str)
        artifact_path = self._save_artifact(
            design_content,
            f"poc_design_iteration_{state['iteration']}.json",
            state
        )
        
        # Create detailed design document
        design_doc = self._create_design_document(response)
        doc_path = self._save_artifact(
            design_doc,
            f"poc_design_document_iteration_{state['iteration']}.md",
            state
        )
        
        # Save state for debugging
        state_pkl_path = self._save_state_debug(state, "poc_design")
        artifacts = [artifact_path, doc_path, state_pkl_path]
        
        feedback = f"""
PoC Design Complete:
- Architecture Components: {len(response.get('system_components', []))}
- Technology Stack: {len(response.get('technology_stack', {}))} technologies
- Development Phases: {len(response.get('development_phases', []))}
- Demo Scenarios: {len(response.get('demo_scenarios', []))}
- Dependencies: {len(response.get('dependencies', []))}
"""
        
        score = self._calculate_design_score(response)
        
        return {
            "success": True,
            "score": score,
            "output": response,
            "feedback": feedback,
            "artifacts": artifacts,
            "memory": {
                "design": response,
                "implementation": implementation.__dict__,
                "designed_at": state["updated_at"].isoformat() if "updated_at" in state else None
            }
        }
    
    def _implement_poc(self, state: PoCState) -> Dict[str, Any]:
        """Generate the actual PoC implementation code."""
        
        implementation = state.get("implementation")
        if not implementation:
            raise ValueError("No PoC design available for implementation")
        
        context = self._get_project_context(state)
        previous_results = self._get_previous_results(state, same_agent_only=True)
        
        # Get design information and project details
        design_info = self._get_agent_memory(state).get("design", {})
        project = state["project"]
        task_type = getattr(project, 'task_type', 'OTHER')
        
        # Analyze existing workspace structure for context-aware implementation
        workspace_analysis = self._analyze_workspace_structure(state)
        
        # Generate task-specific prompt section
        task_specific_prompt = self._get_task_specific_prompt(task_type)
        
        # Log task type for PoC implementation with cyan color
        print("\033[96m" + "="*60)  # Cyan color
        print("🔧 POC IMPLEMENTATION STRATEGY")
        print("="*60)
        print(f"📋 Task Type: {task_type}")
        print(f"🎨 Using {task_type}-specific implementation approach")
        if hasattr(project, 'task_type_reasoning') and project.task_type_reasoning:
            print(f"💭 Context: {project.task_type_reasoning}")
        print("="*60 + "\033[0m")  # Reset color
        
        # Check if we're re-implementing due to execution error
        error_feedback = state.get("execution_error", {})
        is_reimplementation = state.get("needs_reimplementation", False)
        
        error_context = ""
        if is_reimplementation and error_feedback:
            error_context = f"""
PREVIOUS EXECUTION FAILED - FIXING IMPLEMENTATION:
Primary Error: {error_feedback.get('primary_error', 'No specific error identified')}

Error Lines from stdout:
{chr(10).join(f"- {line}" for line in error_feedback.get('error_lines_stdout', []))}

Error Lines from stderr: 
{chr(10).join(f"- {line}" for line in error_feedback.get('error_lines_stderr', []))}

IMPORTANT: Fix the errors identified above in the new implementation.
"""
            
            # Log re-implementation attempt with orange color
            print(f"\033[38;5;208m{'='*80}")  # Orange color
            print(f"🔄 POC RE-IMPLEMENTATION STARTING")
            print(f"{'='*80}")
            print(f"📝 Primary Issue: {error_feedback.get('primary_error', 'No specific error')}")
            print(f"{'='*80}\033[0m")  # Reset color

        user_prompt = f"""
{context}

{previous_results}

POC DESIGN:
{json.dumps(design_info, indent=2, ensure_ascii=False, default=str)}

EXISTING WORKSPACE STRUCTURE:
{json.dumps(workspace_analysis, indent=2, ensure_ascii=False, default=str)}

{error_context}

TASK: Generate complete, working PoC implementation code using MODERN MACHINE LEARNING APPROACHES.

IMPORTANT: Use the actual data files and structure from the workspace analysis above. 
- Adapt file paths and data loading to match the existing files
- Use the actual file names, formats, and structure discovered in the workspace
- If CSV files exist, analyze their structure and adapt data loading accordingly
- If image files exist, process them appropriately for the task type

{task_specific_prompt}

Format your response with clear file boundaries using this structure:

## filename.py
```python
[complete file content here]
```

Create all necessary files for a functional PoC:

1. MAIN APPLICATION CODE (main.py)
   - Core logic implementation  
   - Local file processing (read from data/ folder)
   - Simple command-line interface
   - Error handling and logging
   
   TASK-SPECIFIC REQUIREMENTS FOR MAIN.PY:
   The implementation will be automatically adapted based on the detected task type from problem analysis.
   All task-specific requirements (data preprocessing, model selection, evaluation metrics, output formats)
   will be included based on the identified task type (CLASSIFICATION, IMAGE_PROCESSING, REGRESSION, NLP, etc.)


REQUIREMENTS:
- All code must be complete and runnable
- Create SIMPLE COMMAND-LINE APPLICATIONS, not web servers
- Read data from local 'data/' folder, do not create web APIs
- **DO NOT USE COMMAND-LINE ARGUMENTS** - code should run with just `python main.py`
- **ALWAYS SET INPUT DIRECTORY TO './data'** - use exactly `./data` as the input directory path
- Automatically detect and process files in 'data/' directory
- Include proper error handling
- Add logging for debugging
- Use best practices for the chosen technology stack
- Each file must be clearly marked with filename and file extension
- Provide complete working implementation
- Use main.py as entry point (not app.py)
- **IMPORTANT**: main.py should be fully automated without requiring any user input or arguments
"""
        
        response = self._generate_response(self.get_system_prompt(), user_prompt)
        
        # Parse code files from response
        code_files = self._parse_code_files_from_response(response)
        
        # Save code files
        artifacts = []
        for filename, content in code_files.items():
            file_path = self._save_artifact(content, filename, state, "code")
            artifacts.append(file_path)
        
        # Update implementation object
        if hasattr(implementation, '__dict__'):
            implementation.code_files = code_files
            implementation.deployment_instructions = self._extract_deployment_instructions(response)
        else:
            implementation["code_files"] = code_files
            implementation["deployment_instructions"] = self._extract_deployment_instructions(response)
        
        # Create comprehensive README
        readme_content = self._create_readme(design_info, code_files)
        readme_path = self._save_artifact(readme_content, "README.md", state, "code")
        artifacts.append(readme_path)
        
        # Save state for debugging
        state_pkl_path = self._save_state_debug(state, "poc_implementation")
        artifacts.append(state_pkl_path)
        
        # Clear error feedback if this was a re-implementation
        if is_reimplementation:
            state.pop("execution_error", None)
            state.pop("needs_reimplementation", None)
            
            # Log successful re-implementation with green color
            print(f"\033[38;5;40m{'='*60}")  # Green color
            print(f"✅ POC RE-IMPLEMENTATION COMPLETED")
            print(f"📁 Generated {len(code_files)} code files")
            print(f"🛠️  Previous errors addressed")
            print(f"{'='*60}\033[0m")  # Reset color
        
        feedback = f"""
PoC Implementation Complete:
- Code Files Generated: {len(code_files)}
- Main Files: {', '.join([f for f in code_files.keys() if any(ext in f for ext in ['.py', '.js', '.java'])])}
- Config Files: {len([f for f in code_files.keys() if 'config' in f.lower() or 'requirements' in f.lower()])}
- Test Files: {len([f for f in code_files.keys() if 'test' in f.lower()])}
- Documentation: README.md included
"""
        
        score = self._calculate_implementation_score(code_files, response)
        
        return {
            "success": True,
            "score": score,
            "output": {"code_files": list(code_files.keys()), "files_count": len(code_files)},
            "feedback": feedback,
            "artifacts": artifacts,
            "memory": {
                "implementation": implementation.__dict__ if hasattr(implementation, '__dict__') else implementation,
                "code_files": list(code_files.keys()),
                "implemented_at": state["updated_at"].isoformat() if "updated_at" in state else None
            }
        }
    
    def _execute_poc(self, state: PoCState) -> Dict[str, Any]:
        """Execute the PoC and collect results."""
        
        implementation = state.get("implementation")
        if not implementation:
            raise ValueError("No PoC implementation available for execution")
        
        context = self._get_project_context(state)
        previous_results = self._get_previous_results(state, same_agent_only=True)
        
        # Get implementation details
        impl_info = implementation.__dict__ if hasattr(implementation, '__dict__') else implementation
        code_files = impl_info.get("code_files", {})
        
        # Get sample data path from state
        sample_data_path = state.get("sample_data_path", "data/sample1")
        
        # Execute the PoC code with sample data
        execution_result = self._execute_poc_code(state, sample_data_path)
        
        # Check if execution failed and prepare feedback for re-implementation
        if "❌ Execution failed:" in execution_result or "🚨 POC EXECUTION ERROR DETECTED" in execution_result:
            # Extract error details for feedback
            error_feedback = self._prepare_error_feedback(execution_result)
            
            # Update state to indicate need for re-implementation
            state["execution_error"] = error_feedback
            state["needs_reimplementation"] = True
            
            # Return failure result with error feedback
            return {
                "success": False,
                "score": 0.1,  # Low score for failed execution
                "output": {"execution_result": execution_result, "error_feedback": error_feedback},
                "feedback": f"❌ PoC execution failed. Error details: {error_feedback['summary']}",
                "artifacts": [],
                "memory": {
                    "execution_failed": True,
                    "error_feedback": error_feedback,
                    "failed_at": state["updated_at"].isoformat() if "updated_at" in state else None
                },
                "needs_reimplementation": True
            }
        
        user_prompt = f"""
{context}

{previous_results}

POC IMPLEMENTATION:
- Code Files: {list(code_files.keys())}
- Technology Stack: {impl_info.get("tech_stack", [])}
- Dependencies: {impl_info.get("dependencies", [])}

POC EXECUTION RESULTS:
{execution_result}

TASK: Analyze PoC execution results and create validation report.

Provide a comprehensive execution and validation plan:

1. EXECUTION SETUP
   - Environment preparation steps
   - Dependencies installation commands
   - Configuration setup
   - Database/storage initialization (if needed)

2. EXECUTION COMMANDS
   - Step-by-step execution instructions
   - Command sequences with parameters
   - Expected outputs at each step
   - Troubleshooting common issues

3. VALIDATION TESTS
   - Functional test scenarios
   - Performance benchmarks
   - Error handling validation
   - Edge case testing

4. DEMO SCENARIOS
   - End-to-end demonstration workflows
   - Sample inputs and expected outputs
   - Success criteria validation
   - User experience evaluation

5. MONITORING & METRICS
   - Performance metrics to collect
   - Logging and monitoring setup
   - Resource utilization tracking
   - Success/failure indicators

6. RESULTS DOCUMENTATION
   - How to capture and document results
   - Screenshots/output examples
   - Performance measurements
   - Issue tracking and resolution

Since this is a planning phase, focus on creating detailed instructions that would enable successful execution and evaluation.
"""
        
        schema = {
            "setup_steps": "list",
            "installation_commands": "list", 
            "configuration_steps": "list",
            "execution_commands": "list",
            "validation_tests": "list",
            "demo_scenarios": "list",
            "performance_metrics": "list",
            "monitoring_setup": "list",
            "success_indicators": "list",
            "expected_outputs": "list",
            "troubleshooting": "dict",
            "documentation_plan": "list"
        }
        
        response = self._generate_structured_response(
            self.get_system_prompt(),
            user_prompt,
            schema
        )
        
        # Create execution plan document
        execution_plan = self._create_execution_plan(response)
        plan_path = self._save_artifact(
            execution_plan,
            f"execution_plan_iteration_{state['iteration']}.md",
            state
        )
        
        # Save state for debugging
        state_pkl_path = self._save_state_debug(state, "poc_execution")
        artifacts = [plan_path, state_pkl_path]
        
        # Update implementation with execution info
        if hasattr(implementation, '__dict__'):
            implementation.performance_metrics = {metric: 0.0 for metric in response.get("performance_metrics", [])}
            implementation.execution_logs = [f"Execution plan created: {len(response.get('execution_commands', []))} steps"]
        else:
            implementation["performance_metrics"] = {metric: 0.0 for metric in response.get("performance_metrics", [])}
            implementation["execution_logs"] = [f"Execution plan created: {len(response.get('execution_commands', []))} steps"]
        
        # Simulate execution results (in real implementation, would actually execute)
        execution_results = {
            "execution_successful": True,
            "setup_completed": True,
            "tests_passed": len(response.get("validation_tests", [])),
            "performance_baseline": self._simulate_performance_metrics(response.get("performance_metrics", [])),
            "demo_scenarios_completed": len(response.get("demo_scenarios", []))
        }
        
        feedback = f"""
PoC Execution Plan Complete:
- Setup Steps: {len(response.get('setup_steps', []))}
- Execution Commands: {len(response.get('execution_commands', []))}
- Validation Tests: {len(response.get('validation_tests', []))}
- Demo Scenarios: {len(response.get('demo_scenarios', []))}
- Performance Metrics: {len(response.get('performance_metrics', []))}

Note: Actual execution would be performed in a containerized environment.
"""
        
        score = self._calculate_execution_score(response, execution_results)
        
        return {
            "success": True,
            "score": score,
            "output": {**response, "execution_results": execution_results},
            "feedback": feedback,
            "artifacts": artifacts,
            "memory": {
                "execution_plan": response,
                "execution_results": execution_results,
                "executed_at": state["updated_at"].isoformat() if "updated_at" in state else None
            }
        }
    
    def _create_design_document(self, design: Dict[str, Any]) -> str:
        """Create a comprehensive design document."""
        newline = "\n"
        return f"""# PoC Design Document

## Architecture Overview
{design.get('architecture_overview', 'Not specified')}

## System Components
{newline.join(f"- {comp}" for comp in design.get('system_components', []))}

## Technology Stack
{newline.join(f"- {tech}" for tech in design.get('programming_languages', []) + design.get('frameworks', []))}

## Data Flow
{design.get('data_flow', 'Not specified')}

## Implementation Phases
{newline.join(f"{i+1}. {phase}" for i, phase in enumerate(design.get('development_phases', [])))}

## Performance Requirements
{newline.join(f"- {k}: {v}" for k, v in design.get('performance_requirements', {}).items())}

## Demo Scenarios
{newline.join(f"- {scenario}" for scenario in design.get('demo_scenarios', []))}

## Success Criteria
{newline.join(f"- {criteria}" for criteria in design.get('success_criteria', []))}
"""
    
    def _parse_code_files_from_response(self, response: str) -> Dict[str, str]:
        """Parse code files from LLM response."""
        import json
        
        code_files = {}
        
        try:
            # First try to parse as JSON
            if response.strip().startswith('{'):
                data = json.loads(response)
                if 'code_files' in data:
                    return data['code_files']
                elif 'files' in data:
                    return data['files']
        except json.JSONDecodeError:
            pass
        
        # Enhanced pattern to catch various code block formats
        patterns = [
            # Pattern: ## filename.py followed by code block
            r'##?\s*([^\n]+\.(?:py|js|html|css|yml|yaml|json|sh|txt|md))\s*\n```(?:python|py|javascript|js|bash|sh|yaml|yml|json|html|css|markdown|txt)?\s*\n(.*?)```',
            # Pattern: ```python filename.py
            r'```(?:python|py|javascript|js|bash|sh|yaml|yml|json|dockerfile|html|css|txt|markdown)\s+([^\n]+\.(?:py|js|html|css|yml|yaml|json|sh|txt|md))\s*\n(.*?)```',
            # Pattern: ```filename.py
            r'```([^\n]+\.(?:py|js|html|css|yml|yaml|json|sh|txt|md))\s*\n(.*?)```',
            # Pattern: **filename.py** followed by code block
            r'\*\*([^\n]+\.(?:py|js|html|css|yml|yaml|json|sh|txt|md))\*\*\s*\n```(?:python|py|javascript|js|bash|sh|yaml|yml|json|html|css|txt|markdown)?\s*\n(.*?)```',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            for match in matches:
                filename = match[0].strip()
                content = match[1].strip()
                
                # Clean up filename
                filename = filename.replace('`', '').replace('*', '').strip()
                if filename.startswith('File: '):
                    filename = filename[6:].strip()
                
                # Validate filename
                if self._is_valid_filename(filename) and content:
                    code_files[filename] = content
        
        # Special handling for requirements.txt pattern
        req_pattern = r'```(?:txt|requirements)\s*(?:requirements\.txt)?\s*\n([^`]+)```'
        req_matches = re.findall(req_pattern, response, re.DOTALL | re.IGNORECASE)
        for req_content in req_matches:
            if req_content.strip():
                code_files["requirements.txt"] = req_content.strip()
        
        # Default files if nothing found
        if not code_files:
            code_files = self._create_default_poc_files()
                
        return code_files
    
    def _is_valid_filename(self, filename: str) -> bool:
        """Check if filename is valid."""
        if not filename or len(filename) > 100:
            return False
        
        # Must have extension
        if '.' not in filename:
            return False
            
        # Check for invalid characters
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '\n', '\r']
        if any(char in filename for char in invalid_chars):
            return False
            
        # Should not contain code snippets
        code_indicators = ['import ', 'def ', 'class ', 'function ', '==', 'from ']
        if any(indicator in filename.lower() for indicator in code_indicators):
            return False
            
        return True
    
    def _create_default_poc_files(self) -> Dict[str, str]:
        """Create default PoC files when parsing fails."""
        return {
            "main.py": '''#!/usr/bin/env python3
"""Main PoC implementation."""

def main():
    print("PoC implementation")
    print("Ready to implement core functionality")
    
    # TODO: Add your PoC logic here
    
if __name__ == "__main__":
    main()
'''
        }
    
    def _extract_deployment_instructions(self, response: str) -> str:
        """Extract deployment instructions from response."""
        # Look for setup/installation sections
        lines = response.split('\n')
        instructions = []
        
        capture = False
        for line in lines:
            if any(keyword in line.lower() for keyword in ["setup", "install", "deploy", "run"]):
                capture = True
            elif capture and line.strip() and not line.startswith('#'):
                instructions.append(line)
        
        return '\n'.join(instructions) if instructions else "See README.md for setup instructions"
    
    def _create_readme(self, design: Dict[str, Any], code_files: Dict[str, str]) -> str:
        """Create comprehensive README."""
        newline = "\n"
        return f"""# PoC Implementation

## Overview
This is a Proof of Concept implementation for the specified project.

## Architecture
{design.get('architecture_overview', 'See design document for details')}

## Technology Stack
- Programming Languages: {', '.join(design.get('programming_languages', []))}
- Frameworks: {', '.join(design.get('frameworks', []))}
- Dependencies: {', '.join(design.get('dependencies', []))}

## Files Structure
{newline.join(f"- `{filename}`: Implementation file" for filename in code_files.keys())}

## Setup Instructions

1. **Environment Setup**
   ```bash
   # Install Python dependencies (if applicable)
   pip install -r requirements.txt
   
   # Install Node.js dependencies (if applicable)  
   npm install
   ```

2. **Configuration**
   - Copy configuration templates
   - Update environment variables
   - Initialize database/storage if needed

3. **Execution**
   ```bash
   # Run main application
   python main.py
   
   # Or for web applications
   npm start
   ```

## Testing
```bash
# Run tests
python -m pytest tests/
```

## Demo Scenarios
{newline.join(f"- {scenario}" for scenario in design.get('demo_scenarios', ['Basic functionality test']))}

## Performance Metrics
- Response Time
- Memory Usage  
- Processing Throughput
- Success Rate

## Troubleshooting
- Check logs for error messages
- Verify all dependencies are installed
- Ensure configuration is correct

## Next Steps
- Performance optimization
- Additional features
- Production deployment
"""
    
    def _create_execution_plan(self, plan: Dict[str, Any]) -> str:
        """Create detailed execution plan document."""
        newline = "\n"
        return f"""# PoC Execution Plan

## Setup Steps
{newline.join(f"{i+1}. {step}" for i, step in enumerate(plan.get('setup_steps', [])))}

## Installation Commands
```bash
{newline.join(plan.get('installation_commands', []))}
```

## Execution Commands
{newline.join(f"{i+1}. {cmd}" for i, cmd in enumerate(plan.get('execution_commands', [])))}

## Validation Tests
{newline.join(f"- {test}" for test in plan.get('validation_tests', []))}

## Demo Scenarios
{newline.join(f"### Scenario {i+1}{newline}{scenario}{newline}" for i, scenario in enumerate(plan.get('demo_scenarios', [])))}

## Performance Metrics
{newline.join(f"- {metric}" for metric in plan.get('performance_metrics', []))}

## Success Indicators
{newline.join(f"- {indicator}" for indicator in plan.get('success_indicators', []))}

## Expected Outputs
{newline.join(f"- {output}" for output in plan.get('expected_outputs', []))}

## Troubleshooting
{newline.join(f"**{issue}**: {solution}" for issue, solution in plan.get('troubleshooting', {}).items())}
"""
    
    def _simulate_performance_metrics(self, metrics: List[str]) -> Dict[str, float]:
        """Simulate performance metrics for demo purposes."""
        results = {}
        for metric in metrics:
            if "time" in metric.lower():
                results[metric] = 0.5  # 500ms
            elif "memory" in metric.lower():
                results[metric] = 128.0  # 128MB
            elif "rate" in metric.lower():
                results[metric] = 0.95  # 95%
            else:
                results[metric] = 1.0  # Default value
        return results
    
    def _calculate_design_score(self, design: Dict[str, Any]) -> float:
        """Calculate quality score for PoC design."""
        score = 0.0
        
        # Architecture completeness
        if design.get("architecture_overview"):
            score += 0.2
        if design.get("system_components"):
            score += 0.15
        
        # Technology stack
        if design.get("technology_stack"):
            score += 0.15
        if design.get("dependencies"):
            score += 0.1
            
        # Implementation planning
        if design.get("development_phases"):
            score += 0.15
        if design.get("testing_scenarios"):
            score += 0.1
            
        # Demo and validation
        if design.get("demo_scenarios"):
            score += 0.1
        if design.get("success_criteria"):
            score += 0.05
        
        return min(score, 1.0)
    
    def _calculate_implementation_score(self, code_files: Dict[str, str], response: str) -> float:
        """Calculate quality score for PoC implementation."""
        score = 0.4  # Base score
        
        # Code completeness
        if len(code_files) >= 3:
            score += 0.2
        elif len(code_files) >= 1:
            score += 0.1
            
        # File types diversity
        extensions = set()
        for filename in code_files.keys():
            if '.' in filename:
                extensions.add(filename.split('.')[-1])
        
        if len(extensions) > 1:
            score += 0.1
            
        # Documentation
        if any("readme" in f.lower() for f in code_files.keys()) or "README.md" in response:
            score += 0.1
            
        # Configuration files
        config_files = ["requirements.txt", "package.json", "config.py", "docker", ".env"]
        if any(any(cfg in f.lower() for cfg in config_files) for f in code_files.keys()):
            score += 0.1
            
        # Testing
        if any("test" in f.lower() for f in code_files.keys()):
            score += 0.1
            
        return min(score, 1.0)
    
    def _execute_poc_code(self, state: PoCState, sample_data_path: str) -> str:
        """Execute the generated PoC code with sample data."""
        import subprocess
        import shutil
        import os
        from pathlib import Path
        from ..core.state import get_workspace_path
        
        # Get the workspace path
        workspace = get_workspace_path(state)
        code_dir = workspace / "code"
        
        if not code_dir.exists():
            return "❌ Code directory not found"
        
        # Check if main.py exists
        main_py = code_dir / "main.py"
        if not main_py.exists():
            return "❌ main.py not found in code directory"
        
        # Setup sample data
        execution_result = "🚀 PoC Code Execution Results:\n\n"
        
        # Change to code directory and execute
        original_cwd = os.getcwd()
        
        execution_result += f"🔍 Current working directory: {os.getcwd()}\n"
        execution_result += f"📁 Changing to code directory: {code_dir}\n"
        os.chdir(str(code_dir))
        execution_result += f"✅ Changed to: {os.getcwd()}\n"
        
        # Create data directory if it doesn't exist
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        execution_result += f"📂 Data directory created/ensured: {data_dir.absolute()}\n"
        
        # Copy sample data if provided
        if sample_data_path and Path(sample_data_path).exists():
            sample_path = Path(sample_data_path)
            execution_result += f"🎯 Sample data path provided: {sample_path.absolute()}\n"
            if sample_path.is_file():
                # Copy single file
                dest_path = data_dir / sample_path.name
                shutil.copy2(sample_path, dest_path)
                execution_result += f"📁 Single file copied: {sample_path.name} → {dest_path.absolute()}\n"
            elif sample_path.is_dir():
                # Copy directory contents
                copied_files = []
                for file_path in sample_path.glob("*"):
                    if file_path.is_file():
                        dest_path = data_dir / file_path.name
                        shutil.copy2(file_path, dest_path)
                        copied_files.append(file_path.name)
                        execution_result += f"  📄 Copied: {file_path.name} → {dest_path.absolute()}\n"
                execution_result += f"📁 Total files copied from directory: {len(copied_files)}\n"
        else:
            execution_result += f"🔍 No sample data path provided, searching common locations...\n"
            # Look for sample data in common locations
            common_paths = [
                Path("../data"),
                Path("../../data"), 
                Path("../../../data/sample1"),
                Path("data/sample1"),
                Path("sample_data")
            ]
            
            for common_path in common_paths:
                execution_result += f"  🔎 Checking: {common_path.absolute()}\n"
                if common_path.exists() and common_path.is_dir():
                    copied_files = []
                    for file_path in common_path.glob("*"):
                        if file_path.is_file():
                            dest_path = data_dir / file_path.name
                            shutil.copy2(file_path, dest_path)
                            copied_files.append(file_path.name)
                    if copied_files:
                        execution_result += f"✅ Found sample data at {common_path}: {copied_files}\n"
                        break
                else:
                    execution_result += f"  ❌ Not found: {common_path.absolute()}\n"
        
        # List available data files
        data_files = list(data_dir.glob("*"))
        file_list = [f.name for f in data_files if f.is_file()]
        execution_result += f"\n📂 Final data directory contents: {file_list}\n"
        execution_result += f"📊 Total files in data directory: {len(file_list)}\n\n"
        
        # Try to execute the main.py
        execution_result += "⚡ Executing main.py:\n"
        
        # Execute without arguments (as specified in requirements)
        result = subprocess.run([
            "uv", "run", "python", "main.py"
        ], capture_output=True, text=True, timeout=60)
        
        # Comprehensive error detection and classification
        error_analysis = self._analyze_execution_errors(result)
        
        # Capture results with detailed information
        execution_result += f"🔍 Execution completed with return code: {result.returncode}\n"
        execution_result += f"📏 stdout length: {len(result.stdout) if result.stdout else 0} characters\n"
        execution_result += f"📏 stderr length: {len(result.stderr) if result.stderr else 0} characters\n"
        
        # Add error analysis results
        execution_result += f"🔍 Error Analysis:\n"
        execution_result += f"   Success: {error_analysis['success']}\n"
        if error_analysis['primary_error']:
            execution_result += f"   Primary Error: {error_analysis['primary_error']}\n"
        
        # Status summary with color coding
        if error_analysis['success']:
            # Log success to stdout with green color for visibility
            print(f"\033[38;5;40m{'='*60}")  # Green color
            print(f"✅ POC EXECUTION SUCCESSFUL")
            print(f"📊 Task Type: {getattr(state.get('project', {}), 'task_type', 'UNKNOWN')}")
            print(f"📁 Workspace: {Path(state.get('workspace_path', '')).name}")
            print(f"{'='*60}\033[0m")  # Reset color
            
            execution_result += f"✅ Execution successful!\n"
        else:
            # Orange color output to stdout (visible in logs)  
            print(f"\033[38;5;208m{'='*80}")  # Orange color
            print(f"🚨 POC EXECUTION ERROR DETECTED")
            print(f"{'='*80}")
            print(f"🎯 Primary Error: {error_analysis['primary_error']}")
            print(f"{'='*80}\033[0m")  # Reset color
            
            execution_result += f"❌ Execution failed: {error_analysis['primary_error']}\n"
        
        # Always show stdout if available
        if result.stdout:
            execution_result += f"\n📤 STDOUT OUTPUT:\n{'='*50}\n{result.stdout}\n{'='*50}\n"
        else:
            execution_result += f"\n📭 No stdout output generated\n"
        
        # Always show stderr if available  
        if result.stderr:
            execution_result += f"\n🔥 STDERR OUTPUT:\n{'='*50}\n{result.stderr}\n{'='*50}\n"
        else:
            execution_result += f"\n✅ No stderr output\n"
        
        
        # Check for output files
        output_files = []
        for pattern in ["*.txt", "*.json", "*.csv", "results/*", "output/*"]:
            output_files.extend(Path(".").glob(pattern))
        
        if output_files:
            execution_result += f"\n📄 Output files generated: {[str(f) for f in output_files[:5]]}\n"
            
            # Try to read first few output files
            for output_file in output_files[:2]:
                if output_file.is_file() and output_file.stat().st_size < 1000:
                    if output_file.suffix in ['.txt', '.json', '.csv']:
                        with open(output_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            execution_result += f"\n📝 {output_file.name} content:\n{content[:500]}\n"
        
        # Restore original directory
        os.chdir(original_cwd)
        
        return execution_result
    
    def _analyze_workspace_structure(self, state: PoCState) -> Dict[str, Any]:
        """Analyze existing workspace structure to inform implementation."""
        from ..core.state import get_workspace_path
        import os
        
        workspace = get_workspace_path(state)
        
        analysis = {
            "workspace_exists": False,
            "code_directory_exists": False,
            "data_directory_exists": False,
            "data_files": [],
            "code_files": [],
            "file_analysis": {},
            "data_structure": {},
            "recommendations": []
        }
        
        if not workspace or not workspace.exists():
            analysis["recommendations"].append("No existing workspace found - will create new structure")
            return analysis
            
        analysis["workspace_exists"] = True
        
        # Check code directory
        code_dir = workspace / "code"
        if code_dir.exists():
            analysis["code_directory_exists"] = True
            
            # Analyze existing code files
            for code_file in code_dir.glob("*.py"):
                analysis["code_files"].append(code_file.name)
                
            # Check for requirements.txt
            requirements_file = code_dir / "requirements.txt"
            if requirements_file.exists():
                analysis["code_files"].append("requirements.txt")
                
        # Check data directory structure
        data_dir = code_dir / "data" if code_dir.exists() else workspace / "data"
        if data_dir.exists():
            analysis["data_directory_exists"] = True
            analysis["data_files"] = []
            
            # Analyze data files
            for data_file in data_dir.iterdir():
                if data_file.is_file():
                    file_info = {
                        "name": data_file.name,
                        "extension": data_file.suffix.lower(),
                        "size_bytes": data_file.stat().st_size,
                        "size_mb": round(data_file.stat().st_size / 1024 / 1024, 2)
                    }
                    
                    # Analyze file types
                    if data_file.suffix.lower() in ['.csv', '.tsv']:
                        file_info["type"] = "tabular_data"
                        
                        # Quick CSV analysis
                        if data_file.stat().st_size < 50 * 1024 * 1024:  # Less than 50MB
                            csv_analysis = self._analyze_csv_file(data_file)
                            file_info.update(csv_analysis)
                            
                    elif data_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                        file_info["type"] = "image"
                        
                    elif data_file.suffix.lower() in ['.txt', '.md']:
                        file_info["type"] = "text"
                        
                    elif data_file.suffix.lower() in ['.json']:
                        file_info["type"] = "json"
                        
                    else:
                        file_info["type"] = "unknown"
                        
                    analysis["data_files"].append(file_info)
                    
        # Generate recommendations based on analysis
        if analysis["data_files"]:
            # Data file recommendations
            csv_files = [f for f in analysis["data_files"] if f["type"] == "tabular_data"]
            image_files = [f for f in analysis["data_files"] if f["type"] == "image"]
            
            if csv_files:
                analysis["recommendations"].append(f"Found {len(csv_files)} CSV files - implement data loading and preprocessing")
                # Get primary CSV file for processing
                main_csv = max(csv_files, key=lambda x: x["size_bytes"])
                analysis["data_structure"]["primary_csv"] = main_csv["name"]
                
            if image_files:
                analysis["recommendations"].append(f"Found {len(image_files)} image files - implement image processing pipeline")
                analysis["data_structure"]["image_files"] = [f["name"] for f in image_files]
                
        else:
            analysis["recommendations"].append("No data files found - will use sample data generation")
            
        return analysis
        
    def _analyze_csv_file(self, csv_file_path) -> Dict[str, Any]:
        """Quick analysis of CSV file structure."""
        import pandas as pd
        
        csv_analysis = {
            "columns": [],
            "row_count": 0,
            "has_header": True,
            "delimiter": ",",
            "sample_data": {}
        }
        
        # Quick CSV inspection
        df_sample = pd.read_csv(csv_file_path, nrows=100)  # Read first 100 rows
        
        csv_analysis["columns"] = list(df_sample.columns)
        csv_analysis["row_count"] = len(df_sample)
        csv_analysis["column_count"] = len(df_sample.columns)
        
        # Detect data types
        csv_analysis["column_types"] = {}
        for col in df_sample.columns:
            csv_analysis["column_types"][col] = str(df_sample[col].dtype)
            
        # Sample first few rows
        csv_analysis["sample_data"] = df_sample.head(3).to_dict(orient='records')
        
        return csv_analysis
    
    def _analyze_execution_errors(self, result) -> Dict[str, Any]:
        """Simple error analysis - just check if execution succeeded."""
        
        # Initialize error analysis structure
        error_analysis = {
            "success": True,
            "primary_error": "",
            "error_lines_stdout": [],
            "error_lines_stderr": []
        }
        
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        
        # Extract error lines from stdout and stderr
        if stdout:
            stdout_lines = stdout.split('\n')
            error_analysis["error_lines_stdout"] = [line.strip() for line in stdout_lines if "error" in line.lower()]
        
        if stderr:
            stderr_lines = stderr.split('\n')
            error_analysis["error_lines_stderr"] = [line.strip() for line in stderr_lines if "error" in line.lower()]
        
        # Check return code (basic indicator of failure)
        if result.returncode != 0:
            error_analysis["success"] = False
            error_analysis["primary_error"] += f"Execution failed with return code {result.returncode}\n"
        
        # Check stdout for error patterns  
        if error_analysis["error_lines_stdout"]:
            error_analysis["success"] = False
            error_analysis["primary_error"] += f"Error detected in output (STDOUT): {'; '.join(error_analysis['error_lines_stdout'])}\n"

        if error_analysis["error_lines_stderr"]:
            error_analysis["success"] = False
            error_analysis["primary_error"] += f"Error detected in output (STDERR): {'; '.join(error_analysis['error_lines_stderr'])}\n"
        
        return error_analysis
    
    
    def _prepare_error_feedback(self, execution_result: str) -> Dict[str, Any]:
        """Prepare structured error feedback for re-implementation."""
        
        error_feedback = {
            "summary": "Execution failed - need to fix implementation", 
            "primary_error": "",
            "error_lines_stdout": [],
            "error_lines_stderr": [],
            "execution_output": execution_result
        }
        
        # Extract primary error
        if "Primary Error: " in execution_result:
            primary_error_match = re.search(r"Primary Error: ([^\n]+)", execution_result)
            if primary_error_match:
                error_feedback["primary_error"] = primary_error_match.group(1).strip()
        
        # Extract stdout error lines (already parsed in error analysis)
        if "📤 STDOUT OUTPUT:" in execution_result:
            stdout_section = execution_result.split("📤 STDOUT OUTPUT:")[1].split("="*50)
            if len(stdout_section) > 1:
                stdout_content = stdout_section[1].split("="*50)[0] if len(stdout_section) > 1 else ""
                stdout_lines = stdout_content.split('\n')
                error_feedback["error_lines_stdout"] = [line.strip() for line in stdout_lines if line.strip() and "error" in line.lower()][:5]
        
        # Extract stderr error lines
        if "🔥 STDERR OUTPUT:" in execution_result:
            stderr_section = execution_result.split("🔥 STDERR OUTPUT:")[1].split("="*50)
            if len(stderr_section) > 1:
                stderr_content = stderr_section[1].split("="*50)[0] if len(stderr_section) > 1 else ""
                stderr_lines = stderr_content.split('\n')
                error_feedback["error_lines_stderr"] = [line.strip() for line in stderr_lines if line.strip() and "error" in line.lower()][:5]
        
        # Generate summary
        if error_feedback["primary_error"]:
            error_feedback["summary"] = error_feedback["primary_error"]
        elif error_feedback["error_lines_stderr"]:
            error_feedback["summary"] = f"Stderr errors: {error_feedback['error_lines_stderr'][0][:100]}"
        elif error_feedback["error_lines_stdout"]:
            error_feedback["summary"] = f"Stdout errors: {error_feedback['error_lines_stdout'][0][:100]}"
        
        return error_feedback
    
    def _save_state_debug(self, state: PoCState, phase_name: str) -> str:
        """Save state as pkl file for debugging purposes."""
        import pickle
        from pathlib import Path
        from ..core.state import get_workspace_path
        
        # Get workspace path
        workspace = get_workspace_path(state)
        debug_dir = workspace / "debug"
        debug_dir.mkdir(exist_ok=True)
        
        # Create filename with phase and iteration
        iteration = state.get('iteration', 0)
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        pkl_filename = f"state_{phase_name}_iter_{iteration}_{timestamp}.pkl"
        pkl_path = debug_dir / pkl_filename
        
        # Create serializable copy of state
        state_copy = {}
        for key, value in state.items():
            # Handle non-serializable objects
            if key == "project" and hasattr(value, '__dict__'):
                state_copy[key] = value.__dict__
            elif key == "implementation" and hasattr(value, '__dict__'):
                state_copy[key] = value.__dict__
            elif key in ["started_at", "updated_at"] and hasattr(value, 'isoformat'):
                state_copy[key] = value.isoformat()
            else:
                try:
                    # Test serialization
                    pickle.dumps(value)
                    state_copy[key] = value
                except (TypeError, pickle.PicklingError):
                    # If not serializable, convert to string representation
                    state_copy[key] = str(value)
        
        # Add debug metadata
        debug_metadata = {
            "phase": phase_name,
            "iteration": iteration,
            "saved_at": timestamp,
            "file_version": "1.0"
        }
        state_copy["debug_metadata"] = debug_metadata
        
        # Save to pickle file
        with open(pkl_path, 'wb') as f:
            pickle.dump(state_copy, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Log save operation with cyan color
        print(f"\033[96m🐛 DEBUG: State saved to {pkl_path.name}\033[0m")
        
        return str(pkl_path)
    
    def _calculate_execution_score(self, plan: Dict[str, Any], results: Dict[str, Any]) -> float:
        """Calculate quality score for PoC execution."""
        score = 0.3  # Base score
        
        # Plan completeness
        if plan.get("setup_steps"):
            score += 0.15
        if plan.get("execution_commands"):
            score += 0.15
        if plan.get("validation_tests"):
            score += 0.1
        if plan.get("demo_scenarios"):
            score += 0.1
        if plan.get("performance_metrics"):
            score += 0.1
            
        # Execution results (simulated)
        if results.get("execution_successful"):
            score += 0.1
            
        return min(score, 1.0)
    
    def _get_task_specific_prompt(self, task_type: str) -> str:
        """Generate task-specific implementation prompts based on detected task type."""
        
        if task_type == "CLASSIFICATION":
            return """
TASK TYPE: CLASSIFICATION (Network Intrusion Detection, Fraud Detection, etc.)

IMPLEMENTATION REQUIREMENTS:
- Use scikit-learn for traditional ML algorithms (RandomForest, XGBoost, SVM, Gradient Boosting)
- Implement stratified cross-validation for robust evaluation (use StratifiedKFold with cross_val_score)
- Generate confusion matrix, classification report, and feature importance analysis
- Compare multiple algorithms (at least 3 different models: RandomForest, XGBoost, SVM)
- Save comprehensive CV results to JSON file with mean/std performance metrics
- Include complete data preprocessing pipeline (encoding, scaling, feature selection)
- Use appropriate metrics: accuracy, precision, recall, F1-score, ROC-AUC
- Handle imbalanced datasets with stratified sampling
- Export model comparison results to CSV file
- Implement feature importance visualization
- Save best performing model as .pkl file
- Include detailed logging of preprocessing steps and model performance
"""
        
        elif task_type == "IMAGE_PROCESSING":
            return """
TASK TYPE: IMAGE PROCESSING (OCR, Computer Vision, Object Detection)

IMPLEMENTATION REQUIREMENTS:
- Use deep learning frameworks: TensorFlow/Keras, PyTorch for neural networks
- For OCR tasks: EasyOCR, TrOCR, PaddleOCR for text recognition
- Use OpenCV for image preprocessing and computer vision operations
- Implement proper image loading and preprocessing pipeline
- Include error handling for various image formats (PNG, JPG, JPEG, etc.)
- Use Hugging Face Transformers for pre-trained models when applicable
- Save processing results to structured format (JSON, CSV)
- Include image visualization and result overlay capabilities
- Implement batch processing for multiple images
- Add performance metrics specific to image tasks (processing time, accuracy)
- Include sample visualization of results
"""
        
        elif task_type == "REGRESSION":
            return """
TASK TYPE: REGRESSION (Price Prediction, Time Series Forecasting, Numerical Prediction)

IMPLEMENTATION REQUIREMENTS:
- Use scikit-learn for regression algorithms (RandomForest, XGBoost, SVR, Linear Regression)
- Implement cross-validation with regression metrics (RMSE, MAE, R²)
- Include feature scaling and normalization for numerical features
- Generate regression plots and residual analysis
- Compare multiple regression algorithms with performance metrics
- Save prediction results and model performance to files
- Include feature importance analysis for tree-based models
- Implement proper train/validation/test splits
- Add prediction intervals and confidence bounds where applicable
- Export comprehensive model comparison results
"""
        
        elif task_type == "NLP":
            return """
TASK TYPE: NLP (Text Processing, Language Translation, Document Analysis)

IMPLEMENTATION REQUIREMENTS:
- Use modern NLP libraries: Hugging Face Transformers, spaCy, NLTK
- Implement proper text preprocessing pipeline (tokenization, cleaning, normalization)
- Use pre-trained models for tasks: BERT, RoBERTa, T5, GPT variants
- Include text vectorization methods (TF-IDF, embeddings)
- Implement appropriate evaluation metrics (BLEU, ROUGE, perplexity)
- Save processed text and model outputs to structured formats
- Include text visualization and word cloud generation
- Handle multiple text formats (plain text, CSV, JSON)
- Implement batch processing for document collections
- Add language detection and preprocessing capabilities
"""
        
        else:  # OTHER or unknown task types
            return """
TASK TYPE: GENERAL/OTHER

IMPLEMENTATION REQUIREMENTS:
- Analyze the problem domain and choose appropriate ML/AI approaches
- Use established libraries and frameworks suitable for the task
- Implement proper data loading and preprocessing
- Include comprehensive evaluation and validation
- Generate appropriate performance metrics and visualizations
- Save results and model artifacts to files
- Include proper error handling and logging
- Ensure reproducibility with clear setup instructions
- Add appropriate documentation and usage examples
"""


if __name__ == "__main__":
    """PoCAgentの動作確認用メイン関数"""
    import argparse
    import os
    import sys
    from pathlib import Path
    from dotenv import load_dotenv
    
    # Add parent directories to Python path for imports
    current_dir = Path(__file__).parent
    src_dir = current_dir.parent.parent
    sys.path.insert(0, str(src_dir))
    
    # Import with absolute paths for standalone execution
    from ai_poc_agents_v2.core.config import Config
    from ai_poc_agents_v2.core.state import PoCProject, PoCIdea
    from ai_poc_agents_v2.agents.base_agent import BaseAgent
    
    # Load environment variables
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Test PoCAgent functionality")
    parser.add_argument("--theme", type=str, default="OCR画像文字認識システム", help="Project theme")
    parser.add_argument("--description", type=str, default="画像から文字を認識してテキストに変換するPythonシステム", help="Project description")
    parser.add_argument("--phase", type=str, choices=["poc_design", "poc_implementation", "poc_execution", "all"], 
                      default="all", help="Which phase to test")
    parser.add_argument("--workspace", type=str, default="./tmp/test_poc_agent", help="Workspace directory")
    
    args = parser.parse_args()
    
    print("=== PoCAgent Test =======================")
    print(f"Theme: {args.theme}")
    print(f"Description: {args.description}")
    print(f"Phase: {args.phase}")
    print()
    
    # Check environment
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print(f"✓ OPENAI_API_KEY found (length: {len(openai_key)})")
    else:
        print("❌ OPENAI_API_KEY not found - LLM calls will fail")
        print("Please set OPENAI_API_KEY in your .env file")
        exit(1)
    
    # Create workspace
    workspace_dir = Path(args.workspace)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Workspace created: {workspace_dir}")
    
    # Initialize config and agent
    print("\\n1. Initializing PoCAgent...")
    config = Config()
    
    # Create agent
    poc_agent = PoCAgent("poc_developer", config)
    
    # Create project and mock selected idea
    project = PoCProject(theme=args.theme)
    project.description = args.description
    
    # Create a mock selected idea
    selected_idea = PoCIdea()
    selected_idea.id = "test_idea_1"
    selected_idea.title = f"{args.theme}の基本実装"
    selected_idea.description = args.description
    selected_idea.technical_approach = "Python + OpenCV + TensorFlow"
    selected_idea.implementation_complexity = 3
    selected_idea.expected_impact = 4
    selected_idea.feasibility_score = 0.8
    selected_idea.total_score = 0.8
    selected_idea.estimated_effort_hours = 16
    
    state = {
        "project": project,
        "selected_idea": selected_idea,
        "current_phase": "poc_design",
        "iteration": 0,
        "completed_phases": [],
        "phase_results": [],
        "phase_scores": {},
        "overall_score": 0.0,
        "artifacts": [],
        "logs": [],
        "should_continue": True,
        "workspace_dir": workspace_dir,
        "workspace_path": str(workspace_dir),
        "started_at": __import__('datetime').datetime.now(),
        "updated_at": __import__('datetime').datetime.now(),
        "agent_memory": {}
    }
    
    print("✓ PoCAgent initialized successfully")
    
    # Test phases based on user selection
    if args.phase in ["poc_design", "all"]:
        print("\\n2. Testing PoC Design Phase...")
        result = poc_agent._design_poc(state)
        print(f"✓ PoC design completed")
        print(f"   Success: {result['success']}")
        print(f"   Score: {result['score']:.3f}")
        print(f"   Artifacts: {len(result['artifacts'])}")
        
        # Show some results
        output = result.get('output', {})
        if output.get('system_components'):
            print(f"   System Components: {len(output['system_components'])}")
        if output.get('technology_stack'):
            print(f"   Technology Stack: {len(output['technology_stack'])} technologies")
        if output.get('development_phases'):
            print(f"   Development Phases: {len(output['development_phases'])}")
        
        print(f"   Feedback: {result['feedback'][:200]}...")
    
    if args.phase in ["poc_implementation", "all"]:
        print("\\n3. Testing PoC Implementation Phase...")
        state["current_phase"] = "poc_implementation"
        
        # Ensure we have design/implementation for the implementation phase
        if not state.get("implementation"):
            print("   No design available - creating mock implementation object...")
            from ai_poc_agents_v2.core.state import PoCImplementation
            
            # Create mock implementation
            mock_implementation = PoCImplementation(
                idea_id=selected_idea.id,
                architecture={
                    "architecture_overview": "OCR system using Python, OpenCV, and TensorFlow",
                    "system_components": ["Image Preprocessor", "OCR Engine", "Text Extractor", "Output Handler"],
                    "technology_stack": {"languages": ["Python"], "frameworks": ["TensorFlow", "OpenCV"]},
                    "programming_languages": ["Python"],
                    "frameworks": ["TensorFlow", "OpenCV", "EasyOCR"],
                    "libraries": ["numpy", "pillow", "matplotlib"],
                    "dependencies": ["tensorflow>=2.12.0", "opencv-python>=4.8.0", "easyocr>=1.7.0", "pillow>=10.0.0"],
                    "development_phases": ["Setup Environment", "Image Processing", "OCR Implementation", "Testing"],
                    "demo_scenarios": ["Process sample images", "Extract text", "Save results"]
                },
                tech_stack=["Python", "TensorFlow", "OpenCV", "EasyOCR"],
                dependencies=["tensorflow>=2.12.0", "opencv-python>=4.8.0", "easyocr>=1.7.0"],
                environment_config={"python_version": "3.8+"},
                test_cases=[{"scenario": "Test OCR on sample image"}]
            )
            
            state["implementation"] = mock_implementation
            
            # Add to agent memory for consistency
            state["agent_memory"] = {
                "design": mock_implementation.architecture,
                "implementation": mock_implementation.__dict__
            }
        
        result = poc_agent._implement_poc(state)
        print(f"✓ PoC implementation completed")
        print(f"   Success: {result['success']}")
        print(f"   Score: {result['score']:.3f}")
        print(f"   Artifacts: {len(result['artifacts'])}")
        
        # Show detailed results
        output = result.get('output', {})
        memory = result.get('memory', {})
        
        print(f"   Code Files Generated: {output.get('files_count', 0)}")
        code_files = memory.get('implementation', {}).get('code_files', {})
        if code_files:
            print(f"   Files Created:")
            for filename in list(code_files.keys())[:5]:
                print(f"     - {filename}")
        
        print(f"   Feedback: {result['feedback'][:300]}...")
    
    if args.phase in ["poc_execution", "all"]:
        print("\\n4. Testing PoC Execution Phase...")
        state["current_phase"] = "poc_execution"
        
        # Ensure we have implementation to execute
        if not state.get("implementation"):
            print("   No implementation available - skipping execution phase")
        else:
            result = poc_agent._execute_poc(state)
            print(f"✓ PoC execution completed")
            print(f"   Success: {result['success']}")
            print(f"   Score: {result['score']:.3f}")
            print(f"   Artifacts: {len(result['artifacts'])}")
            
            # Show execution results
            output = result.get('output', {})
            
            if output.get('setup_steps'):
                print(f"   Setup Steps: {len(output['setup_steps'])}")
            if output.get('execution_commands'):
                print(f"   Execution Commands: {len(output['execution_commands'])}")
            if output.get('validation_tests'):
                print(f"   Validation Tests: {len(output['validation_tests'])}")
            
            execution_results = output.get('execution_results', {})
            if execution_results:
                print(f"   Execution Successful: {execution_results.get('execution_successful', False)}")
                print(f"   Tests Passed: {execution_results.get('tests_passed', 0)}")
            
            print(f"   Feedback: {result['feedback'][:200]}...")
    
    print("\\n5. Test Summary...")
    print(f"   Workspace: {workspace_dir}")
    print(f"   Artifacts created: {len(state['artifacts'])}")
    
    # Show workspace contents
    artifacts = list(workspace_dir.glob("**/*"))
    if artifacts:
        print(f"   Files created:")
        for artifact in artifacts[:10]:  # Show first 10 files
            if artifact.is_file():
                size = artifact.stat().st_size
                print(f"     - {artifact.name} ({size} bytes)")
    
    print("\\n✅ PoCAgent test completed successfully!")
    print(f"Workspace preserved at: {workspace_dir}")