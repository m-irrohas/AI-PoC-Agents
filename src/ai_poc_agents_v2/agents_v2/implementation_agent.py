"""Implementation Agent for AI-PoC-Agents-v2."""

from typing import Dict, Any, List, Optional
import json
from datetime import datetime
from pathlib import Path

from ..agents.base_agent import BaseAgent
from ..core.state import PoCState, PoCImplementation


class ImplementationAgent(BaseAgent):
    """Agent responsible for code generation using sample code references and design specifications."""
    
    def __init__(self, config):
        """Initialize ImplementationAgent."""
        super().__init__("poc_implementer", config)
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        return """
You are an expert Implementation Agent for PoC development and code generation.

Your responsibilities:
- CODE GENERATION: Create actual Python code based on design specifications
- SAMPLE CODE INTEGRATION: Leverage sample code patterns and examples from research
- ARCHITECTURE IMPLEMENTATION: Transform design specifications into working code
- FILE STRUCTURE: Generate appropriate file structure and organization
- CONFIGURATION SETUP: Create configuration files, requirements, and setup scripts
- ERROR HANDLING: Implement robust error handling and logging
- DOCUMENTATION: Generate inline code documentation and README files

Key principles:
- Generate production-ready, well-structured Python code
- Follow best practices for code organization and modularity
- Integrate patterns and approaches from sample code references
- Implement the full architecture as specified in the design
- Create complete, runnable code that fulfills functional requirements
- Include proper error handling, logging, and configuration
- Generate necessary setup and deployment files
- Ensure code is maintainable, readable, and well-documented

CRITICAL: Never use try-except blocks in implementation. Use defensive programming with proper pre-condition checks and explicit error handling through return values and status codes.

Always generate complete, implementable code that can be executed successfully.
"""
    
    def execute_phase(self, state: PoCState) -> Dict[str, Any]:
        """Execute code implementation based on design specifications."""
        
        project = state["project"]
        context = self._get_project_context(state)
        previous_results = self._get_previous_results(state)
        
        # Get implementation design from previous phase
        implementation = state.get("implementation")
        if not implementation:
            return {
                "success": False,
                "score": 0.0,
                "output": {},
                "feedback": "No implementation design found. Please run PoC design first.",
                "artifacts": [],
                "memory": {}
            }
        
        # Check for execution feedback (improvement loop)
        execution_feedback = state.get("execution_feedback")
        is_feedback_iteration = execution_feedback is not None
        
        # Get inputs from previous agents
        design_memory = state["agent_memory"].get("poc_designer", {})
        search_results = state["agent_memory"].get("search_problem", {}).get("search_results", {})
        
        # Prepare implementation context
        implementation_context = self._prepare_implementation_context(implementation, design_memory)
        sample_code_context = self._prepare_sample_code_context(search_results)
        sample_data_context = self._prepare_sample_data_context(state)
        
        # Prepare feedback information if this is a feedback iteration
        feedback_context = ""
        if is_feedback_iteration:
            feedback_context = f"""
EXECUTION FEEDBACK - IMPROVEMENT ITERATION #{execution_feedback.get('feedback_iteration', 1)}:
The previous implementation had execution issues that need to be fixed:

FIX INSTRUCTIONS:
{chr(10).join('- ' + instruction for instruction in execution_feedback.get('fix_instructions', []))}

ERROR ANALYSIS:
{json.dumps(execution_feedback.get('error_analysis', {}), indent=2)}

CRITICAL: This is a code improvement iteration. Fix the specific issues identified above while maintaining the core functionality.
"""

        user_prompt = f"""
{context}

{previous_results}

{feedback_context}

IMPLEMENTATION DESIGN SPECIFICATION:
{json.dumps(self._implementation_to_dict(implementation), indent=2, ensure_ascii=False)}

{implementation_context}

{sample_code_context}

{sample_data_context}

TASK: {'IMPROVE AND FIX' if is_feedback_iteration else 'Generate complete, working'} Python code that implements the PoC as a COMMAND-LINE APPLICATION.

CRITICAL: Create a CONSOLIDATED MAIN.PY that can be executed with 'python main.py' or 'uv run python main.py'.

REQUIREMENTS:
- Single main.py file with hardcoded data path
- No try-except blocks (defensive programming)  
- No command line arguments required
- Executable with 'python main.py' (no arguments)

Generate only the Python code content for main.py.
"""
        print(user_prompt)
        print(len(user_prompt))
        
        # Get raw Python code directly from LLM without JSON structure
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": user_prompt}
        ]
        
        raw_response = self.llm.invoke(messages)
        python_code = raw_response.content.strip()
        
        # Remove markdown code blocks if present
        if python_code.startswith("```python"):
            python_code = python_code[9:]
        if python_code.startswith("```"):
            python_code = python_code[3:]
        if python_code.endswith("```"):
            python_code = python_code[:-3]
        
        # Create response structure with just main.py
        response = {
            "code_files": {
                "main.py": python_code.strip()
            }
        }
        
        # Update implementation with generated code
        implementation.code_files = response.get("code_files", {})
        
        # Create project structure and save files
        artifacts = self._save_code_files(response, state)
        
        # Update implementation design in state
        state["implementation"] = implementation
        
        # Save implementation summary
        implementation_data = self._implementation_to_dict(implementation)
        summary_path = self._save_json_artifact(
            {
                "implementation_summary": implementation_data,
                "code_generation_context": {
                    "files_generated": len(response.get("code_files", {})),
                    "total_lines": sum(len(content.split('\n')) for content in response.get("code_files", {}).values()),
                    "dependencies_added": 0,
                    "setup_complexity": 1
                }
            },
            f"implementation_iteration_{state['iteration']}.json",
            state
        )
        artifacts.append(summary_path)
        
        # Create simple README
        readme_content = f"# {project.theme}\n\nGenerated code files:\n"
        for filename in response.get("code_files", {}).keys():
            readme_content += f"- {filename}\n"
        readme_path = self._save_artifact(
            readme_content,
            f"README_iteration_{state['iteration']}.md",
            state
        )
        artifacts.append(readme_path)
        
        # Log implementation summary
        self._log_implementation_summary(response, implementation)
        
        # Calculate values
        code_files = response.get('code_files', {})
        total_lines = sum(len(content.split('\n')) for content in code_files.values())
        
        feedback = f"Code files generated: {len(code_files)}, Total lines: {total_lines}"
        
        score = self._calculate_implementation_score(response, implementation, search_results)
        
        return {
            "success": True,
            "score": score,
            "output": {
                "code_files": list(response.get("code_files", {}).keys()),
                "files_count": len(response.get("code_files", {}))
            },
            "feedback": feedback,
            "artifacts": artifacts,
            "memory": {
                "generated_code_files": list(response.get("code_files", {}).keys()),
                "total_lines_generated": total_lines,
                "dependencies_used": response.get("dependencies", []),
                "implementation_patterns": self._extract_implementation_patterns(response),
                "is_feedback_iteration": is_feedback_iteration,
                "feedback_iteration_number": execution_feedback.get('feedback_iteration', 0) if execution_feedback else 0,
                "fixed_issues": execution_feedback.get('fix_instructions', []) if execution_feedback else [],
                "implemented_at": datetime.now().isoformat()
            }
        }
    
    def _prepare_implementation_context(self, implementation: PoCImplementation, design_memory: Dict[str, Any]) -> str:
        """Prepare simplified implementation context from design specifications."""
        tech_stack = ', '.join(implementation.tech_stack) if implementation.tech_stack else 'Not specified'
        dependencies = ', '.join(implementation.dependencies) if implementation.dependencies else 'None'
        test_count = len(implementation.test_cases) if implementation.test_cases else 0
        
        context = f"""
IMPLEMENTATION CONTEXT:
- Technology Stack: {tech_stack}
- Dependencies: {dependencies}
- Test Cases: {test_count} defined
- Architecture: {len(implementation.architecture)} components
"""
        return context
    
    def _prepare_sample_code_context(self, search_results: Dict[str, Any]) -> str:
        """Prepare sample code context for implementation guidance."""
        sample_code = search_results.get('sample_code_collection', [])
        if not sample_code:
            return "No sample code available for reference."
        
        context = f"""
SAMPLE CODE PATTERNS FOR IMPLEMENTATION ({len(sample_code)} references):

Code Examples and Patterns:
"""
        
        # Show detailed sample code for implementation
        for i, snippet in enumerate(sample_code[:2], 1):  # Show top 2 for implementation
            context += f"""
Pattern {i} - {snippet.get('language', 'Python')}:
Source: {snippet.get('article_title', snippet.get('file_path', 'Unknown'))}
Context: {snippet.get('context', 'Code pattern')}
Code:
```{snippet.get('language', 'python')}
{snippet.get('code_snippet', '')}
```

Implementation Notes:
{snippet.get('explanation', 'Use this pattern for similar functionality')}

---
"""
        
        return context
    
    def _prepare_sample_data_context(self, state: PoCState) -> str:
        """Prepare sample data context for implementation guidance."""
        sample_data_info = state.get("sample_data_info", {})
        
        if not sample_data_info or not sample_data_info.get("file_exists"):
            return "No sample data available."
        
        # If it's a directory with multiple files, use train data
        if sample_data_info.get("is_directory"):
            train_data = sample_data_info.get("train_data")
            if not train_data:
                return "No training data file found in directory."
            
            context = f"""
SAMPLE DATA DIRECTORY AVAILABLE:
- Source Directory: {sample_data_info.get('source_path', '')}
- Files Count: {sample_data_info.get('file_count', 0)}

TRAINING DATA FOR IMPLEMENTATION:
- File Path: {train_data.get('file_path', '')}
- File Name: {train_data.get('file_name', '')}
- File Size: {train_data.get('file_size_mb', 0)} MB
"""
            
            if train_data.get('file_type') == 'csv':
                context += f"""
Dataset Structure:
- Total Columns: {train_data.get('num_columns', 0)}
- Total Rows: {train_data.get('total_rows', 'unknown')}
- Column Names: {', '.join(train_data.get('column_names', [])[:10])}
- Target Column Candidates: {', '.join(train_data.get('target_column_candidates', []))}

CRITICAL IMPLEMENTATION REQUIREMENTS:
1. Your main.py MUST use the training data file path directly in the code
2. Hardcode the exact file path: {train_data.get('file_path', '')}
3. Do NOT use argparse or command line arguments
4. The file will be executed as: python main.py (without arguments)

Required data loading setup:
```python
# Hardcode the training data file path directly in main.py
DATA_FILE_PATH = r"{train_data.get('file_path', '')}"

def main():
    # Load training data using the hardcoded path
    data = pd.read_csv(DATA_FILE_PATH)
    # Your implementation here...

if __name__ == "__main__":
    main()
```

The training dataset contains:
- {train_data.get('num_columns', 0)} columns including potential target: {train_data.get('target_column_candidates', ['unknown'])[0] if train_data.get('target_column_candidates') else 'unknown'}
- Suitable for classification tasks
"""
        
        else:
            # Single file case
            context = f"""
SAMPLE DATA AVAILABLE FOR TESTING:

File Information:
- File Path: {sample_data_info.get('file_path', '')}
- File Name: {sample_data_info.get('file_name', '')}
- File Type: {sample_data_info.get('file_type', 'unknown')}
- File Size: {sample_data_info.get('file_size_mb', 0)} MB
"""
            
            if sample_data_info.get('file_type') == 'csv':
                context += f"""
Dataset Structure:
- Total Columns: {sample_data_info.get('num_columns', 0)}
- Total Rows: {sample_data_info.get('total_rows', 'unknown')}
- Column Names: {', '.join(sample_data_info.get('column_names', [])[:10])}
- Target Column Candidates: {', '.join(sample_data_info.get('target_column_candidates', []))}

CRITICAL IMPLEMENTATION REQUIREMENTS:
1. Your main.py MUST use the data file path directly in the code
2. Hardcode the exact file path: {sample_data_info.get('file_path', '')}
3. Do NOT use argparse or command line arguments
4. The file will be executed as: python main.py (without arguments)

Required data loading setup:
```python
# Hardcode the data file path directly in main.py
DATA_FILE_PATH = r"{sample_data_info.get('file_path', '')}"

def main():
    # Load data using the hardcoded path
    data = pd.read_csv(DATA_FILE_PATH)
    # Your implementation here...

if __name__ == "__main__":
    main()
```

The dataset contains:
- {sample_data_info.get('num_columns', 0)} columns including potential target: {sample_data_info.get('target_column_candidates', ['unknown'])[0] if sample_data_info.get('target_column_candidates') else 'unknown'}
- Suitable for classification tasks
"""
        
        return context
    
    def _implementation_to_dict(self, implementation: PoCImplementation) -> Dict[str, Any]:
        """Convert PoCImplementation to dictionary."""
        return {
            "idea_id": implementation.idea_id,
            "architecture": implementation.architecture,
            "tech_stack": implementation.tech_stack,
            "environment_config": implementation.environment_config,
            "test_cases": implementation.test_cases,
            "deployment_instructions": implementation.deployment_instructions,
            "dependencies": implementation.dependencies,
            "performance_metrics": implementation.performance_metrics,
            "code_files_count": len(implementation.code_files),
            "generated_files": list(implementation.code_files.keys())
        }
    
    def _save_code_files(self, response: Dict[str, Any], state: PoCState) -> List[str]:
        """Save generated code files to workspace."""
        artifacts = []
        code_files = response.get("code_files", {})
        
        # Normalize LLM output structure - handle nested dict format
        normalized_files = {}
        for filename, content in code_files.items():
            print(content)
            if isinstance(content, dict):
                # LLM returned nested structure, take first value
                if content:
                    normalized_files[filename] = next(iter(content.values()))
                else:
                    continue
            elif isinstance(content, str):
                # LLM returned direct string content (expected format)
                normalized_files[filename] = content
            else:
                # Skip invalid content
                continue
        
        # Create code directory
        workspace_path = Path(state["workspace_path"])
        code_dir = workspace_path / "generated_code" / f"iteration_{state['iteration']}"
        code_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each code file
        for filename, content in normalized_files.items():
            # Handle nested directories
            file_path = code_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file content
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            artifacts.append(str(file_path))
        
        # Save project structure file
        if response.get("project_structure"):
            structure_content = self._format_project_structure(response["project_structure"])
            structure_path = code_dir / "project_structure.txt"
            with open(structure_path, "w", encoding="utf-8") as f:
                f.write(structure_content)
            artifacts.append(str(structure_path))
        
        # Save requirements.txt if dependencies exist
        dependencies = response.get("dependencies", [])
        if dependencies:
            req_path = code_dir / "requirements.txt"
            with open(req_path, "w", encoding="utf-8") as f:
                for dep in dependencies:
                    f.write(f"{dep}\n")
            artifacts.append(str(req_path))
        
        # Save setup instructions
        setup_instructions = response.get("setup_instructions", "")
        if setup_instructions:
            setup_path = code_dir / "SETUP.md"
            with open(setup_path, "w", encoding="utf-8") as f:
                f.write(f"# Setup Instructions\n\n{setup_instructions}")
            artifacts.append(str(setup_path))
        
        return artifacts
    
    def _format_project_structure(self, structure: List[str]) -> str:
        """Format project structure as tree."""
        formatted = "Project Structure:\n"
        for item in structure:
            formatted += f"  {item}\n"
        return formatted
    
    def _create_readme(self, response: Dict[str, Any], implementation: PoCImplementation, project) -> str:
        """Create comprehensive README for the implementation."""
        readme = f"""# {project.theme} - PoC Implementation

## Overview
{project.description}

**Domain**: {project.domain}
**Timeline**: {project.timeline_days} days

## Architecture

### Technology Stack
{', '.join(implementation.tech_stack)}

### Dependencies
```
{chr(10).join(response.get('dependencies', []))}
```

## Project Structure
```
{self._format_project_structure(response.get('project_structure', []))}
```

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation
```bash
# Clone or download the project
# Navigate to project directory

# Install dependencies
pip install -r requirements.txt

# Set up environment (if .env file exists)
cp .env.example .env
# Edit .env with your configuration
```

### Configuration
{response.get('configuration_notes', 'No specific configuration required.')}

## Usage

### Entry Points
"""
        
        entry_points = response.get('entry_points', {})
        for name, description in entry_points.items():
            readme += f"""
#### {name}
{description}
```bash
python {name}
```
"""
        
        readme += f"""

## Testing

The implementation includes test files for validation:
"""
        
        test_files = [f for f in response.get('code_files', {}).keys() if 'test' in f.lower()]
        for test_file in test_files:
            readme += f"- `{test_file}`\n"
        
        readme += f"""

Run tests with:
```bash
python -m pytest  # if using pytest
# or
python -m unittest discover  # if using unittest
```

## Implementation Details

### Functional Requirements
{json.dumps(implementation.environment_config.get('functional_requirements', {}), indent=2)}

### Architecture Components
{json.dumps(implementation.architecture.get('components', []), indent=2)}

### Performance Metrics
{json.dumps(implementation.performance_metrics, indent=2)}

## Deployment

{implementation.deployment_instructions}

## Files Generated

"""
        
        file_descriptions = response.get('file_descriptions', {})
        for filename, description in file_descriptions.items():
            readme += f"- **{filename}**: {description}\n"
        
        readme += f"""

## Troubleshooting

### Common Issues
1. **Missing Dependencies**: Run `pip install -r requirements.txt`
2. **Configuration Errors**: Check your .env file and configuration settings
3. **Import Errors**: Ensure you're running from the project root directory

### Logging
The implementation includes comprehensive logging. Check log files or console output for debugging information.

## Next Steps

1. Run the implementation and validate functionality
2. Execute test cases to verify correctness
3. Monitor performance metrics
4. Consider scaling and optimization opportunities

---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Implementation Agent - AI-PoC-Agents-v2*
"""
        
        return readme
    
    def _calculate_structure_depth(self, structure: List[str]) -> int:
        """Calculate the depth of the project structure."""
        if not structure:
            return 0
        
        max_depth = 0
        for item in structure:
            depth = item.count('/') + item.count('\\')
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _count_sample_code_usage(self, response: Dict[str, Any], search_results: Dict[str, Any]) -> int:
        """Count how many sample code patterns were used."""
        sample_code = search_results.get('sample_code_collection', [])
        if not sample_code:
            return 0
        
        usage_count = 0
        code_files = response.get('code_files', {})
        
        # Simple pattern matching to detect usage
        for snippet in sample_code[:10]:  # Check top 10 snippets
            snippet_code = snippet.get('code_snippet', '').lower()
            
            # Extract key patterns (function names, class names, imports)
            import re
            patterns = re.findall(r'def\s+(\w+)|class\s+(\w+)|from\s+(\w+)|import\s+(\w+)', snippet_code)
            
            for code_content in code_files.values():
                content_lower = code_content.lower()
                for pattern_group in patterns:
                    for pattern in pattern_group:
                        if pattern and pattern in content_lower:
                            usage_count += 1
                            break
                    break
        
        return min(usage_count, len(sample_code))
    
    def _extract_implementation_patterns(self, response: Dict[str, Any]) -> List[str]:
        """Extract implementation patterns used in the code."""
        patterns = []
        code_files = response.get('code_files', {})
        
        for filename, content in code_files.items():
            if '.py' in filename:
                # Detect common patterns
                if 'class' in content:
                    patterns.append('object_oriented_design')
                if 'def __init__' in content:
                    patterns.append('constructor_pattern')
                if 'import' in content:
                    patterns.append('modular_design')
                if 'logging' in content:
                    patterns.append('logging_integration')
                if 'config' in content.lower():
                    patterns.append('configuration_management')
                if 'test' in filename.lower():
                    patterns.append('test_driven_development')
        
        return list(set(patterns))  # Remove duplicates
    
    def _log_implementation_summary(self, response: Dict[str, Any], implementation: PoCImplementation) -> None:
        """Log implementation summary with color formatting."""
        print("\033[92m" + "="*60)  # Green color
        print("💻 CODE IMPLEMENTATION RESULTS")
        print("="*60)
        print(f"📁 Files Generated: {len(response.get('code_files', {}))}")
        total_lines_log = sum(len(content.split('\n')) for content in response.get('code_files', {}).values())
        print(f"📏 Total Lines: {total_lines_log}")
        print(f"📦 Dependencies: {len(response.get('dependencies', []))}")
        print(f"🚀 Entry Points: {len(response.get('entry_points', {}))}")
        print(f"🏗️ Structure Depth: {self._calculate_structure_depth(response.get('project_structure', []))}")
        print(f"🔧 Tech Stack: {len(implementation.tech_stack)} technologies")
        
        print(f"\n📄 Generated Files:")
        for filename in list(response.get('code_files', {}).keys())[:8]:  # Show first 8
            file_content = response.get('code_files', {}).get(filename, '')
            lines = len(file_content.split('\n'))
            print(f"   • {filename} ({lines} lines)")
        
        if len(response.get('code_files', {})) > 8:
            print(f"   ... and {len(response.get('code_files', {})) - 8} more files")
        
        print("="*60 + "\033[0m")  # Reset color
    
    def _calculate_implementation_score(self, response: Dict[str, Any], implementation: PoCImplementation, search_results: Dict[str, Any]) -> float:
        """Calculate quality score for the implementation."""
        score = 0.0
        
        # Base score for having generated code
        code_files = response.get('code_files', {})
        if code_files:
            score += 0.3
        
        # Score for number of files (shows completeness)
        file_count = len(code_files)
        if file_count >= 5:
            score += 0.15
        elif file_count >= 3:
            score += 0.1
        
        # Score for code quality indicators
        total_lines = sum(len(content.split('\n')) for content in code_files.values())
        if total_lines >= 200:
            score += 0.1
        
        # Score for dependencies and setup
        if response.get('dependencies'):
            score += 0.1
        
        # Score for project structure
        structure = response.get('project_structure', [])
        if len(structure) >= 5:
            score += 0.1
        
        # Score for entry points
        if response.get('entry_points'):
            score += 0.1
        
        # Score for sample code integration
        sample_usage = self._count_sample_code_usage(response, search_results)
        if sample_usage > 0:
            score += min(sample_usage * 0.05, 0.15)
        
        return min(score, 1.0)