"""Execute Agent for AI-PoC-Agents-v2."""

import subprocess
import sys
import os
from typing import Dict, Any, List, Tuple, Optional
import json
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from ..agents.base_agent import BaseAgent
from ..core.state import PoCState, PoCImplementation, EvaluationResult


class ExecuteAgent(BaseAgent):
    """Agent responsible for code execution, evaluation, and results capture."""
    
    def __init__(self, config):
        """Initialize ExecuteAgent."""
        super().__init__("poc_executor", config)
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        return """
You are an expert Execute Agent for PoC code execution and evaluation.

Your responsibilities:
- CODE EXECUTION: Run the generated PoC code in a controlled environment
- ERROR HANDLING: Capture and analyze execution errors and issues
- PERFORMANCE MEASUREMENT: Measure execution time, memory usage, and other metrics
- OUTPUT ANALYSIS: Analyze program outputs and results
- ENVIRONMENT SETUP: Prepare and manage execution environments
- VALIDATION: Validate code functionality against requirements
- LOGGING: Comprehensive logging of execution processes and results

Key principles:
- Execute code safely in isolated environments
- Capture all execution outputs, errors, and performance metrics
- Provide detailed analysis of execution results
- Identify and categorize different types of issues
- Measure quantitative performance indicators
- Validate functionality against design specifications
- Generate actionable feedback for improvement
- Document execution context and conditions

CRITICAL: Handle all execution scenarios gracefully, including failures, timeouts, and unexpected behaviors. Provide comprehensive analysis regardless of execution outcome.

Always execute code thoroughly and provide detailed evaluation of results.
"""
    
    def execute_phase(self, state: PoCState) -> Dict[str, Any]:
        """Execute the generated PoC code and evaluate results."""
        
        project = state["project"]
        context = self._get_project_context(state)
        previous_results = self._get_previous_results(state)
        
        # Get implementation from previous phase
        implementation = state.get("implementation")
        if not implementation or not implementation.code_files:
            return {
                "success": False,
                "score": 0.0,
                "output": {},
                "feedback": "No code implementation found. Please run implementation first.",
                "artifacts": [],
                "memory": {}
            }
        
        # Get inputs from previous agents
        implementation_memory = state["agent_memory"].get("poc_implementer", {})
        
        # Create execution environment
        execution_env = self._create_execution_environment(state)
        if not execution_env["success"]:
            return {
                "success": False,
                "score": 0.0,
                "output": {},
                "feedback": f"Failed to create execution environment: {execution_env['error']}",
                "artifacts": [],
                "memory": {}
            }
        
        # Execute code and capture results
        execution_results = self._execute_poc_code(execution_env["env_path"], implementation, state)
        
        # Run tests if available
        test_results = self._run_tests(execution_env["env_path"], implementation, state)
        
        # Measure performance metrics
        performance_metrics = self._measure_performance(execution_results, test_results)
        
        # Validate against requirements
        validation_results = self._validate_against_requirements(execution_results, implementation, project, state)
        
        # Analyze errors and determine if implementation needs improvement
        error_analysis = self._analyze_execution_errors(execution_results, test_results, implementation, project)
        
        # Create evaluation result
        evaluation_result = self._create_evaluation_result(
            execution_results, test_results, performance_metrics, validation_results, implementation, project
        )
        
        # Store evaluation results in state
        state["evaluation_results"] = evaluation_result
        
        # Update implementation with execution logs
        implementation.execution_logs.extend(execution_results.get("logs", []))
        implementation.performance_metrics.update(performance_metrics)
        
        # Save execution results
        artifacts = self._save_execution_artifacts(execution_results, test_results, evaluation_result, state)
        
        # Clean up execution environment
        self._cleanup_execution_environment(execution_env["env_path"])
        
        # Log execution summary
        self._log_execution_summary(execution_results, test_results, evaluation_result)
        
        feedback = f"""
Code Execution Complete:
- Execution Status: {'SUCCESS' if execution_results.get('success', False) else 'FAILED'}
- Test Results: {test_results.get('tests_passed', 0)}/{test_results.get('total_tests', 0)} passed
- Performance Score: {performance_metrics.get('overall_performance', 0.0):.2f}/1.0
- Validation Score: {validation_results.get('overall_validation', 0.0):.2f}/1.0
- Overall Evaluation: {evaluation_result.overall_score:.2f}/1.0
- Execution Time: {execution_results.get('execution_time', 0):.2f}s
- Memory Usage: {performance_metrics.get('memory_usage', 'N/A')}
- Critical Issues: {len(execution_results.get('errors', []))}
"""
        
        score = evaluation_result.overall_score
        
        return {
            "success": execution_results.get('success', False),
            "score": score,
            "output": {
                "execution_success": execution_results.get('success', False),
                "test_results": test_results,
                "performance_metrics": performance_metrics,
                "validation_score": validation_results.get('overall_validation', 0.0),
                "evaluation_score": evaluation_result.overall_score,
                "error_analysis": error_analysis
            },
            "feedback": feedback,
            "artifacts": artifacts,
            "memory": {
                "execution_results": execution_results,
                "test_results": test_results,
                "performance_metrics": performance_metrics,
                "validation_results": validation_results,
                "evaluation_summary": self._evaluation_to_dict(evaluation_result),
                "error_analysis": error_analysis,
                "executed_at": datetime.now().isoformat()
            },
            # Add feedback loop flags
            "needs_implementation_fix": error_analysis.get("needs_implementation_fix", False),
            "fix_instructions": error_analysis.get("fix_instructions", [])
        }
    
    def _create_execution_environment(self, state: PoCState) -> Dict[str, Any]:
        """Create isolated execution environment for the PoC code."""
        # Create temporary directory for execution
        temp_dir = tempfile.mkdtemp(prefix="poc_execution_")
        execution_path = Path(temp_dir)
        
        # Copy generated code to execution environment
        workspace_path = Path(state["workspace_path"])
        code_dir = workspace_path / "generated_code" / f"iteration_{state['iteration']}"
        
        if not code_dir.exists():
            return {
                "success": False,
                "error": f"Generated code directory not found: {code_dir}"
            }
        
        # Copy all files to execution environment
        shutil.copytree(code_dir, execution_path / "poc_code", dirs_exist_ok=True)
        
        # Create virtual environment if needed (optional)
        # This could be extended to create actual virtual environments
        
        return {
            "success": True,
            "env_path": execution_path,
            "code_path": execution_path / "poc_code"
        }
    
    def _execute_poc_code(self, env_path: Path, implementation: PoCImplementation, state: PoCState) -> Dict[str, Any]:
        """Execute the PoC code and capture results."""
        execution_results = {
            "success": False,
            "outputs": [],
            "errors": [],
            "logs": [],
            "execution_time": 0.0,
            "exit_code": None,
            "stdout": "",
            "stderr": ""
        }
        
        code_path = env_path / "poc_code"
        
        # Get entry points from implementation memory
        implementation_memory = state["agent_memory"].get("poc_implementer", {})
        entry_points = implementation_memory.get("entry_points", {})
        
        # Find main execution files
        main_files = self._find_main_files(code_path)
        
        if not main_files and not entry_points:
            execution_results["errors"].append("No main execution files found")
            return execution_results
        
        # Execute main files or entry points
        start_time = datetime.now()
        
        for main_file in main_files[:3]:  # Execute up to 3 main files
            result = self._execute_python_file(code_path, main_file)
            execution_results["outputs"].append(result)
            execution_results["logs"].extend(result.get("logs", []))
            
            if result.get("success"):
                execution_results["success"] = True
            
            execution_results["errors"].extend(result.get("errors", []))
        
        end_time = datetime.now()
        execution_results["execution_time"] = (end_time - start_time).total_seconds()
        
        # Execute entry points if specified
        for entry_name, entry_file in entry_points.items():
            if entry_file in [f.name for f in main_files]:
                continue  # Already executed
            
            entry_result = self._execute_python_file(code_path, entry_file)
            execution_results["outputs"].append(entry_result)
            execution_results["logs"].extend(entry_result.get("logs", []))
            
            if entry_result.get("success"):
                execution_results["success"] = True
        
        # Combine all outputs
        all_stdout = []
        all_stderr = []
        for output in execution_results["outputs"]:
            all_stdout.append(output.get("stdout", ""))
            all_stderr.append(output.get("stderr", ""))
        
        execution_results["stdout"] = "\n".join(all_stdout)
        execution_results["stderr"] = "\n".join(all_stderr)
        
        return execution_results
    
    def _find_main_files(self, code_path: Path) -> List[Path]:
        """Find main executable Python files."""
        main_files = []
        
        # Look for common main file patterns
        patterns = ["main.py", "app.py", "run.py", "__main__.py", "server.py", "cli.py"]
        
        for pattern in patterns:
            main_file = code_path / pattern
            if main_file.exists():
                main_files.append(main_file)
        
        # If no standard main files, look for files with if __name__ == "__main__"
        if not main_files:
            for py_file in code_path.glob("*.py"):
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                if 'if __name__ == "__main__"' in content:
                    main_files.append(py_file)
        
        return main_files
    
    def _execute_python_file(self, code_path: Path, python_file: Path, timeout: int = 3000) -> Dict[str, Any]:
        """Execute a single Python file and capture results."""
        result = {
            "file": str(python_file),
            "success": False,
            "stdout": "",
            "stderr": "",
            "exit_code": None,
            "execution_time": 0.0,
            "logs": [],
            "errors": []
        }
        
        # Check if file exists
        if not python_file.exists():
            result["errors"].append(f"File not found: {python_file}")
            return result
        
        # Prepare execution command
        cmd = [sys.executable, str(python_file)]
        
        # Execute with timeout
        start_time = datetime.now()
        
        # Change to code directory for execution
        original_cwd = os.getcwd()
        os.chdir(code_path)
        
        # Execute the command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=code_path
        )
        
        # Wait for completion or timeout
        stdout, stderr = process.communicate(timeout=timeout)
        result["stdout"] = stdout
        result["stderr"] = stderr
        result["exit_code"] = process.returncode
        result["success"] = process.returncode == 0
        
        if result["success"]:
            result["logs"].append(f"Successfully executed {python_file.name}")
        else:
            result["errors"].append(f"Execution failed with exit code {process.returncode}")
            if stderr:
                result["errors"].append(f"Error output: {stderr}")
        
        # Restore original working directory
        os.chdir(original_cwd)
        
        end_time = datetime.now()
        result["execution_time"] = (end_time - start_time).total_seconds()
        
        return result
    
    def _run_tests(self, env_path: Path, implementation: PoCImplementation, state: PoCState) -> Dict[str, Any]:
        """Run tests if available and capture results."""
        test_results = {
            "tests_run": False,
            "total_tests": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "test_outputs": [],
            "test_errors": [],
            "test_execution_time": 0.0
        }
        
        code_path = env_path / "poc_code"
        
        # Find test files
        test_files = list(code_path.glob("*test*.py")) + list(code_path.glob("test_*.py"))
        
        if not test_files:
            test_results["test_errors"].append("No test files found")
            return test_results
        
        test_results["tests_run"] = True
        
        # Run each test file
        start_time = datetime.now()
        
        for test_file in test_files:
            test_result = self._execute_python_file(code_path, test_file)
            test_results["test_outputs"].append(test_result)
            
            # Parse test results (basic parsing)
            if test_result["success"]:
                test_results["tests_passed"] += 1
            else:
                test_results["tests_failed"] += 1
                test_results["test_errors"].extend(test_result["errors"])
            
            test_results["total_tests"] += 1
        
        end_time = datetime.now()
        test_results["test_execution_time"] = (end_time - start_time).total_seconds()
        
        return test_results
    
    def _measure_performance(self, execution_results: Dict[str, Any], test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Measure and analyze performance metrics."""
        performance_metrics = {
            "execution_time": execution_results.get("execution_time", 0.0),
            "test_execution_time": test_results.get("test_execution_time", 0.0),
            "total_execution_time": execution_results.get("execution_time", 0.0) + test_results.get("test_execution_time", 0.0),
            "success_rate": 1.0 if execution_results.get("success") else 0.0,
            "test_success_rate": 0.0,
            "overall_performance": 0.0,
            "memory_usage": "N/A",  # Could be extended with actual memory measurement
            "cpu_usage": "N/A"      # Could be extended with actual CPU measurement
        }
        
        # Calculate test success rate
        if test_results.get("total_tests", 0) > 0:
            performance_metrics["test_success_rate"] = test_results.get("tests_passed", 0) / test_results.get("total_tests", 1)
        
        # Calculate overall performance score
        time_score = min(1.0, max(0.0, (60.0 - performance_metrics["total_execution_time"]) / 60.0))  # 60s max
        success_score = performance_metrics["success_rate"] * 0.6 + performance_metrics["test_success_rate"] * 0.4
        
        performance_metrics["overall_performance"] = (time_score * 0.3 + success_score * 0.7)
        
        return performance_metrics
    
    def _validate_against_requirements(self, execution_results: Dict[str, Any], implementation: PoCImplementation, project, state: PoCState) -> Dict[str, Any]:
        """Validate execution results against project requirements."""
        validation_results = {
            "requirements_met": [],
            "requirements_failed": [],
            "functional_validation": 0.0,
            "technical_validation": 0.0,
            "overall_validation": 0.0,
            "validation_details": []
        }
        
        # Check functional requirements
        functional_req = implementation.environment_config.get("functional_requirements", {})
        success = execution_results.get("success", False)
        
        # Basic validation based on execution success
        if success:
            validation_results["requirements_met"].append("Code executes successfully")
            validation_results["functional_validation"] += 0.5
        else:
            validation_results["requirements_failed"].append("Code execution failed")
        
        # Check for output generation
        if execution_results.get("stdout") or execution_results.get("outputs"):
            validation_results["requirements_met"].append("Generates output")
            validation_results["functional_validation"] += 0.3
        
        # Check for error handling (should not have critical errors)
        critical_errors = [e for e in execution_results.get("errors", []) if "error" in e.lower()]
        if len(critical_errors) == 0:
            validation_results["requirements_met"].append("No critical errors")
            validation_results["technical_validation"] += 0.4
        else:
            validation_results["requirements_failed"].append(f"Critical errors found: {len(critical_errors)}")
        
        # Check execution time (should be reasonable)
        exec_time = execution_results.get("execution_time", 0)
        if exec_time < 30.0:  # 30 seconds threshold
            validation_results["requirements_met"].append("Reasonable execution time")
            validation_results["technical_validation"] += 0.3
        else:
            validation_results["requirements_failed"].append("Execution time too long")
        
        # Check project requirements
        for requirement in project.requirements:
            # Simple keyword matching validation
            stdout_content = execution_results.get("stdout", "").lower()
            if any(word in stdout_content for word in requirement.lower().split()):
                validation_results["requirements_met"].append(f"Requirement addressed: {requirement}")
                validation_results["functional_validation"] += 0.1
        
        # Normalize scores
        validation_results["functional_validation"] = min(1.0, validation_results["functional_validation"])
        validation_results["technical_validation"] = min(1.0, validation_results["technical_validation"])
        validation_results["overall_validation"] = (validation_results["functional_validation"] + validation_results["technical_validation"]) / 2.0
        
        return validation_results
    
    def _create_evaluation_result(self, execution_results: Dict[str, Any], test_results: Dict[str, Any], 
                                performance_metrics: Dict[str, Any], validation_results: Dict[str, Any],
                                implementation: PoCImplementation, project) -> EvaluationResult:
        """Create comprehensive evaluation result."""
        evaluation = EvaluationResult()
        
        # Overall score calculation
        execution_score = 1.0 if execution_results.get("success") else 0.0
        performance_score = performance_metrics.get("overall_performance", 0.0)
        validation_score = validation_results.get("overall_validation", 0.0)
        
        evaluation.overall_score = (execution_score * 0.4 + performance_score * 0.3 + validation_score * 0.3)
        
        # Component scores
        evaluation.technical_score = performance_score
        evaluation.business_score = validation_score
        evaluation.innovation_score = min(1.0, len(implementation.tech_stack) * 0.2)
        
        # Success criteria
        evaluation.success_criteria_met = [
            execution_results.get("success", False),
            len(execution_results.get("errors", [])) == 0,
            performance_metrics.get("execution_time", 0) < 30.0
        ]
        
        # Metrics
        evaluation.quantitative_metrics = {
            "execution_time": execution_results.get("execution_time", 0.0),
            "test_pass_rate": performance_metrics.get("test_success_rate", 0.0),
            "error_count": len(execution_results.get("errors", [])),
            "lines_of_output": len(execution_results.get("stdout", "").split("\n"))
        }
        
        # Qualitative feedback
        if execution_results.get("success"):
            evaluation.qualitative_feedback = "PoC executed successfully with expected functionality."
        else:
            evaluation.qualitative_feedback = f"PoC execution failed with {len(execution_results.get('errors', []))} errors."
        
        # Strengths and weaknesses
        evaluation.strengths = validation_results.get("requirements_met", [])
        evaluation.weaknesses = validation_results.get("requirements_failed", [])
        
        # Improvement suggestions
        evaluation.improvement_suggestions = []
        if not execution_results.get("success"):
            evaluation.improvement_suggestions.append("Fix execution errors and ensure code runs successfully")
        if performance_metrics.get("execution_time", 0) > 10.0:
            evaluation.improvement_suggestions.append("Optimize performance to reduce execution time")
        if test_results.get("tests_failed", 0) > 0:
            evaluation.improvement_suggestions.append("Fix failing tests to ensure code quality")
        
        # Next steps
        evaluation.next_steps = [
            "Analyze execution results and address any issues",
            "Optimize performance and resource usage",
            "Enhance error handling and robustness",
            "Consider scalability and production readiness"
        ]
        
        # Lessons learned
        evaluation.lessons_learned = [
            f"Code execution {'succeeded' if execution_results.get('success') else 'failed'}",
            f"Performance metrics show {performance_metrics.get('overall_performance', 0.0):.2f}/1.0 score",
            f"Validation against requirements scored {validation_score:.2f}/1.0"
        ]
        
        return evaluation
    
    def _save_execution_artifacts(self, execution_results: Dict[str, Any], test_results: Dict[str, Any],
                                evaluation_result: EvaluationResult, state: PoCState) -> List[str]:
        """Save execution artifacts and results."""
        artifacts = []
        
        # Save execution results
        execution_path = self._save_json_artifact(
            {
                "execution_results": execution_results,
                "test_results": test_results,
                "evaluation": self._evaluation_to_dict(evaluation_result)
            },
            f"execution_iteration_{state['iteration']}.json",
            state
        )
        artifacts.append(execution_path)
        
        # Save execution log
        log_content = self._create_execution_log(execution_results, test_results, evaluation_result)
        log_path = self._save_artifact(
            log_content,
            f"execution_log_iteration_{state['iteration']}.md",
            state
        )
        artifacts.append(log_path)
        
        # Save outputs if significant
        if execution_results.get("stdout"):
            output_path = self._save_artifact(
                execution_results["stdout"],
                f"execution_output_iteration_{state['iteration']}.txt",
                state
            )
            artifacts.append(output_path)
        
        # Save error log if errors exist
        if execution_results.get("stderr") or execution_results.get("errors"):
            error_content = "STDERR:\n" + execution_results.get("stderr", "")
            error_content += "\n\nERRORS:\n" + "\n".join(execution_results.get("errors", []))
            
            error_path = self._save_artifact(
                error_content,
                f"execution_errors_iteration_{state['iteration']}.txt",
                state
            )
            artifacts.append(error_path)
        
        return artifacts
    
    def _create_execution_log(self, execution_results: Dict[str, Any], test_results: Dict[str, Any],
                            evaluation_result: EvaluationResult) -> str:
        """Create comprehensive execution log."""
        log = f"""# Execution Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Execution Summary
- **Status**: {'SUCCESS' if execution_results.get('success') else 'FAILED'}
- **Execution Time**: {execution_results.get('execution_time', 0):.2f} seconds
- **Exit Code**: {execution_results.get('exit_code', 'N/A')}
- **Errors**: {len(execution_results.get('errors', []))}

## Test Results
- **Tests Run**: {'Yes' if test_results.get('tests_run') else 'No'}
- **Total Tests**: {test_results.get('total_tests', 0)}
- **Tests Passed**: {test_results.get('tests_passed', 0)}
- **Tests Failed**: {test_results.get('tests_failed', 0)}
- **Test Execution Time**: {test_results.get('test_execution_time', 0):.2f} seconds

## Performance Metrics
- **Overall Performance Score**: {evaluation_result.quantitative_metrics.get('execution_time', 0.0):.3f}/1.0
- **Success Rate**: {(1.0 if execution_results.get('success') else 0.0):.3f}
- **Test Success Rate**: {test_results.get('tests_passed', 0) / max(test_results.get('total_tests', 1), 1):.3f}

## Evaluation Results
- **Overall Score**: {evaluation_result.overall_score:.3f}/1.0
- **Technical Score**: {evaluation_result.technical_score:.3f}/1.0
- **Business Score**: {evaluation_result.business_score:.3f}/1.0
- **Innovation Score**: {evaluation_result.innovation_score:.3f}/1.0

## Execution Output
```
{execution_results.get('stdout', 'No output generated')}
```

## Errors and Issues
"""
        
        if execution_results.get('errors'):
            for i, error in enumerate(execution_results['errors'], 1):
                log += f"{i}. {error}\n"
        else:
            log += "No errors encountered.\n"
        
        log += f"""
## Validation Results
### Requirements Met
"""
        for req in evaluation_result.strengths:
            log += f"- ✅ {req}\n"
        
        log += "\n### Requirements Failed\n"
        for req in evaluation_result.weaknesses:
            log += f"- ❌ {req}\n"
        
        log += f"""
## Recommendations
"""
        for suggestion in evaluation_result.improvement_suggestions:
            log += f"- {suggestion}\n"
        
        log += f"""
## Next Steps
"""
        for step in evaluation_result.next_steps:
            log += f"- {step}\n"
        
        return log
    
    def _evaluation_to_dict(self, evaluation: EvaluationResult) -> Dict[str, Any]:
        """Convert EvaluationResult to dictionary."""
        return {
            "overall_score": evaluation.overall_score,
            "technical_score": evaluation.technical_score,
            "business_score": evaluation.business_score,
            "innovation_score": evaluation.innovation_score,
            "success_criteria_met": evaluation.success_criteria_met,
            "quantitative_metrics": evaluation.quantitative_metrics,
            "qualitative_feedback": evaluation.qualitative_feedback,
            "strengths": evaluation.strengths,
            "weaknesses": evaluation.weaknesses,
            "improvement_suggestions": evaluation.improvement_suggestions,
            "next_steps": evaluation.next_steps,
            "lessons_learned": evaluation.lessons_learned
        }
    
    def _cleanup_execution_environment(self, env_path: Path) -> None:
        """Clean up the execution environment."""
        if env_path.exists():
            shutil.rmtree(env_path, ignore_errors=True)
    
    def _log_execution_summary(self, execution_results: Dict[str, Any], test_results: Dict[str, Any], evaluation_result: EvaluationResult) -> None:
        """Log execution summary with color formatting."""
        success = execution_results.get('success', False)
        color = "\033[92m" if success else "\033[91m"  # Green if success, red if failed
        
        print(color + "="*60)
        print("⚡ CODE EXECUTION RESULTS")
        print("="*60)
        print(f"🎯 Execution Status: {'SUCCESS' if success else 'FAILED'}")
        print(f"⏱️ Execution Time: {execution_results.get('execution_time', 0):.2f}s")
        print(f"🧪 Tests: {test_results.get('tests_passed', 0)}/{test_results.get('total_tests', 0)} passed")
        print(f"📊 Overall Score: {evaluation_result.overall_score:.3f}/1.0")
        print(f"🔧 Technical Score: {evaluation_result.technical_score:.3f}/1.0")
        print(f"💼 Business Score: {evaluation_result.business_score:.3f}/1.0")
        print(f"❌ Errors: {len(execution_results.get('errors', []))}")
        
        if execution_results.get('stdout'):
            stdout_content = execution_results.get('stdout', '')
            output_lines = len(stdout_content.split('\n'))
            print(f"📄 Output Lines: {output_lines}")
        
        print(f"\n📝 Key Results:")
        for strength in evaluation_result.strengths[:3]:
            print(f"   ✅ {strength}")
        
        for weakness in evaluation_result.weaknesses[:3]:
            print(f"   ❌ {weakness}")
        
        print("="*60 + "\033[0m")  # Reset color
    
    def _analyze_execution_errors(self, execution_results: Dict[str, Any], test_results: Dict[str, Any], 
                                implementation: PoCImplementation, project) -> Dict[str, Any]:
        """Analyze execution errors and determine if implementation needs fixing."""
        
        error_analysis = {
            "needs_implementation_fix": False,
            "fix_instructions": [],
            "error_categories": [],
            "severity": "low",  # low, medium, high, critical
            "fixable_errors": [],
            "systemic_issues": []
        }
        
        # Check execution success
        execution_success = execution_results.get('success', False)
        errors = execution_results.get('errors', [])
        stderr = execution_results.get('stderr', '')
        
        # Analyze different types of errors
        if not execution_success or errors:
            
            # Import/Module errors (usually fixable)
            import_errors = []
            syntax_errors = []
            runtime_errors = []
            dependency_errors = []
            
            all_error_text = ' '.join(errors) + ' ' + stderr
            
            # Categorize errors
            if 'ModuleNotFoundError' in all_error_text or 'ImportError' in all_error_text:
                import_errors.append("Missing module dependencies")
                dependency_errors.append("Check requirements.txt and install missing packages")
                
            if 'SyntaxError' in all_error_text or 'IndentationError' in all_error_text:
                syntax_errors.append("Code syntax issues")
                
            if 'NameError' in all_error_text or 'AttributeError' in all_error_text:
                runtime_errors.append("Variable or method name issues")
                
            if 'FileNotFoundError' in all_error_text:
                runtime_errors.append("Missing files or incorrect paths")
                
            if 'TypeError' in all_error_text or 'ValueError' in all_error_text:
                runtime_errors.append("Data type or value issues")
            
            # Determine if implementation fix is needed
            fixable_issues = len(import_errors) + len(syntax_errors) + len(runtime_errors)
            
            if fixable_issues > 0:
                error_analysis["needs_implementation_fix"] = True
                error_analysis["severity"] = "high" if fixable_issues > 2 else "medium"
                
                # Generate specific fix instructions
                if import_errors:
                    error_analysis["fix_instructions"].append(
                        "Fix import dependencies: Update requirements.txt and check module availability"
                    )
                    
                if syntax_errors:
                    error_analysis["fix_instructions"].append(
                        "Fix syntax errors: Check indentation, brackets, and Python syntax"
                    )
                    
                if runtime_errors:
                    error_analysis["fix_instructions"].append(
                        "Fix runtime issues: Check variable names, file paths, and data types"
                    )
                    
                if dependency_errors:
                    error_analysis["fix_instructions"].append(
                        "Fix dependencies: " + "; ".join(dependency_errors)
                    )
                
                error_analysis["error_categories"] = import_errors + syntax_errors + runtime_errors
                error_analysis["fixable_errors"] = errors[:5]  # Limit to first 5 errors
        
        # Check test failures
        tests_failed = test_results.get('total_tests', 0) - test_results.get('tests_passed', 0)
        if tests_failed > 0:
            error_analysis["fix_instructions"].append(
                f"Fix failing tests: {tests_failed} tests failed, check test implementation"
            )
            if tests_failed > 2:
                error_analysis["needs_implementation_fix"] = True
                error_analysis["severity"] = "medium"
        
        # Check if no output was produced (might indicate core logic issues)
        stdout = execution_results.get('stdout', '')
        if execution_success and not stdout.strip():
            error_analysis["systemic_issues"].append("No output produced - check main logic flow")
            error_analysis["fix_instructions"].append(
                "Add output generation: Ensure the code produces expected results"
            )
        
        return error_analysis