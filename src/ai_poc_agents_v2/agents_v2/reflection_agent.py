"""Reflection Agent for AI-PoC-Agents-v2."""

from typing import Dict, Any, List, Optional
import json
from datetime import datetime
from pathlib import Path

from ..agents.base_agent import BaseAgent
from ..core.state import PoCState, EvaluationResult


class ReflectionAgent(BaseAgent):
    """Agent responsible for overall analysis, improvement suggestions, and process reflection."""
    
    def __init__(self, config):
        """Initialize ReflectionAgent."""
        super().__init__("reflector", config)
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        return """
You are an expert Reflection Agent for comprehensive PoC analysis and improvement guidance.

Your responsibilities:
- HOLISTIC ANALYSIS: Analyze the entire PoC development process from start to finish
- PROCESS EVALUATION: Evaluate the effectiveness of each phase and agent interaction
- IMPROVEMENT IDENTIFICATION: Identify specific areas for enhancement and optimization
- LESSONS LEARNED: Extract valuable insights and learning from the development process
- SUCCESS ASSESSMENT: Provide objective assessment of PoC success against original goals
- STRATEGIC RECOMMENDATIONS: Offer strategic guidance for next steps and future development
- QUALITY REFLECTION: Reflect on quality indicators and areas for improvement

Key principles:
- Provide honest, constructive, and actionable analysis
- Consider both technical and business perspectives
- Balance success recognition with improvement opportunities
- Focus on systemic insights that can improve future PoC development
- Identify patterns, bottlenecks, and optimization opportunities
- Consider scalability, maintainability, and production readiness
- Provide specific, measurable recommendations
- Highlight both process and outcome improvements

Always provide comprehensive, insightful reflection that drives continuous improvement.
"""
    
    def execute_phase(self, state: PoCState) -> Dict[str, Any]:
        """Execute reflection analysis on the entire PoC development process."""
        
        project = state["project"]
        context = self._get_project_context(state)
        previous_results = self._get_previous_results(state)
        
        # Get comprehensive data from all previous phases
        reflection_data = self._gather_reflection_data(state)
        
        # Analyze execution results if available
        evaluation_results = state.get("evaluation_results")
        if not evaluation_results:
            return {
                "success": False,
                "score": 0.0,
                "output": {},
                "feedback": "No evaluation results found. Please run execution phase first.",
                "artifacts": [],
                "memory": {}
            }
        
        # Get key execution results
        execution_success = evaluation_results.overall_score > 0.5 if evaluation_results else False
        
        # Try multiple possible keys for implementation agent memory
        implementation_memory = (state["agent_memory"].get("implementation", {}) or 
                               state["agent_memory"].get("poc_implementer", {}))
        code_files_count = len(implementation_memory.get("generated_code_files", []))
        
        # Also check if there's an implementation object with code_files
        if code_files_count == 0 and "implementation" in state:
            implementation = state["implementation"]
            if hasattr(implementation, 'code_files'):
                code_files_count = len(implementation.code_files)
        
        user_prompt = f"""
PROJECT: {project.theme}
DESCRIPTION: {project.description}

EXECUTION RESULTS:
- Code Generated: {code_files_count} files
- Execution Success: {execution_success}
- Overall Score: {evaluation_results.overall_score if evaluation_results else 0.0:.2f}

TASK: Provide concise reflection focusing on problem solution and execution results.

Analyze these key aspects:

1. PROBLEM SOLUTION ASSESSMENT
   - Did the PoC solve the original problem?
   - Quality of the solution approach
   - Overall success score (1-10)

2. EXECUTION RESULTS  
   - Code generation success
   - Runtime execution status
   - Technical functionality

3. KEY IMPROVEMENTS
   - Most critical issues to fix
   - Next steps for better results

Provide concise, actionable analysis.
"""
        
        schema = {
            "problem_solution_assessment": "dict",
            "execution_results": "dict",
            "key_improvements": "dict"
        }
        
        response = self._generate_structured_response(
            self.get_system_prompt(),
            user_prompt,
            schema
        )
        
        # Create simplified reflection summary
        reflection_summary = self._create_simple_reflection_summary(response, evaluation_results, project)
        
        # Save reflection artifacts
        artifacts = self._save_simple_reflection_artifacts(response, reflection_summary, state)
        
        # Calculate reflection quality score
        reflection_score = self._calculate_simple_reflection_score(response, evaluation_results)
        
        # Log reflection summary
        self._log_reflection_summary(reflection_summary, reflection_score)
        
        feedback = f"""
PoC Reflection Complete:
- Overall Success Score: {reflection_summary.get('overall_success_score', 0.0):.1f}/10
- Code Generated: {len(state["agent_memory"].get("poc_implementer", {}).get("generated_code_files", []))} files
- Execution Success: {evaluation_results.overall_score > 0.5 if evaluation_results else False}
- Key Improvements: {len(response.get('key_improvements', {}).get('critical_fixes', []))}
"""
        
        # Determine if a different idea should be tried
        overall_success = reflection_summary.get('overall_success_score', 0.0)
        execution_success = evaluation_results.overall_score > 0.5 if evaluation_results else False
        needs_idea_change = self._should_try_different_idea(overall_success, execution_success, response, state)
        
        return {
            "success": True,
            "score": reflection_score,
            "output": {
                "reflection_summary": reflection_summary,
                "improvement_count": len(response.get('key_improvements', {}).get('critical_fixes', [])),
                "overall_success": reflection_summary.get('overall_success_score', 0.0),
                "execution_success": execution_success
            },
            "feedback": feedback,
            "artifacts": artifacts,
            "memory": {
                "reflection_analysis": response,
                "reflection_summary": reflection_summary,
                "overall_assessment": reflection_summary.get('overall_success_score', 0.0),
                "key_improvements": response.get('key_improvements', {}),
                "reflected_at": datetime.now().isoformat()
            },
            # Add feedback loop flags for idea change
            "needs_idea_change": needs_idea_change,
            "idea_change_reasons": self._get_idea_change_reasons(response, overall_success, execution_success) if needs_idea_change else []
        }
    
    def _gather_reflection_data(self, state: PoCState) -> Dict[str, Any]:
        """Gather comprehensive data from all phases for reflection."""
        reflection_data = {
            "project_info": {
                "theme": state["project"].theme,
                "description": state["project"].description,
                "domain": state["project"].domain,
                "requirements": state["project"].requirements,
                "success_criteria": state["project"].success_criteria,
                "timeline_days": state["project"].timeline_days
            },
            "workflow_metrics": {
                "total_phases": len(state["phase_results"]),
                "completed_phases": len(state["completed_phases"]),
                "current_iteration": state["iteration"],
                "overall_score": state["overall_score"],
                "phase_scores": state["phase_scores"]
            },
            "agent_performance": {},
            "execution_timeline": [],
            "artifacts_generated": len(state["artifacts"]),
            "total_logs": len(state["logs"])
        }
        
        # Analyze agent performance
        for agent_type, memory in state["agent_memory"].items():
            agent_results = [r for r in state["phase_results"] if r.agent == agent_type]
            if agent_results:
                avg_score = sum(r.score for r in agent_results) / len(agent_results)
                avg_time = sum(r.execution_time for r in agent_results) / len(agent_results)
                
                reflection_data["agent_performance"][agent_type] = {
                    "average_score": avg_score,
                    "average_execution_time": avg_time,
                    "total_executions": len(agent_results),
                    "memory_size": len(str(memory)),
                    "last_execution": agent_results[-1].timestamp.isoformat() if agent_results else None
                }
        
        # Create execution timeline
        for result in state["phase_results"]:
            reflection_data["execution_timeline"].append({
                "phase": result.phase,
                "agent": result.agent,
                "iteration": result.iteration,
                "score": result.score,
                "execution_time": result.execution_time,
                "success": result.success,
                "timestamp": result.timestamp.isoformat()
            })
        
        # Add specific agent insights
        reflection_data["agent_insights"] = {
            "problem_identification": state["agent_memory"].get("problem_identifier", {}),
            "search_results": state["agent_memory"].get("search_problem", {}).get("search_results", {}),
            "ideas_generated": state["agent_memory"].get("idea_generation", {}).get("generation_count", 0),
            "design_complexity": state["agent_memory"].get("poc_designer", {}).get("architecture_complexity", 0),
            "implementation_files": len(state["agent_memory"].get("poc_implementer", {}).get("generated_code_files", [])),
            "execution_success": state["agent_memory"].get("poc_executor", {}).get("execution_results", {}).get("success", False)
        }
        
        return reflection_data
    
    def _analyze_development_process(self, state: PoCState, reflection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the effectiveness of the development process."""
        analysis = {
            "phase_efficiency": {},
            "bottlenecks_identified": [],
            "handoff_quality": [],
            "resource_utilization": {},
            "process_improvements": []
        }
        
        # Analyze phase efficiency
        for phase, score in state["phase_scores"].items():
            phase_results = [r for r in state["phase_results"] if r.phase == phase]
            if phase_results:
                avg_time = sum(r.execution_time for r in phase_results) / len(phase_results)
                iterations = len(phase_results)
                
                analysis["phase_efficiency"][phase] = {
                    "average_score": score,
                    "average_time": avg_time,
                    "iterations_needed": iterations,
                    "efficiency_rating": min(10, max(1, (score * 10) / max(iterations, 1)))
                }
                
                # Identify bottlenecks
                if iterations > 2:
                    analysis["bottlenecks_identified"].append(f"Phase {phase} required {iterations} iterations")
                if avg_time > 30.0:
                    analysis["bottlenecks_identified"].append(f"Phase {phase} took {avg_time:.1f}s on average")
        
        # Analyze agent handoffs
        for i in range(len(state["phase_results"]) - 1):
            current = state["phase_results"][i]
            next_result = state["phase_results"][i + 1]
            
            handoff_quality = {
                "from_agent": current.agent,
                "to_agent": next_result.agent,
                "score_transition": next_result.score - current.score,
                "time_gap": (next_result.timestamp - current.timestamp).total_seconds(),
                "success": next_result.success and current.success
            }
            analysis["handoff_quality"].append(handoff_quality)
        
        # Resource utilization analysis
        total_time = sum(r.execution_time for r in state["phase_results"])
        analysis["resource_utilization"] = {
            "total_execution_time": total_time,
            "average_phase_time": total_time / len(state["phase_results"]) if state["phase_results"] else 0,
            "artifacts_per_hour": len(state["artifacts"]) / max(total_time / 3600, 0.1),
            "efficiency_score": state["overall_score"] / max(total_time / 60, 1)  # Score per minute
        }
        
        return analysis
    
    def _analyze_outcomes(self, evaluation_results: EvaluationResult, state: PoCState) -> Dict[str, Any]:
        """Analyze the outcomes and deliverables."""
        analysis = {
            "deliverable_quality": {
                "overall_score": evaluation_results.overall_score,
                "technical_score": evaluation_results.technical_score,
                "business_score": evaluation_results.business_score,
                "innovation_score": evaluation_results.innovation_score
            },
            "success_criteria_analysis": {
                "criteria_met": sum(evaluation_results.success_criteria_met),
                "total_criteria": len(evaluation_results.success_criteria_met),
                "success_rate": sum(evaluation_results.success_criteria_met) / max(len(evaluation_results.success_criteria_met), 1)
            },
            "quantitative_achievements": evaluation_results.quantitative_metrics,
            "qualitative_assessment": evaluation_results.qualitative_feedback,
            "strengths_identified": evaluation_results.strengths,
            "improvement_areas": evaluation_results.weaknesses,
            "artifacts_quality": {
                "total_artifacts": len(state["artifacts"]),
                "code_files_generated": len(state["agent_memory"].get("poc_implementer", {}).get("generated_code_files", [])),
                "documentation_completeness": self._assess_documentation_completeness(state)
            }
        }
        
        return analysis
    
    def _assess_overall_quality(self, state: PoCState, reflection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall quality across all dimensions."""
        quality_assessment = {
            "technical_quality": {
                "code_generation": min(10, len(state["agent_memory"].get("poc_implementer", {}).get("generated_code_files", [])) * 2),
                "architecture_design": state["agent_memory"].get("poc_designer", {}).get("architecture_complexity", 3),
                "execution_success": 10 if state["agent_memory"].get("poc_executor", {}).get("execution_results", {}).get("success", False) else 3,
                "test_coverage": self._assess_test_coverage(state)
            },
            "process_quality": {
                "phase_completion": len(state["completed_phases"]) * 1.5,
                "iteration_efficiency": max(1, 10 - state["iteration"]),
                "score_progression": self._analyze_score_progression(state),
                "error_handling": 10 - len([r for r in state["phase_results"] if not r.success])
            },
            "innovation_quality": {
                "idea_diversity": state["agent_memory"].get("idea_generation", {}).get("generation_count", 0),
                "technology_usage": len(state.get("implementation", {}).tech_stack if state.get("implementation") else []),
                "creative_solutions": self._assess_creative_solutions(state),
                "research_integration": self._assess_research_integration(state)
            },
            "business_quality": {
                "requirement_alignment": self._assess_requirement_alignment(state),
                "value_proposition": self._assess_value_proposition(state),
                "market_readiness": self._assess_market_readiness(state),
                "scalability_potential": self._assess_scalability_potential(state)
            }
        }
        
        return quality_assessment
    
    def _assess_documentation_completeness(self, state: PoCState) -> float:
        """Assess completeness of documentation."""
        doc_indicators = 0
        
        # Check for README files
        if any("README" in artifact for artifact in state["artifacts"]):
            doc_indicators += 2
        
        # Check for design documentation
        if any("design" in artifact.lower() for artifact in state["artifacts"]):
            doc_indicators += 2
        
        # Check for execution logs
        if any("log" in artifact.lower() for artifact in state["artifacts"]):
            doc_indicators += 1
        
        # Check for setup instructions
        if any("setup" in artifact.lower() or "SETUP" in artifact for artifact in state["artifacts"]):
            doc_indicators += 1
        
        return min(10.0, doc_indicators * 1.5)
    
    def _assess_test_coverage(self, state: PoCState) -> float:
        """Assess test coverage quality."""
        implementation = state.get("implementation")
        if not implementation:
            return 0.0
        
        test_score = 0.0
        
        # Check for test cases definition
        if implementation.test_cases:
            test_score += min(5.0, len(implementation.test_cases))
        
        # Check for actual test execution
        execution_memory = state["agent_memory"].get("poc_executor", {})
        test_results = execution_memory.get("test_results", {})
        
        if test_results.get("tests_run"):
            test_score += 3.0
            
            # Bonus for passing tests
            if test_results.get("tests_passed", 0) > 0:
                test_score += 2.0
        
        return min(10.0, test_score)
    
    def _analyze_score_progression(self, state: PoCState) -> float:
        """Analyze how scores progressed through phases."""
        scores = [r.score for r in state["phase_results"]]
        if len(scores) < 2:
            return 5.0
        
        # Calculate trend
        improvements = 0
        for i in range(1, len(scores)):
            if scores[i] > scores[i-1]:
                improvements += 1
        
        improvement_rate = improvements / (len(scores) - 1)
        return min(10.0, improvement_rate * 10 + 2)
    
    def _assess_creative_solutions(self, state: PoCState) -> float:
        """Assess creativity and innovation in solutions."""
        creativity_score = 0.0
        
        # Check idea diversity
        ideas_count = state["agent_memory"].get("idea_generation", {}).get("generation_count", 0)
        creativity_score += min(3.0, ideas_count * 0.5)
        
        # Check technology diversity
        implementation = state.get("implementation")
        if implementation and implementation.tech_stack:
            creativity_score += min(3.0, len(implementation.tech_stack) * 0.5)
        
        # Check architectural complexity
        arch_complexity = state["agent_memory"].get("poc_designer", {}).get("architecture_complexity", 0)
        creativity_score += min(4.0, arch_complexity)
        
        return min(10.0, creativity_score)
    
    def _assess_research_integration(self, state: PoCState) -> float:
        """Assess how well research was integrated."""
        search_memory = state["agent_memory"].get("search_problem", {})
        search_results = search_memory.get("search_results", {})
        
        integration_score = 0.0
        
        # Check search sources
        sources = len(search_results.get("sources_searched", []))
        integration_score += min(3.0, sources * 0.5)
        
        # Check sample code usage
        sample_code = len(search_results.get("sample_code_collection", []))
        integration_score += min(4.0, sample_code * 0.2)
        
        # Check technical approaches
        tech_approaches = len(search_results.get("technical_approaches", []))
        integration_score += min(3.0, tech_approaches * 0.3)
        
        return min(10.0, integration_score)
    
    def _assess_requirement_alignment(self, state: PoCState) -> float:
        """Assess alignment with original requirements."""
        project = state["project"]
        alignment_score = 0.0
        
        # Base score for having requirements
        if project.requirements:
            alignment_score += 3.0
        
        # Check if evaluation addressed requirements
        evaluation_results = state.get("evaluation_results")
        if evaluation_results and evaluation_results.strengths:
            # Simple check for requirement-related strengths
            req_related = sum(1 for strength in evaluation_results.strengths 
                            if any(req.lower() in strength.lower() for req in project.requirements))
            alignment_score += min(4.0, req_related * 2)
        
        # Check success criteria
        if project.success_criteria and evaluation_results:
            criteria_met = sum(evaluation_results.success_criteria_met)
            total_criteria = len(evaluation_results.success_criteria_met)
            if total_criteria > 0:
                alignment_score += (criteria_met / total_criteria) * 3.0
        
        return min(10.0, alignment_score)
    
    def _assess_value_proposition(self, state: PoCState) -> float:
        """Assess business value proposition."""
        value_score = 0.0
        
        # Check execution success
        execution_success = state["agent_memory"].get("poc_executor", {}).get("execution_results", {}).get("success", False)
        if execution_success:
            value_score += 4.0
        
        # Check innovation potential
        evaluation_results = state.get("evaluation_results")
        if evaluation_results:
            value_score += evaluation_results.innovation_score * 3.0
            value_score += evaluation_results.business_score * 3.0
        
        return min(10.0, value_score)
    
    def _assess_market_readiness(self, state: PoCState) -> float:
        """Assess market readiness of the solution."""
        readiness_score = 0.0
        
        # Check completeness
        if state.get("implementation") and state.get("implementation").code_files:
            readiness_score += 3.0
        
        # Check documentation
        if any("README" in artifact for artifact in state["artifacts"]):
            readiness_score += 2.0
        
        # Check deployment instructions
        implementation = state.get("implementation")
        if implementation and implementation.deployment_instructions:
            readiness_score += 2.0
        
        # Check test coverage
        test_coverage = self._assess_test_coverage(state)
        readiness_score += min(3.0, test_coverage * 0.3)
        
        return min(10.0, readiness_score)
    
    def _assess_scalability_potential(self, state: PoCState) -> float:
        """Assess scalability potential."""
        scalability_score = 0.0
        
        # Check architecture design
        arch_complexity = state["agent_memory"].get("poc_designer", {}).get("architecture_complexity", 0)
        scalability_score += min(4.0, arch_complexity)
        
        # Check technology choices
        implementation = state.get("implementation")
        if implementation and implementation.tech_stack:
            scalability_score += min(3.0, len(implementation.tech_stack) * 0.5)
        
        # Check performance metrics
        evaluation_results = state.get("evaluation_results")
        if evaluation_results and evaluation_results.technical_score > 0.7:
            scalability_score += 3.0
        
        return min(10.0, scalability_score)
    
    def _create_reflection_summary(self, response: Dict[str, Any], reflection_data: Dict[str, Any], 
                                 evaluation_results: EvaluationResult, project) -> Dict[str, Any]:
        """Create comprehensive reflection summary."""
        summary = {
            "overall_success_score": 0.0,
            "process_effectiveness_score": 0.0,
            "technical_quality_score": 0.0,
            "business_value_score": 0.0,
            "innovation_score": 0.0,
            "key_achievements": [],
            "major_challenges": [],
            "critical_improvements": [],
            "strategic_priorities": [],
            "success_factors": [],
            "risk_factors": [],
            "next_phase_readiness": 0.0
        }
        
        # Extract scores from response
        success_assessment = response.get("overall_success_assessment", {})
        summary["overall_success_score"] = success_assessment.get("satisfaction_score", 5.0)
        
        process_analysis = response.get("process_effectiveness_analysis", {})
        summary["process_effectiveness_score"] = process_analysis.get("overall_effectiveness", 5.0)
        
        technical_eval = response.get("technical_quality_evaluation", {})
        summary["technical_quality_score"] = technical_eval.get("overall_technical_score", 5.0)
        
        business_eval = response.get("business_value_assessment", {})
        summary["business_value_score"] = business_eval.get("overall_business_score", 5.0)
        
        innovation_eval = response.get("innovation_and_learning", {})
        summary["innovation_score"] = innovation_eval.get("innovation_score", 5.0)
        
        # Extract key insights
        summary["key_achievements"] = success_assessment.get("key_achievements", [])
        summary["major_challenges"] = response.get("risk_and_challenge_analysis", {}).get("major_challenges", [])
        summary["critical_improvements"] = response.get("improvement_recommendations", {}).get("priority_improvements", [])
        summary["strategic_priorities"] = response.get("strategic_next_steps", {}).get("strategic_priorities", [])
        
        # Calculate next phase readiness
        readiness_factors = [
            summary["overall_success_score"] / 10.0,
            summary["technical_quality_score"] / 10.0,
            evaluation_results.overall_score,
            min(1.0, len(summary["key_achievements"]) * 0.2),
            max(0.0, 1.0 - len(summary["major_challenges"]) * 0.1)
        ]
        summary["next_phase_readiness"] = sum(readiness_factors) / len(readiness_factors) * 10
        
        return summary
    
    def _save_reflection_artifacts(self, response: Dict[str, Any], reflection_summary: Dict[str, Any],
                                 reflection_data: Dict[str, Any], state: PoCState) -> List[str]:
        """Save reflection artifacts and analysis."""
        artifacts = []
        
        # Save comprehensive reflection analysis
        reflection_path = self._save_json_artifact(
            {
                "comprehensive_reflection": response,
                "reflection_summary": reflection_summary,
                "reflection_data": reflection_data,
                "analysis_timestamp": datetime.now().isoformat()
            },
            f"reflection_iteration_{state['iteration']}.json",
            state
        )
        artifacts.append(reflection_path)
        
        # Create and save reflection report
        reflection_report = self._create_reflection_report(response, reflection_summary, reflection_data, state)
        report_path = self._save_artifact(
            reflection_report,
            f"reflection_report_iteration_{state['iteration']}.md",
            state
        )
        artifacts.append(report_path)
        
        # Create executive summary
        exec_summary = self._create_executive_summary(reflection_summary, response, state)
        exec_path = self._save_artifact(
            exec_summary,
            f"executive_summary_iteration_{state['iteration']}.md",
            state
        )
        artifacts.append(exec_path)
        
        # Save improvement action plan
        action_plan = self._create_action_plan(response, reflection_summary, state)
        action_path = self._save_artifact(
            action_plan,
            f"improvement_action_plan_iteration_{state['iteration']}.md",
            state
        )
        artifacts.append(action_path)
        
        return artifacts
    
    def _create_reflection_report(self, response: Dict[str, Any], reflection_summary: Dict[str, Any],
                                reflection_data: Dict[str, Any], state: PoCState) -> str:
        """Create comprehensive reflection report."""
        report = f"""# Comprehensive PoC Development Reflection Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Project**: {state['project'].theme}  
**Domain**: {state['project'].domain}  
**Iteration**: {state['iteration']}

## Executive Summary

### Overall Assessment Scores
- **Overall Success**: {reflection_summary.get('overall_success_score', 0.0):.1f}/10
- **Process Effectiveness**: {reflection_summary.get('process_effectiveness_score', 0.0):.1f}/10  
- **Technical Quality**: {reflection_summary.get('technical_quality_score', 0.0):.1f}/10
- **Business Value**: {reflection_summary.get('business_value_score', 0.0):.1f}/10
- **Innovation Score**: {reflection_summary.get('innovation_score', 0.0):.1f}/10
- **Next Phase Readiness**: {reflection_summary.get('next_phase_readiness', 0.0):.1f}/10

## Detailed Analysis

### 1. Overall Success Assessment
{json.dumps(response.get('overall_success_assessment', {}), indent=2)}

### 2. Process Effectiveness Analysis  
{json.dumps(response.get('process_effectiveness_analysis', {}), indent=2)}

### 3. Technical Quality Evaluation
{json.dumps(response.get('technical_quality_evaluation', {}), indent=2)}

### 4. Business Value Assessment
{json.dumps(response.get('business_value_assessment', {}), indent=2)}

### 5. Innovation and Learning
{json.dumps(response.get('innovation_and_learning', {}), indent=2)}

### 6. Risk and Challenge Analysis
{json.dumps(response.get('risk_and_challenge_analysis', {}), indent=2)}

## Improvement Recommendations

### Technical Improvements
"""
        
        tech_improvements = response.get('improvement_recommendations', {}).get('technical_improvements', [])
        for i, improvement in enumerate(tech_improvements, 1):
            report += f"{i}. {improvement}\n"
        
        report += f"""
### Process Improvements
"""
        
        process_improvements = response.get('improvement_recommendations', {}).get('process_improvements', [])
        for i, improvement in enumerate(process_improvements, 1):
            report += f"{i}. {improvement}\n"
        
        report += f"""
## Strategic Next Steps

### Short-term Actions (1-2 weeks)
"""
        
        short_term = response.get('strategic_next_steps', {}).get('short_term_actions', [])
        for i, action in enumerate(short_term, 1):
            report += f"{i}. {action}\n"
        
        report += f"""
### Medium-term Plan (1-3 months)  
"""
        
        medium_term = response.get('strategic_next_steps', {}).get('medium_term_plan', [])
        for i, action in enumerate(medium_term, 1):
            report += f"{i}. {action}\n"
        
        report += f"""
### Long-term Strategic Direction (6+ months)
"""
        
        long_term = response.get('strategic_next_steps', {}).get('long_term_direction', [])
        for i, action in enumerate(long_term, 1):
            report += f"{i}. {action}\n"
        
        report += f"""
## Key Lessons Learned

### What Worked Well
"""
        
        worked_well = response.get('lessons_learned', {}).get('what_worked_well', [])
        for item in worked_well:
            report += f"- {item}\n"
        
        report += f"""
### Areas for Improvement  
"""
        
        improvements = response.get('lessons_learned', {}).get('areas_for_improvement', [])
        for item in improvements:
            report += f"- {item}\n"
        
        report += f"""
### Unexpected Discoveries
"""
        
        discoveries = response.get('lessons_learned', {}).get('unexpected_discoveries', [])
        for item in discoveries:
            report += f"- {item}\n"
        
        report += f"""
## Development Metrics

### Phase Performance
"""
        
        for phase, data in reflection_data.get('agent_performance', {}).items():
            report += f"- **{phase}**: Score {data.get('average_score', 0.0):.3f}, Time {data.get('average_execution_time', 0.0):.1f}s\n"
        
        report += f"""
### Artifacts Generated
- Total Artifacts: {reflection_data.get('artifacts_generated', 0)}
- Code Files: {len(state['agent_memory'].get('poc_implementer', {}).get('generated_code_files', []))}
- Documentation Files: {len([a for a in state['artifacts'] if '.md' in a or 'README' in a])}

### Execution Statistics  
- Total Execution Time: {sum(r.execution_time for r in state['phase_results']):.1f} seconds
- Average Phase Time: {sum(r.execution_time for r in state['phase_results']) / max(len(state['phase_results']), 1):.1f} seconds
- Success Rate: {sum(1 for r in state['phase_results'] if r.success) / max(len(state['phase_results']), 1):.1%}

---
*This reflection report provides comprehensive analysis and recommendations for continuous improvement.*
"""
        
        return report
    
    def _create_executive_summary(self, reflection_summary: Dict[str, Any], response: Dict[str, Any], state: PoCState) -> str:
        """Create executive summary for stakeholders."""
        summary = f"""# Executive Summary - PoC Development Results

**Project**: {state['project'].theme}  
**Completion Date**: {datetime.now().strftime('%Y-%m-%d')}  
**Overall Success Score**: {reflection_summary.get('overall_success_score', 0.0):.1f}/10

## Key Achievements

"""
        
        achievements = reflection_summary.get('key_achievements', [])
        for achievement in achievements[:5]:  # Top 5 achievements
            summary += f"✅ {achievement}\n"
        
        summary += f"""
## Success Metrics

- **Technical Quality**: {reflection_summary.get('technical_quality_score', 0.0):.1f}/10
- **Business Value**: {reflection_summary.get('business_value_score', 0.0):.1f}/10  
- **Innovation Level**: {reflection_summary.get('innovation_score', 0.0):.1f}/10
- **Next Phase Readiness**: {reflection_summary.get('next_phase_readiness', 0.0):.1f}/10

## Key Challenges Addressed

"""
        
        challenges = reflection_summary.get('major_challenges', [])
        for challenge in challenges[:3]:  # Top 3 challenges
            summary += f"⚠️ {challenge}\n"
        
        summary += f"""
## Strategic Recommendations

### Immediate Actions Required
"""
        
        strategic_priorities = reflection_summary.get('strategic_priorities', [])
        for priority in strategic_priorities[:3]:
            summary += f"🎯 {priority}\n"
        
        summary += f"""
### Investment Requirements
{json.dumps(response.get('strategic_next_steps', {}).get('investment_requirements', 'To be determined'), indent=2)}

### Expected ROI
{json.dumps(response.get('business_value_assessment', {}).get('roi_potential', 'Positive outlook based on technical success'), indent=2)}

## Conclusion

The PoC development process has achieved a **{reflection_summary.get('overall_success_score', 0.0):.1f}/10** overall success rating. 
Key strengths include technical implementation and innovation, while areas for improvement focus on 
{', '.join(reflection_summary.get('critical_improvements', [])[:2])}.

**Recommendation**: {'Proceed to next phase' if reflection_summary.get('next_phase_readiness', 0.0) > 6.0 else 'Address critical issues before proceeding'}

---
*Prepared by: AI-PoC-Agents-v2 Reflection Agent*
"""
        
        return summary
    
    def _create_action_plan(self, response: Dict[str, Any], reflection_summary: Dict[str, Any], state: PoCState) -> str:
        """Create actionable improvement plan."""
        plan = f"""# PoC Improvement Action Plan

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Priority Level**: {'High' if reflection_summary.get('overall_success_score', 0.0) < 7.0 else 'Medium'}

## Critical Improvements (High Priority)

"""
        
        critical_improvements = response.get('improvement_recommendations', {}).get('critical_improvements', [])
        for i, improvement in enumerate(critical_improvements, 1):
            plan += f"### {i}. {improvement}\n"
            plan += "- **Timeline**: 1-2 weeks\n"
            plan += "- **Resources**: TBD\n"
            plan += "- **Success Criteria**: TBD\n\n"
        
        plan += f"""
## Technical Improvements (Medium Priority)

"""
        
        tech_improvements = response.get('improvement_recommendations', {}).get('technical_improvements', [])[:5]
        for i, improvement in enumerate(tech_improvements, 1):
            plan += f"{i}. {improvement}\n"
        
        plan += f"""
## Process Improvements (Medium Priority)

"""
        
        process_improvements = response.get('improvement_recommendations', {}).get('process_improvements', [])[:5]
        for i, improvement in enumerate(process_improvements, 1):
            plan += f"{i}. {improvement}\n"
        
        plan += f"""
## Implementation Timeline

### Week 1-2: Critical Issues
- Address execution failures and critical bugs
- Implement essential missing features
- Improve code quality and documentation

### Week 3-4: Technical Enhancements  
- Optimize performance and scalability
- Enhance error handling and robustness
- Expand test coverage

### Month 2: Process Optimization
- Improve development workflow
- Enhance documentation and training
- Implement monitoring and logging

### Month 3+: Strategic Development
- Plan production deployment
- Develop maintenance procedures
- Create scaling and evolution roadmap

## Success Metrics

- **Technical**: Code execution success rate > 95%
- **Quality**: Test coverage > 80%
- **Performance**: Response time < 2 seconds
- **Business**: User satisfaction > 8/10

## Resource Requirements

- **Development**: {len(tech_improvements)} technical tasks
- **Process**: {len(process_improvements)} process improvements  
- **Documentation**: {len([a for a in state['artifacts'] if '.md' in a])} documents to update
- **Testing**: {len(state.get('implementation', {}).test_cases if state.get('implementation') else [])} test cases to enhance

---
*This action plan provides structured approach to continuous improvement and success.*
"""
        
        return plan
    
    def _calculate_reflection_score(self, response: Dict[str, Any], reflection_data: Dict[str, Any], evaluation_results: EvaluationResult) -> float:
        """Calculate quality score for the reflection analysis."""
        score = 0.0
        
        # Base score for comprehensive analysis
        score += 0.2
        
        # Score for analysis depth
        analysis_sections = [
            'overall_success_assessment',
            'process_effectiveness_analysis', 
            'technical_quality_evaluation',
            'business_value_assessment',
            'improvement_recommendations',
            'strategic_next_steps'
        ]
        
        complete_sections = sum(1 for section in analysis_sections if response.get(section))
        score += (complete_sections / len(analysis_sections)) * 0.3
        
        # Score for actionable recommendations
        improvements = response.get('improvement_recommendations', {})
        if improvements and len(str(improvements)) > 100:
            score += 0.2
        
        # Score for strategic planning
        next_steps = response.get('strategic_next_steps', {})
        if next_steps and len(str(next_steps)) > 100:
            score += 0.15
        
        # Score for lessons learned quality
        lessons = response.get('lessons_learned', {})
        if lessons and len(str(lessons)) > 50:
            score += 0.1
        
        # Score for evaluation integration
        if evaluation_results.overall_score > 0.5:
            score += 0.05
        
        return min(score, 1.0)
    
    def _create_simple_reflection_summary(self, response: Dict[str, Any], evaluation_results: EvaluationResult, project) -> Dict[str, Any]:
        """Create simplified reflection summary."""
        summary = {
            "overall_success_score": 0.0,
            "execution_success": False,
            "key_achievements": [],
            "critical_issues": [],
            "next_steps": []
        }
        
        # Extract scores from response
        problem_assessment = response.get("problem_solution_assessment", {})
        summary["overall_success_score"] = problem_assessment.get("success_score", 5.0)
        
        # Check execution success
        if evaluation_results:
            summary["execution_success"] = evaluation_results.overall_score > 0.5
        
        # Extract key insights
        execution_results = response.get("execution_results", {})
        summary["key_achievements"] = execution_results.get("achievements", [])
        
        improvements = response.get("key_improvements", {})
        summary["critical_issues"] = improvements.get("critical_fixes", [])
        summary["next_steps"] = improvements.get("next_steps", [])
        
        return summary
    
    def _save_simple_reflection_artifacts(self, response: Dict[str, Any], reflection_summary: Dict[str, Any], state: PoCState) -> List[str]:
        """Save simplified reflection artifacts."""
        artifacts = []
        
        # Save reflection analysis
        reflection_path = self._save_json_artifact(
            {
                "reflection_analysis": response,
                "reflection_summary": reflection_summary,
                "analysis_timestamp": datetime.now().isoformat()
            },
            f"reflection_iteration_{state['iteration']}.json",
            state
        )
        artifacts.append(reflection_path)
        
        # Create simple report
        simple_report = self._create_simple_report(response, reflection_summary, state)
        report_path = self._save_artifact(
            simple_report,
            f"reflection_report_iteration_{state['iteration']}.md",
            state
        )
        artifacts.append(report_path)
        
        return artifacts
    
    def _create_simple_report(self, response: Dict[str, Any], reflection_summary: Dict[str, Any], state: PoCState) -> str:
        """Create simple reflection report."""
        report = f"""# PoC Reflection Report

**Project**: {state['project'].theme}  
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Success Score**: {reflection_summary.get('overall_success_score', 0.0):.1f}/10
- **Execution Success**: {reflection_summary.get('execution_success', False)}
- **Code Files Generated**: {len(state["agent_memory"].get("poc_implementer", {}).get("generated_code_files", []))}

## Problem Solution Assessment
{response.get('problem_solution_assessment', 'No assessment available')}

## Execution Results
{response.get('execution_results', 'No execution results available')}

## Key Improvements Needed
{response.get('key_improvements', 'No improvements identified')}

---
*Generated by AI-PoC-Agents-v2 Reflection Agent*
"""
        
        return report
    
    def _calculate_simple_reflection_score(self, response: Dict[str, Any], evaluation_results: EvaluationResult) -> float:
        """Calculate simplified reflection quality score."""
        score = 0.0
        
        # Base score for having response
        if response:
            score += 0.3
        
        # Score for each key section
        sections = ['problem_solution_assessment', 'execution_results', 'key_improvements']
        complete_sections = sum(1 for section in sections if response.get(section))
        score += (complete_sections / len(sections)) * 0.4
        
        # Score for execution success
        if evaluation_results and evaluation_results.overall_score > 0.5:
            score += 0.3
        
        return min(score, 1.0)
    
    def _log_reflection_summary(self, reflection_summary: Dict[str, Any], reflection_score: float) -> None:
        """Log simplified reflection summary with color formatting."""
        print("\033[95m" + "="*50)  # Magenta color
        print("🔍 PoC REFLECTION ANALYSIS")
        print("="*50)
        print(f"🎯 Overall Success: {reflection_summary.get('overall_success_score', 0.0):.1f}/10")
        print(f"🔧 Execution Success: {reflection_summary.get('execution_success', False)}")
        print(f"📊 Reflection Quality: {reflection_score:.3f}/1.0")
        
        print(f"\n🏆 Key Achievements:")
        for achievement in reflection_summary.get('key_achievements', [])[:3]:
            print(f"   ✅ {achievement}")
        
        print(f"\n🔧 Critical Issues:")
        for issue in reflection_summary.get('critical_issues', [])[:3]:
            print(f"   🛠️ {issue}")
        
        status = "🟢 SUCCESS" if reflection_summary.get('execution_success', False) else "🔴 NEEDS WORK"
        print(f"\n📈 Status: {status}")
        
        print("="*50 + "\033[0m")  # Reset color
    
    def _should_try_different_idea(self, overall_success: float, execution_success: bool, 
                                 response: Dict[str, Any], state: PoCState) -> bool:
        """Determine if a different idea should be tried based on reflection results."""
        
        # Check if we've already tried alternative ideas to prevent infinite loops
        idea_change_count = state.get("idea_change_count", 0)
        max_idea_changes = 2  # Maximum number of idea changes
        
        if idea_change_count >= max_idea_changes:
            return False
        
        # Criteria for needing idea change
        needs_change = False
        
        # 1. Very low overall success score
        if overall_success < 4.0:  # Less than 4/10
            needs_change = True
        
        # 2. Execution completely failed
        if not execution_success and overall_success < 6.0:
            needs_change = True
        
        # 3. Multiple critical issues identified
        critical_fixes = response.get('key_improvements', {}).get('critical_fixes', [])
        if len(critical_fixes) > 3 and overall_success < 5.0:
            needs_change = True
        
        # 4. Problem solution assessment indicates fundamental issues
        problem_assessment = response.get('problem_solution_assessment', {})
        if isinstance(problem_assessment, dict):
            solved_problem = problem_assessment.get('did_the_PoC_solve_the_original_problem', True)
            if not solved_problem and overall_success < 6.0:
                needs_change = True
        
        return needs_change
    
    def _get_idea_change_reasons(self, response: Dict[str, Any], overall_success: float, 
                               execution_success: bool) -> List[str]:
        """Get specific reasons for why idea change is needed."""
        reasons = []
        
        if overall_success < 4.0:
            reasons.append(f"Very low success score: {overall_success:.1f}/10")
        
        if not execution_success:
            reasons.append("Complete execution failure")
        
        critical_fixes = response.get('key_improvements', {}).get('critical_fixes', [])
        if len(critical_fixes) > 3:
            reasons.append(f"Too many critical issues: {len(critical_fixes)} identified")
        
        problem_assessment = response.get('problem_solution_assessment', {})
        if isinstance(problem_assessment, dict):
            solved_problem = problem_assessment.get('did_the_PoC_solve_the_original_problem', True)
            if not solved_problem:
                reasons.append("PoC failed to solve the original problem")
        
        return reasons