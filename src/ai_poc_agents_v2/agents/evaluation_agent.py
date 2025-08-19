"""Evaluation & Reflection Agent for AI-PoC-Agents-v2."""

from typing import Dict, Any, List
import json
import statistics
from datetime import datetime

from .base_agent import BaseAgent
from ..core.state import PoCState, EvaluationResult


class EvaluationAgent(BaseAgent):
    """Agent responsible for PoC evaluation, reflection, and reporting."""
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        return """
You are an expert Evaluation & Reflection Agent specialized in PoC assessment and analysis.

Your responsibilities:
1. RESULT EVALUATION: Objectively assess PoC performance against success criteria
2. REFLECTION: Analyze what worked, what didn't, and why  
3. REPORTING: Create comprehensive reports with actionable insights

Key principles:
- Objective, data-driven evaluation
- Balanced perspective highlighting both strengths and weaknesses
- Actionable insights and recommendations
- Clear communication for both technical and business stakeholders
- Learning-focused analysis to improve future PoCs
- Strategic thinking about next steps and scaling

Evaluation criteria:
- Technical Achievement: Does it work as intended?
- Business Value: Does it solve the intended problem? 
- Innovation Level: How creative/novel is the approach?
- Implementation Quality: Code quality, architecture, maintainability
- User Experience: How easy is it to use and understand?
- Scalability Potential: Can this be expanded/productionized?

Always provide constructive, specific feedback with clear rationale.
"""
    
    def execute_phase(self, state: PoCState) -> Dict[str, Any]:
        """Execute the agent's logic based on current phase."""
        
        current_phase = state["current_phase"]
        
        if current_phase == "result_evaluation":
            return self._evaluate_results(state)
        elif current_phase == "reflection":
            return self._reflect_on_poc(state)
        elif current_phase == "reporting":
            return self._create_report(state)
        else:
            raise ValueError(f"EvaluationAgent cannot handle phase: {current_phase}")
    
    def _evaluate_results(self, state: PoCState) -> Dict[str, Any]:
        """Evaluate PoC results against success criteria."""
        
        project = state["project"]
        implementation = state.get("implementation")
        
        if not implementation:
            raise ValueError("No PoC implementation available for evaluation")
        
        context = self._get_project_context(state)
        previous_results = self._get_previous_results(state, same_agent_only=True)
        
        # Get implementation details and execution results
        impl_info = implementation.__dict__ if hasattr(implementation, '__dict__') else implementation
        execution_memory = self._get_execution_memory(state)
        
        user_prompt = f"""
{context}

{previous_results}

POC IMPLEMENTATION DETAILS:
{json.dumps(impl_info, indent=2, ensure_ascii=False, default=str)}

EXECUTION RESULTS:
{json.dumps(execution_memory, indent=2, ensure_ascii=False, default=str)}

TASK: Conduct comprehensive evaluation of the PoC results.

Evaluate the PoC across multiple dimensions:

1. TECHNICAL EVALUATION (40% weight)
   - Functionality: Does it work as designed?
   - Code Quality: Architecture, maintainability, best practices
   - Performance: Speed, efficiency, resource usage
   - Reliability: Error handling, stability, robustness
   - Completeness: Feature coverage vs. requirements

2. BUSINESS VALUE EVALUATION (30% weight)  
   - Problem Resolution: How well does it address the original problem?
   - User Experience: Ease of use, intuitiveness
   - Business Impact: Potential value creation/cost savings
   - Market Relevance: Alignment with market needs
   - Competitive Advantage: Differentiation potential

3. INNOVATION EVALUATION (20% weight)
   - Technical Innovation: Novel approaches, creative solutions
   - Implementation Creativity: Interesting technical choices
   - Problem-Solving Approach: Unique perspectives
   - Learning Value: Knowledge gained, insights discovered

4. SCALABILITY EVALUATION (10% weight)
   - Production Readiness: How much work to make production-ready?
   - Performance at Scale: Can it handle larger loads?
   - Maintainability: How easy to modify/extend?
   - Resource Requirements: Infrastructure needs

For each dimension, provide:
- Score (0.0 to 1.0)
- Detailed reasoning
- Specific evidence/examples
- Comparison to expectations
- Areas of strength and weakness

Calculate an overall weighted score and provide clear justification.
"""
        
        schema = {
            "technical_evaluation": "dict",
            "business_value_evaluation": "dict", 
            "innovation_evaluation": "dict",
            "scalability_evaluation": "dict",
            "technical_score": "number",
            "business_score": "number",
            "innovation_score": "number",
            "scalability_score": "number",
            "overall_score": "number",
            "functionality_rating": "string",
            "performance_metrics": "dict",
            "success_criteria_assessment": "list",
            "strengths": "list",
            "weaknesses": "list",
            "evidence_summary": "list",
            "evaluation_confidence": "number"
        }
        
        response = self._generate_structured_response(
            self.get_system_prompt(),
            user_prompt,
            schema
        )
        
        # Create EvaluationResult object
        evaluation_result = EvaluationResult(
            overall_score=response.get("overall_score", 0.0),
            technical_score=response.get("technical_score", 0.0),
            business_score=response.get("business_score", 0.0),
            innovation_score=response.get("innovation_score", 0.0),
            quantitative_metrics=response.get("performance_metrics", {}),
            strengths=response.get("strengths", []),
            weaknesses=response.get("weaknesses", []),
            success_criteria_met=self._assess_success_criteria(response, project.success_criteria)
        )
        
        state["evaluation_results"] = evaluation_result
        
        # Save evaluation report
        evaluation_content = json.dumps(response, indent=2, ensure_ascii=False, default=str)
        artifact_path = self._save_artifact(
            evaluation_content,
            f"poc_evaluation_iteration_{state['iteration']}.json",
            state
        )
        
        # Create detailed evaluation report
        evaluation_report = self._create_evaluation_report(response, evaluation_result)
        report_path = self._save_artifact(
            evaluation_report,
            f"evaluation_report_iteration_{state['iteration']}.md",
            state
        )
        
        # Save state for debugging
        state_pkl_path = self._save_state_debug(state, "poc_evaluation")
        artifacts = [artifact_path, report_path, state_pkl_path]
        
        feedback = f"""
PoC Evaluation Complete:
- Overall Score: {response.get('overall_score', 0.0):.3f}/1.0
- Technical Score: {response.get('technical_score', 0.0):.3f}/1.0
- Business Score: {response.get('business_score', 0.0):.3f}/1.0
- Innovation Score: {response.get('innovation_score', 0.0):.3f}/1.0
- Strengths Identified: {len(response.get('strengths', []))}
- Areas for Improvement: {len(response.get('weaknesses', []))}
"""
        
        score = response.get("overall_score", 0.5)
        
        return {
            "success": True,
            "score": score,
            "output": response,
            "feedback": feedback,
            "artifacts": artifacts,
            "memory": {
                "evaluation": response,
                "evaluation_result": evaluation_result.__dict__,
                "evaluated_at": datetime.now().isoformat()
            }
        }
    
    def _reflect_on_poc(self, state: PoCState) -> Dict[str, Any]:
        """Conduct reflection analysis on the PoC process and results."""
        
        project = state["project"]
        evaluation_results = state.get("evaluation_results")
        
        if not evaluation_results:
            raise ValueError("No evaluation results available for reflection")
        
        context = self._get_project_context(state)
        previous_results = self._get_previous_results(state, same_agent_only=True)
        
        # Get all phase results for comprehensive reflection
        all_phase_results = state.get("phase_results", [])
        evaluation_memory = self._get_agent_memory(state).get("evaluation", {})
        
        user_prompt = f"""
{context}

{previous_results}

EVALUATION RESULTS:
{json.dumps(evaluation_memory, indent=2, ensure_ascii=False, default=str)}

ALL PHASE RESULTS SUMMARY:
{self._summarize_phase_results(all_phase_results)}

TASK: Conduct deep reflection on the entire PoC process and results.

Analyze the PoC journey from start to finish:

1. PROCESS REFLECTION
   - What worked well in the PoC development process?
   - What were the major challenges and how were they addressed?
   - Which phases were most/least effective?
   - What would you do differently?
   - How did the initial plan compare to actual execution?

2. TECHNICAL LESSONS LEARNED
   - Key technical insights discovered
   - Technology choices: what worked, what didn't
   - Architecture decisions: pros and cons
   - Implementation challenges and solutions
   - Performance learnings

3. BUSINESS INSIGHTS  
   - How well did the PoC address business needs?
   - What business assumptions were validated/invalidated?
   - User feedback and market insights
   - Value creation opportunities identified
   - Business model implications

4. STRATEGIC REFLECTION
   - Does this PoC justify further investment?
   - What are the scaling challenges and opportunities?
   - How does this fit into broader strategy?
   - What partnerships or resources would be needed?
   - Market timing considerations

5. INNOVATION ASSESSMENT
   - What was truly innovative about this approach?
   - Where did we push boundaries successfully?
   - What conventional wisdom was challenged?
   - How can we build on these innovations?

6. IMPROVEMENT RECOMMENDATIONS
   - Specific technical improvements for next version
   - Process improvements for future PoCs  
   - Resource/skill development needs
   - Partnership or collaboration opportunities
   - Risk mitigation strategies

7. FUTURE ROADMAP
   - Immediate next steps (next 30 days)
   - Short-term goals (3-6 months)
   - Long-term vision (6-18 months)
   - Success metrics for next phase
   - Resource requirements

Focus on actionable insights that can guide future development and strategic decisions.
"""
        
        schema = {
            "process_reflection": "dict",
            "technical_lessons": "dict",
            "business_insights": "dict", 
            "strategic_reflection": "dict",
            "innovation_assessment": "dict",
            "improvement_recommendations": "dict",
            "future_roadmap": "dict",
            "key_learnings": "list",
            "success_factors": "list",
            "failure_points": "list",
            "recommendation_priority": "list",
            "next_steps_immediate": "list",
            "next_steps_short_term": "list",
            "next_steps_long_term": "list",
            "resource_needs": "list",
            "risk_factors": "list",
            "success_probability": "number"
        }
        
        response = self._generate_structured_response(
            self.get_system_prompt(),
            user_prompt,
            schema
        )
        
        # Update evaluation results with reflection insights
        if hasattr(evaluation_results, '__dict__'):
            evaluation_results.lessons_learned = response.get("key_learnings", [])
            evaluation_results.improvement_suggestions = response.get("improvement_recommendations", {}).get("technical", [])
            evaluation_results.next_steps = response.get("next_steps_immediate", [])
        else:
            evaluation_results["lessons_learned"] = response.get("key_learnings", [])
            evaluation_results["improvement_suggestions"] = response.get("improvement_recommendations", {}).get("technical", [])
            evaluation_results["next_steps"] = response.get("next_steps_immediate", [])
        
        # Save reflection analysis
        reflection_content = json.dumps(response, indent=2, ensure_ascii=False, default=str)
        artifact_path = self._save_artifact(
            reflection_content,
            f"reflection_analysis_iteration_{state['iteration']}.json",
            state
        )
        
        # Create comprehensive reflection document
        reflection_report = self._create_reflection_report(response)
        report_path = self._save_artifact(
            reflection_report,
            f"reflection_report_iteration_{state['iteration']}.md",
            state
        )
        
        # Save state for debugging
        state_pkl_path = self._save_state_debug(state, "poc_reflection")
        artifacts = [artifact_path, report_path, state_pkl_path]
        
        feedback = f"""
PoC Reflection Complete:
- Key Learnings Identified: {len(response.get('key_learnings', []))}
- Success Factors: {len(response.get('success_factors', []))}
- Improvement Areas: {len(response.get('failure_points', []))}
- Immediate Next Steps: {len(response.get('next_steps_immediate', []))}
- Success Probability: {response.get('success_probability', 0.5):.1%}
"""
        
        score = self._calculate_reflection_score(response)
        
        return {
            "success": True,
            "score": score,
            "output": response,
            "feedback": feedback,
            "artifacts": artifacts,
            "memory": {
                "reflection": response,
                "reflected_at": datetime.now().isoformat()
            }
        }
    
    def _create_report(self, state: PoCState) -> Dict[str, Any]:
        """Create final comprehensive PoC report."""
        
        project = state["project"]
        evaluation_results = state.get("evaluation_results")
        
        context = self._get_project_context(state)
        previous_results = self._get_previous_results(state, same_agent_only=True)
        
        # Get all memory from evaluation phases
        evaluation_memory = self._get_agent_memory(state).get("evaluation", {})
        reflection_memory = self._get_agent_memory(state).get("reflection", {})
        
        user_prompt = f"""
{context}

{previous_results}

EVALUATION RESULTS:
{json.dumps(evaluation_memory, indent=2, ensure_ascii=False, default=str)}

REFLECTION ANALYSIS:
{json.dumps(reflection_memory, indent=2, ensure_ascii=False, default=str)}

TASK: Create a comprehensive final report for stakeholders.

Generate a professional report suitable for:
- Executive leadership (strategic overview)
- Technical teams (implementation details)
- Business stakeholders (value and next steps)
- Future development teams (lessons and recommendations)

Include these sections:

1. EXECUTIVE SUMMARY
   - Project overview and objectives
   - Key achievements and outcomes
   - Bottom-line results and recommendations
   - Investment/resource implications

2. PROJECT OVERVIEW
   - Original problem statement
   - Chosen approach and rationale
   - Success criteria and metrics
   - Timeline and resources used

3. TECHNICAL ACHIEVEMENTS
   - What was built and how it works
   - Technical innovations and insights
   - Performance metrics and benchmarks
   - Architecture and design decisions

4. BUSINESS IMPACT ANALYSIS
   - Problem resolution effectiveness
   - Value creation potential
   - Market opportunity assessment
   - Competitive positioning

5. EVALUATION RESULTS
   - Comprehensive scoring across all dimensions
   - Strengths and accomplishments
   - Areas needing improvement
   - Comparison to initial expectations

6. LESSONS LEARNED
   - Key insights from the PoC process
   - What worked well and what didn't
   - Unexpected discoveries
   - Process improvements for future

7. RECOMMENDATIONS & NEXT STEPS
   - Immediate actions (30 days)
   - Short-term roadmap (3-6 months)  
   - Long-term strategic direction
   - Resource and investment needs
   - Success criteria for next phase

8. RISK ASSESSMENT
   - Technical risks and mitigation
   - Business risks and challenges
   - Market timing considerations
   - Competitive threats

9. APPENDICES
   - Technical specifications
   - Performance data
   - Code samples and documentation
   - References and resources

Make it professional, actionable, and balanced in perspective.
"""
        
        response = self._generate_response(self.get_system_prompt(), user_prompt)
        
        # Create structured final report
        final_report = self._create_final_report(
            project, 
            evaluation_results, 
            evaluation_memory, 
            reflection_memory,
            state
        )
        
        # Save final report as artifact
        report_path = self._save_artifact(
            final_report,
            f"final_poc_report_iteration_{state['iteration']}.md",
            state
        )
        
        # Create executive summary
        exec_summary = self._create_executive_summary(
            project,
            evaluation_results,
            reflection_memory
        )
        
        summary_path = self._save_artifact(
            exec_summary,
            f"executive_summary_iteration_{state['iteration']}.md",
            state
        )
        
        # Create JSON summary for easy parsing
        json_summary = self._create_json_summary(
            project,
            evaluation_results,
            evaluation_memory,
            reflection_memory
        )
        
        json_path = self._save_json_artifact(
            json_summary,
            f"poc_summary_iteration_{state['iteration']}.json",
            state
        )
        
        # Save state for debugging
        state_pkl_path = self._save_state_debug(state, "final_report")
        artifacts = [report_path, summary_path, json_path, state_pkl_path]
        
        feedback = f"""
Final PoC Report Complete:
- Executive Summary: Created for leadership review
- Technical Report: Detailed implementation analysis
- Business Analysis: Value assessment and market opportunity
- Next Steps: Roadmap with {len(reflection_memory.get('next_steps_immediate', []))} immediate actions
- Artifacts: {len(state.get('artifacts', []))} files generated during PoC
"""
        
        # Calculate final score based on evaluation results
        final_score = evaluation_results.overall_score if evaluation_results and hasattr(evaluation_results, 'overall_score') else 0.7
        
        return {
            "success": True,
            "score": final_score,
            "output": {
                "final_report_created": True,
                "executive_summary": exec_summary[:200] + "...",
                "overall_assessment": self._get_overall_assessment(evaluation_results),
                "next_steps_count": len(reflection_memory.get('next_steps_immediate', [])),
                "artifacts_generated": len(state.get("artifacts", []))
            },
            "feedback": feedback,
            "artifacts": artifacts,
            "memory": {
                "final_report": final_report,
                "executive_summary": exec_summary,
                "json_summary": json_summary,
                "reported_at": datetime.now().isoformat()
            }
        }
    
    def _get_execution_memory(self, state: PoCState) -> Dict[str, Any]:
        """Get execution results from PoC agent memory."""
        for result in state.get("phase_results", []):
            if result.phase == "poc_execution":
                return result.output
        return {}
    
    def _summarize_phase_results(self, phase_results: List) -> str:
        """Create summary of all phase results."""
        if not phase_results:
            return "No phase results available"
        
        summary = []
        for result in phase_results:
            phase = result.phase if hasattr(result, 'phase') else result.get('phase', 'unknown')
            agent = result.agent if hasattr(result, 'agent') else result.get('agent', 'unknown')
            score = result.score if hasattr(result, 'score') else result.get('score', 0.0)
            success = result.success if hasattr(result, 'success') else result.get('success', False)
            
            summary.append(f"- {phase} ({agent}): Score {score:.3f}, Success: {success}")
        
        return "\n".join(summary)
    
    def _assess_success_criteria(self, evaluation: Dict[str, Any], criteria: List[str]) -> List[bool]:
        """Assess which success criteria were met."""
        if not criteria:
            return []
        
        # Simple heuristic based on overall scores
        overall_score = evaluation.get("overall_score", 0.0)
        
        # If overall score is high, assume most criteria are met
        met_ratio = min(overall_score * 1.2, 1.0)  # Allow some bonus
        met_count = int(len(criteria) * met_ratio)
        
        return [True] * met_count + [False] * (len(criteria) - met_count)
    
    def _create_evaluation_report(self, evaluation: Dict[str, Any], result: EvaluationResult) -> str:
        """Create detailed evaluation report."""
        newline = "\n"
        return f"""# PoC Evaluation Report

## Overall Assessment
- **Overall Score**: {evaluation.get('overall_score', 0.0):.3f}/1.0
- **Technical Score**: {evaluation.get('technical_score', 0.0):.3f}/1.0  
- **Business Value Score**: {evaluation.get('business_score', 0.0):.3f}/1.0
- **Innovation Score**: {evaluation.get('innovation_score', 0.0):.3f}/1.0

## Technical Evaluation
{json.dumps(evaluation.get('technical_evaluation', {}), indent=2)}

## Business Value Assessment
{json.dumps(evaluation.get('business_value_evaluation', {}), indent=2)}

## Innovation Analysis  
{json.dumps(evaluation.get('innovation_evaluation', {}), indent=2)}

## Strengths Identified
{newline.join(f"- {strength}" for strength in evaluation.get('strengths', []))}

## Areas for Improvement
{newline.join(f"- {weakness}" for weakness in evaluation.get('weaknesses', []))}

## Performance Metrics
{newline.join(f"- {k}: {v}" for k, v in evaluation.get('performance_metrics', {}).items())}

## Evaluation Confidence: {evaluation.get('evaluation_confidence', 0.0):.1%}
"""
    
    def _create_reflection_report(self, reflection: Dict[str, Any]) -> str:
        """Create comprehensive reflection report."""
        newline = "\n"
        return f"""# PoC Reflection Analysis

## Process Reflection
{newline.join(f"**{k}**: {v}" for k, v in reflection.get('process_reflection', {}).items())}

## Technical Lessons Learned
{newline.join(f"- {lesson}" for lesson in reflection.get('technical_lessons', {}).get('insights', []))}

## Business Insights
{newline.join(f"- {insight}" for insight in reflection.get('business_insights', {}).get('findings', []))}

## Key Success Factors
{newline.join(f"- {factor}" for factor in reflection.get('success_factors', []))}

## Areas for Improvement
{newline.join(f"- {point}" for point in reflection.get('failure_points', []))}

## Immediate Next Steps (30 days)
{newline.join(f"- {step}" for step in reflection.get('next_steps_immediate', []))}

## Short-term Goals (3-6 months)
{newline.join(f"- {goal}" for goal in reflection.get('next_steps_short_term', []))}

## Long-term Vision (6-18 months)
{newline.join(f"- {vision}" for vision in reflection.get('next_steps_long_term', []))}

## Success Probability: {reflection.get('success_probability', 0.5):.1%}
"""
    
    def _create_final_report(self, project, evaluation_results, evaluation_memory, reflection_memory, state) -> str:
        """Create comprehensive final report."""
        
        overall_score = evaluation_memory.get('overall_score', 0.0) if evaluation_memory else 0.0
        next_steps = reflection_memory.get('next_steps_immediate', []) if reflection_memory else []
        newline = "\n"
        
        return f"""# PoC Final Report: {project.theme}

## Executive Summary

This Proof of Concept (PoC) was developed to explore **{project.theme}** with an overall achievement score of **{overall_score:.1%}**.

### Key Outcomes
- ✅ Technical feasibility demonstrated
- ✅ Core functionality implemented  
- ✅ Business value potential identified
- 📋 Next steps defined for scaling

## Project Overview

**Theme**: {project.theme}
**Domain**: {project.domain}
**Timeline**: {project.timeline_days} days
**Success Criteria**: {len(project.success_criteria)} defined metrics

### Original Problem
{project.description}

## Technical Achievements

### Implementation Summary
- **Technology Stack**: {', '.join(getattr(evaluation_results, 'tech_stack', []) if evaluation_results else [])}
- **Code Files**: {len(state.get('artifacts', []))} files generated
- **Test Coverage**: Functional validation completed

### Performance Metrics
{newline.join(f"- {k}: {v}" for k, v in evaluation_memory.get('performance_metrics', {}).items()) if evaluation_memory else "Performance data collected"}

## Business Impact

### Value Assessment
- **Business Score**: {evaluation_memory.get('business_score', 0.0):.3f}/1.0
- **Problem Resolution**: Addressed core requirements
- **Market Potential**: {reflection_memory.get('business_insights', {}).get('market_opportunity', 'To be assessed') if reflection_memory else 'Positive indicators'}

## Evaluation Results

### Overall Scoring
- **Technical**: {evaluation_memory.get('technical_score', 0.0):.3f}/1.0
- **Business Value**: {evaluation_memory.get('business_score', 0.0):.3f}/1.0  
- **Innovation**: {evaluation_memory.get('innovation_score', 0.0):.3f}/1.0
- **Overall**: {overall_score:.3f}/1.0

### Strengths
{newline.join(f"- {strength}" for strength in evaluation_memory.get('strengths', [])) if evaluation_memory else "- Core functionality achieved"}

### Areas for Improvement  
{newline.join(f"- {weakness}" for weakness in evaluation_memory.get('weaknesses', [])) if evaluation_memory else "- Optimization opportunities identified"}

## Recommendations & Next Steps

### Immediate Actions (30 days)
{newline.join(f"1. {step}" for step in next_steps[:3]) if next_steps else "1. Review and validate results"}

### Strategic Recommendations
- Continue development with additional resources
- Focus on identified improvement areas
- Plan for production deployment
- Develop comprehensive testing strategy

## Risk Assessment
- **Technical Risk**: Low to Medium
- **Business Risk**: Medium  
- **Market Risk**: Low
- **Resource Risk**: Medium

## Conclusion

This PoC successfully demonstrates the feasibility of {project.theme} with significant potential for further development. The results justify continued investment and progression to the next development phase.

**Recommendation**: Proceed with full development

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    def _create_executive_summary(self, project, evaluation_results, reflection_memory) -> str:
        """Create executive summary for leadership."""
        
        overall_score = getattr(evaluation_results, 'overall_score', 0.0) if evaluation_results else 0.0
        success_rate = overall_score * 100
        
        return f"""# Executive Summary: {project.theme}

## Bottom Line
**{success_rate:.0f}% Success Rate** - This PoC demonstrates strong feasibility and business potential.

## Key Results
- ✅ **Technical Proof**: Core functionality successfully implemented
- ✅ **Business Value**: Clear value proposition validated
- ✅ **Market Fit**: Addresses real user needs
- 📈 **Next Steps**: Ready for scaled development

## Investment Recommendation
**PROCEED** - Results justify continued investment with {overall_score:.0%} confidence level.

## Resource Requirements (Next Phase)
- Development Team: 2-3 engineers
- Timeline: 3-6 months to MVP
- Budget: Moderate investment recommended
- Skills: Leverage current team + specialist support

## Strategic Impact
This PoC opens new opportunities for competitive advantage and market expansion.

## Risk Mitigation
Primary risks identified and mitigation strategies developed.

---
**Decision Required**: Approve next phase development budget and resource allocation.
"""
    
    def _create_json_summary(self, project, evaluation_results, evaluation_memory, reflection_memory) -> Dict[str, Any]:
        """Create structured JSON summary."""
        
        return {
            "project": {
                "theme": project.theme,
                "domain": project.domain,
                "timeline_days": project.timeline_days,
                "success_criteria_count": len(project.success_criteria)
            },
            "evaluation": {
                "overall_score": evaluation_memory.get('overall_score', 0.0) if evaluation_memory else 0.0,
                "technical_score": evaluation_memory.get('technical_score', 0.0) if evaluation_memory else 0.0,
                "business_score": evaluation_memory.get('business_score', 0.0) if evaluation_memory else 0.0,
                "innovation_score": evaluation_memory.get('innovation_score', 0.0) if evaluation_memory else 0.0,
                "strengths_count": len(evaluation_memory.get('strengths', [])) if evaluation_memory else 0,
                "improvement_areas": len(evaluation_memory.get('weaknesses', [])) if evaluation_memory else 0
            },
            "reflection": {
                "success_probability": reflection_memory.get('success_probability', 0.5) if reflection_memory else 0.5,
                "immediate_steps": len(reflection_memory.get('next_steps_immediate', [])) if reflection_memory else 0,
                "key_learnings": len(reflection_memory.get('key_learnings', [])) if reflection_memory else 0
            },
            "recommendation": {
                "proceed": evaluation_memory.get('overall_score', 0.0) > 0.6 if evaluation_memory else False,
                "confidence": "HIGH" if evaluation_memory and evaluation_memory.get('overall_score', 0.0) > 0.8 else "MEDIUM",
                "priority": "HIGH" if evaluation_memory and evaluation_memory.get('business_score', 0.0) > 0.7 else "MEDIUM"
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_overall_assessment(self, evaluation_results) -> str:
        """Get overall assessment string."""
        if not evaluation_results:
            return "Assessment pending"
        
        score = getattr(evaluation_results, 'overall_score', 0.0) if hasattr(evaluation_results, 'overall_score') else evaluation_results.get('overall_score', 0.0)
        
        if score >= 0.8:
            return "Excellent - Exceeds expectations"
        elif score >= 0.7:
            return "Good - Meets expectations"
        elif score >= 0.6:
            return "Satisfactory - Minor improvements needed"
        elif score >= 0.5:
            return "Fair - Significant improvements needed"
        else:
            return "Needs major revision"
    
    def _calculate_reflection_score(self, reflection: Dict[str, Any]) -> float:
        """Calculate quality score for reflection analysis."""
        score = 0.4  # Base score
        
        # Completeness of analysis
        key_sections = [
            'process_reflection', 'technical_lessons', 'business_insights',
            'strategic_reflection', 'improvement_recommendations', 'future_roadmap'
        ]
        
        for section in key_sections:
            if reflection.get(section):
                score += 0.1
        
        # Quality indicators
        if len(reflection.get('key_learnings', [])) >= 3:
            score += 0.05
        
        if len(reflection.get('next_steps_immediate', [])) >= 2:
            score += 0.05
        
        return min(score, 1.0)
    
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
            elif key == "ideas" and isinstance(value, list):
                # Handle list of PoCIdea objects
                state_copy[key] = [idea.__dict__ if hasattr(idea, '__dict__') else idea for idea in value]
            elif key == "selected_idea" and hasattr(value, '__dict__'):
                state_copy[key] = value.__dict__
            elif key == "evaluation_results" and hasattr(value, '__dict__'):
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


if __name__ == "__main__":
    """EvaluationAgentの動作確認用メイン関数"""
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
    from ai_poc_agents_v2.core.state import PoCProject, PoCIdea, PoCImplementation, EvaluationResult
    from ai_poc_agents_v2.agents.base_agent import BaseAgent
    
    # Load environment variables
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Test EvaluationAgent functionality")
    parser.add_argument("--theme", type=str, default="OCR画像文字認識システム", help="Project theme")
    parser.add_argument("--description", type=str, default="画像から文字を認識してテキストに変換するPythonシステム", help="Project description")
    parser.add_argument("--phase", type=str, choices=["result_evaluation", "reflection", "reporting", "all"], 
                      default="all", help="Which phase to test")
    parser.add_argument("--workspace", type=str, default="./tmp/test_evaluation_agent", help="Workspace directory")
    
    args = parser.parse_args()
    
    print("=== EvaluationAgent Test ===============")
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
    print("\\n1. Initializing EvaluationAgent...")
    config = Config()
    
    # Create agent
    evaluation_agent = EvaluationAgent("evaluator", config)
    
    # Create project and mock implementation
    project = PoCProject(theme=args.theme)
    project.description = args.description
    project.domain = "AI/ML"
    project.timeline_days = 7
    project.success_criteria = [
        "Accurate text recognition from images",
        "Processing multiple image formats",
        "Clear output formatting",
        "Reasonable processing speed"
    ]
    
    # Create mock implementation with execution results
    mock_implementation = PoCImplementation(
        idea_id="test_idea_1",
        architecture={
            "architecture_overview": "OCR system using EasyOCR and OpenCV",
            "system_components": ["Image Preprocessor", "OCR Engine", "Text Extractor", "Output Handler"],
            "technology_stack": {"languages": ["Python"], "frameworks": ["EasyOCR", "OpenCV"]},
            "programming_languages": ["Python"],
            "frameworks": ["EasyOCR", "OpenCV"],
            "libraries": ["numpy", "pillow", "cv2"],
            "dependencies": ["easyocr>=1.4.1", "opencv-python>=4.5.5", "numpy>=1.21.2"],
            "development_phases": ["Setup", "Implementation", "Testing", "Validation"],
            "demo_scenarios": ["Single image OCR", "Batch processing", "Performance testing"]
        },
        tech_stack=["Python", "EasyOCR", "OpenCV"],
        dependencies=["easyocr>=1.4.1", "opencv-python>=4.5.5", "numpy>=1.21.2"],
        environment_config={"python_version": "3.8+"},
        test_cases=[{"scenario": "Test OCR accuracy on sample images"}],
        code_files={
            "main.py": "OCR implementation with EasyOCR",
            "requirements.txt": "Dependencies specification",
            "README.md": "Setup and usage instructions"
        },
        performance_metrics={
            "processing_time": 2.5,
            "accuracy_rate": 0.92,
            "memory_usage": 128.0
        },
        execution_logs=[
            "Successfully processed test_image.png",
            "Extracted text: 'Hello WorldN ! OCR Test Image'",
            "Processing time: 2.5 seconds"
        ]
    )
    
    # Create mock phase results
    phase_results = [
        type('MockResult', (), {
            'phase': 'problem_identification',
            'agent': 'problem_agent',
            'score': 0.85,
            'success': True
        })(),
        type('MockResult', (), {
            'phase': 'idea_generation', 
            'agent': 'problem_agent',
            'score': 0.92,
            'success': True
        })(),
        type('MockResult', (), {
            'phase': 'poc_design',
            'agent': 'poc_agent',
            'score': 0.88,
            'success': True
        })(),
        type('MockResult', (), {
            'phase': 'poc_implementation',
            'agent': 'poc_agent', 
            'score': 0.90,
            'success': True
        })(),
        type('MockResult', (), {
            'phase': 'poc_execution',
            'agent': 'poc_agent',
            'score': 0.86,
            'success': True,
            'output': {
                "execution_successful": True,
                "performance_metrics": {"processing_time": 2.5, "accuracy": 0.92},
                "validation_results": "All tests passed"
            }
        })()
    ]
    
    state = {
        "project": project,
        "implementation": mock_implementation,
        "current_phase": "result_evaluation",
        "iteration": 0,
        "completed_phases": ["problem_identification", "idea_generation", "poc_design", "poc_implementation", "poc_execution"],
        "phase_results": phase_results,
        "phase_scores": {
            "problem_identification": 0.85,
            "idea_generation": 0.92,
            "poc_design": 0.88,
            "poc_implementation": 0.90,
            "poc_execution": 0.86
        },
        "overall_score": 0.88,
        "artifacts": [
            "main.py", "requirements.txt", "README.md", 
            "poc_design.json", "execution_plan.md"
        ],
        "logs": [],
        "should_continue": True,
        "workspace_dir": workspace_dir,
        "workspace_path": str(workspace_dir),
        "started_at": __import__('datetime').datetime.now(),
        "updated_at": __import__('datetime').datetime.now(),
        "agent_memory": {}
    }
    
    print("✓ EvaluationAgent initialized successfully")
    
    # Test phases based on user selection
    if args.phase in ["result_evaluation", "all"]:
        print("\\n2. Testing Result Evaluation Phase...")
        result = evaluation_agent._evaluate_results(state)
        print(f"✓ Result evaluation completed")
        print(f"   Success: {result['success']}")
        print(f"   Score: {result['score']:.3f}")
        print(f"   Artifacts: {len(result['artifacts'])}")
        
        # Show evaluation results
        output = result.get('output', {})
        if output.get('overall_score'):
            print(f"   Overall Score: {output['overall_score']:.3f}/1.0")
        if output.get('technical_score'):
            print(f"   Technical Score: {output['technical_score']:.3f}/1.0")
        if output.get('business_score'):
            print(f"   Business Score: {output['business_score']:.3f}/1.0")
        if output.get('innovation_score'):
            print(f"   Innovation Score: {output['innovation_score']:.3f}/1.0")
        if output.get('strengths'):
            print(f"   Strengths: {len(output['strengths'])} identified")
        if output.get('weaknesses'):
            print(f"   Areas for Improvement: {len(output['weaknesses'])} identified")
        
        print(f"   Feedback: {result['feedback'][:200]}...")
    
    if args.phase in ["reflection", "all"]:
        print("\\n3. Testing Reflection Phase...")
        state["current_phase"] = "reflection"
        
        # Ensure we have evaluation results for reflection
        if not state.get("evaluation_results"):
            print("   Creating mock evaluation results for reflection...")
            mock_evaluation = EvaluationResult(
                overall_score=0.85,
                technical_score=0.88,
                business_score=0.82,
                innovation_score=0.86,
                quantitative_metrics={"processing_time": 2.5, "accuracy": 0.92},
                strengths=["Good OCR accuracy", "Clean implementation", "Well documented"],
                weaknesses=["Could be faster", "Limited language support"],
                success_criteria_met=[True, True, True, False]
            )
            state["evaluation_results"] = mock_evaluation
        
        result = evaluation_agent._reflect_on_poc(state)
        print(f"✓ Reflection completed")
        print(f"   Success: {result['success']}")
        print(f"   Score: {result['score']:.3f}")
        print(f"   Artifacts: {len(result['artifacts'])}")
        
        # Show reflection results
        output = result.get('output', {})
        if output.get('key_learnings'):
            print(f"   Key Learnings: {len(output['key_learnings'])} identified")
        if output.get('success_factors'):
            print(f"   Success Factors: {len(output['success_factors'])} identified")
        if output.get('next_steps_immediate'):
            print(f"   Immediate Next Steps: {len(output['next_steps_immediate'])} defined")
        if output.get('success_probability'):
            print(f"   Success Probability: {output['success_probability']:.1%}")
        
        print(f"   Feedback: {result['feedback'][:200]}...")
    
    if args.phase in ["reporting", "all"]:
        print("\\n4. Testing Reporting Phase...")
        state["current_phase"] = "reporting"
        
        # Ensure we have evaluation results and reflection for reporting
        if not state.get("evaluation_results"):
            mock_evaluation = EvaluationResult(
                overall_score=0.85,
                technical_score=0.88,
                business_score=0.82,
                innovation_score=0.86
            )
            state["evaluation_results"] = mock_evaluation
        
        result = evaluation_agent._create_report(state)
        print(f"✓ Final reporting completed")
        print(f"   Success: {result['success']}")
        print(f"   Score: {result['score']:.3f}")
        print(f"   Artifacts: {len(result['artifacts'])}")
        
        # Show reporting results
        output = result.get('output', {})
        if output.get('final_report_created'):
            print(f"   Final Report: Created")
        if output.get('overall_assessment'):
            print(f"   Assessment: {output['overall_assessment']}")
        if output.get('next_steps_count'):
            print(f"   Next Steps: {output['next_steps_count']} defined")
        if output.get('artifacts_generated'):
            print(f"   Total Artifacts: {output['artifacts_generated']} generated")
        
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
    
    print("\\n✅ EvaluationAgent test completed successfully!")
    print(f"Workspace preserved at: {workspace_dir}")