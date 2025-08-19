"""Idea Reflection Agent for AI-PoC-Agents-v2."""

from typing import Dict, Any, List
import json
from datetime import datetime

from ..agents.base_agent import BaseAgent
from ..core.state import PoCState, PoCIdea


class IdeaReflectionAgent(BaseAgent):
    """Agent responsible for evaluating ideas and assessing implementation feasibility."""
    
    def __init__(self, config):
        """Initialize IdeaReflectionAgent."""
        super().__init__("idea_reflection", config)
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        return """
You are an expert Idea Reflection and Evaluation Agent for PoC development.

Your responsibilities:
- IDEA EVALUATION: Critically assess each generated idea for feasibility and impact
- IMPLEMENTATION ASSESSMENT: Determine if ideas can be realistically implemented
- TECHNICAL VALIDATION: Validate technical approaches against available resources and constraints
- RISK ANALYSIS: Identify potential risks and blockers for each idea
- SELECTION PREPARATION: Rank and prepare ideas for final selection
- FEASIBILITY SCORING: Provide detailed feasibility analysis for implementation

Key principles:
- Apply rigorous technical evaluation criteria
- Consider real-world implementation constraints
- Assess resource requirements realistically
- Identify potential technical blockers early
- Balance innovation with practicality
- Provide constructive feedback for improvement
- Ensure ideas align with PoC success criteria

Focus on practical implementability while maintaining innovation potential.
"""
    
    def execute_phase(self, state: PoCState) -> Dict[str, Any]:
        """Execute idea reflection and feasibility evaluation."""
        
        project = state["project"]
        context = self._get_project_context(state)
        previous_results = self._get_previous_results(state)
        
        # Get ideas from previous phase
        ideas = state.get("ideas", [])
        if not ideas:
            return {
                "success": False,
                "score": 0.0,
                "output": {},
                "feedback": "No ideas found. Please run idea generation first.",
                "artifacts": [],
                "memory": {}
            }
        
        # Check for idea change feedback (alternative idea selection)
        idea_change_feedback = state.get("idea_change_feedback")
        is_idea_change_iteration = idea_change_feedback is not None
        
        # Get supporting context
        problem_analysis = state["agent_memory"].get("problem_identification", {}).get("problem_analysis", {})
        search_results = state["agent_memory"].get("search_problem", {}).get("search_results", {})
        
        # Convert ideas to dict format for evaluation
        ideas_data = [self._idea_to_dict(idea) for idea in ideas]
        
        # Perform detailed evaluation
        evaluation_results = self._evaluate_ideas(ideas_data, problem_analysis, search_results, project)
        
        # Update ideas with evaluation scores
        updated_ideas = self._update_ideas_with_evaluation(ideas, evaluation_results)
        
        # Rank ideas based on comprehensive scoring
        ranked_ideas = self._rank_ideas(updated_ideas, evaluation_results)
        
        # Prepare implementation readiness assessment
        implementation_readiness = self._assess_implementation_readiness(ranked_ideas, search_results)
        
        # Update state with evaluated ideas
        state["ideas"] = ranked_ideas
        state["idea_evaluation"] = evaluation_results
        
        # Select idea based on feedback or default behavior
        if ranked_ideas:
            if is_idea_change_iteration:
                # Try to select a different idea than the previous one
                selected_idea = self._select_alternative_idea(ranked_ideas, idea_change_feedback, state)
                self.logger.info(f"🔄 Selected alternative idea: {selected_idea.title} (score: {selected_idea.total_score:.2f})")
            else:
                # Default: select highest-scoring idea
                selected_idea = ranked_ideas[0]  # Top-ranked idea
                self.logger.info(f"🏆 Selected idea for implementation: {selected_idea.title} (score: {selected_idea.total_score:.2f})")
            
            state["selected_idea"] = selected_idea
        else:
            self.logger.warning("⚠️ No ideas available for selection")
        
        # Save evaluation results
        artifact_path = self._save_evaluation_results(evaluation_results, implementation_readiness, state)
        
        # Log evaluation summary
        self._log_evaluation_summary(ranked_ideas, evaluation_results)
        
        feedback = f"""
Idea Reflection Complete:
- Ideas Evaluated: {len(ideas)}
- Selected for Implementation: {ranked_ideas[0].title if ranked_ideas else 'None'} (score: {ranked_ideas[0].total_score:.2f} if ranked_ideas else 'N/A')
- Implementation Ready: {sum(1 for idea in ranked_ideas if getattr(idea, 'implementation_feasible', False))}
- High Risk Ideas: {sum(1 for idea in ranked_ideas if getattr(idea, 'risk_level', 'medium') == 'high')}
- Average Feasibility Score: {sum(idea.feasibility_score for idea in ranked_ideas) / len(ranked_ideas):.2f}
"""
        
        score = self._calculate_evaluation_score(evaluation_results, ranked_ideas)
        
        return {
            "success": True,
            "score": score,
            "output": {
                "evaluated_ideas": [self._idea_to_dict(idea) for idea in ranked_ideas],
                "evaluation_results": evaluation_results,
                "implementation_readiness": implementation_readiness
            },
            "feedback": feedback,
            "artifacts": [artifact_path],
            "memory": {
                "idea_evaluation": evaluation_results,
                "ranked_ideas": [self._idea_to_dict(idea) for idea in ranked_ideas],
                "implementation_readiness": implementation_readiness,
                "evaluated_at": datetime.now().isoformat()
            }
        }
    
    def _evaluate_ideas(
        self, 
        ideas_data: List[Dict], 
        problem_analysis: Dict[str, Any], 
        search_results: Dict[str, Any],
        project
    ) -> Dict[str, Any]:
        """Perform comprehensive evaluation of all ideas."""
        
        user_prompt = f"""
TASK: Evaluate PoC ideas for implementation feasibility.

PROJECT: {project.theme} | Task Type: {getattr(project, 'task_type', 'OTHER')} | Timeline: {project.timeline_days} days

PROBLEM: {problem_analysis.get('core_problem', 'OCR system development')}

AVAILABLE RESOURCES (from search results):
- Technical Approaches: {len(search_results.get('technical_approaches', []))}
- Sample Code Snippets: {len(search_results.get('sample_code_collection', []))}
- Implementation Patterns: {len(search_results.get('implementation_patterns', []))}
- Recommended Technologies: {search_results.get('recommended_technologies', [])}

IDEAS TO EVALUATE:
{json.dumps([{
    "id": idea["id"], 
    "title": idea["title"],
    "approach": idea["technical_approach"],
    "complexity": idea["implementation_complexity"],
    "score": idea["total_score"]
} for idea in ideas_data], indent=2, ensure_ascii=False)}

For each idea, evaluate:
1. FEASIBILITY SCORE (0.0-1.0) - Can this be implemented with available resources?
2. RISK LEVEL (low/medium/high) - Technical and timeline risks
3. KEY BLOCKERS - Main implementation challenges

Return ranking as list of objects with id and rank fields.
"""
        schema = {
            "evaluations": "list",  # One evaluation per idea with feasibility_score, risk_level, key_blockers
            "implementation_ranking": "list",  # Ideas ranked by feasibility: [{"id": "idea_2", "rank": 1}, ...]
            "recommendations": "list"  # Brief improvement suggestions
        }
        
        return self._generate_structured_response(
            self.get_system_prompt(),
            user_prompt,
            schema
        )
    
    def _update_ideas_with_evaluation(self, ideas: List[PoCIdea], evaluation_results: Dict[str, Any]) -> List[PoCIdea]:
        """Update ideas with evaluation scores and metadata."""
        evaluations = evaluation_results.get("evaluations", [])
        
        for idea in ideas:
            # Find corresponding evaluation
            idea_evaluation = None
            for eval_data in evaluations:
                if eval_data.get("idea_id") == idea.id or eval_data.get("title") == idea.title:
                    idea_evaluation = eval_data
                    break
            
            if idea_evaluation:
                # Update feasibility and risk scores
                idea.implementation_feasible = idea_evaluation.get("technical_feasibility", {}).get("implementation_readiness_score", 0.5) > 0.6
                idea.risk_level = idea_evaluation.get("risk_assessment", {}).get("risk_severity", "medium")
                idea.technical_risks = idea_evaluation.get("risk_assessment", {}).get("technical_risks", [])
                idea.resource_requirements = idea_evaluation.get("resource_requirements", {})
                idea.implementation_blockers = idea_evaluation.get("implementation_blockers", [])
                idea.improvement_recommendations = idea_evaluation.get("improvement_recommendations", [])
                
                # Update composite scores based on evaluation
                technical_feasibility = idea_evaluation.get("technical_feasibility", {}).get("implementation_readiness_score", 0.5)
                alignment_score = idea_evaluation.get("alignment_analysis", {}).get("alignment_score", 0.5)
                
                # Recalculate feasibility score with evaluation data
                idea.feasibility_score = (idea.feasibility_score + technical_feasibility + alignment_score) / 3.0
                
                # Adjust total score based on risk level
                risk_multiplier = {"low": 1.0, "medium": 0.9, "high": 0.7}.get(idea.risk_level, 0.8)
                idea.total_score = idea.total_score * risk_multiplier
        
        return ideas
    
    def _rank_ideas(self, ideas: List[PoCIdea], evaluation_results: Dict[str, Any]) -> List[PoCIdea]:
        """Rank ideas based on comprehensive evaluation."""
        # Get implementation ranking from evaluation if available
        implementation_ranking = evaluation_results.get("implementation_ranking", [])
        
        if implementation_ranking:
            # Sort ideas based on evaluation ranking
            try:
                if isinstance(implementation_ranking[0], dict) and "rank" in implementation_ranking[0]:
                    # New format: [{"id": "idea_2", "rank": 1}, ...]
                    ranking_map = {item["id"]: item["rank"] for item in implementation_ranking}
                    ideas.sort(key=lambda x: ranking_map.get(x.id, 999))
                elif isinstance(implementation_ranking[0], dict) and "id" in implementation_ranking[0]:
                    # Fallback: assume order is rank
                    ranking_map = {item["id"]: i+1 for i, item in enumerate(implementation_ranking)}
                    ideas.sort(key=lambda x: ranking_map.get(x.id, 999))
                else:
                    # Simple ID list format
                    ranking_map = {idea_id: i for i, idea_id in enumerate(implementation_ranking)}
                    ideas.sort(key=lambda x: ranking_map.get(x.id, len(ranking_map)))
            except (IndexError, KeyError, TypeError):
                # Fallback to score-based ranking if ranking parsing fails
                ideas.sort(key=lambda x: x.total_score, reverse=True)
        else:
            # Fallback to total score ranking
            ideas.sort(key=lambda x: x.total_score, reverse=True)
        
        return ideas
    
    def _assess_implementation_readiness(self, ideas: List[PoCIdea], search_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall implementation readiness."""
        ready_ideas = [idea for idea in ideas if getattr(idea, 'implementation_feasible', False)]
        high_risk_ideas = [idea for idea in ideas if getattr(idea, 'risk_level', 'medium') == 'high']
        
        # Check resource availability
        available_sample_code = len(search_results.get('sample_code_collection', []))
        available_patterns = len(search_results.get('implementation_patterns', []))
        
        return {
            "total_ideas": len(ideas),
            "implementation_ready": len(ready_ideas),
            "high_risk_ideas": len(high_risk_ideas),
            "readiness_percentage": len(ready_ideas) / len(ideas) * 100 if ideas else 0,
            "resource_support": {
                "sample_code_available": available_sample_code > 0,
                "implementation_patterns_available": available_patterns > 0,
                "technical_guidance_available": bool(search_results.get('best_practices'))
            },
            "recommended_for_implementation": [idea.id for idea in ready_ideas[:3]],
            "requires_further_research": [idea.id for idea in ideas if getattr(idea, 'risk_level', 'medium') == 'high']
        }
    
    def _idea_to_dict(self, idea: PoCIdea) -> Dict[str, Any]:
        """Convert PoCIdea object to dictionary with evaluation data."""
        base_dict = {
            "id": idea.id,
            "title": idea.title,
            "description": idea.description,
            "technical_approach": idea.approach,
            "implementation_complexity": idea.implementation_complexity,
            "expected_impact": idea.expected_impact,
            "innovation_score": idea.innovation_score,
            "feasibility_score": idea.feasibility_score,
            "total_score": idea.total_score,
            "estimated_effort_hours": idea.estimated_effort_hours
        }
        
        # Add evaluation-specific fields if available
        if hasattr(idea, 'implementation_feasible'):
            base_dict['implementation_feasible'] = idea.implementation_feasible
        if hasattr(idea, 'risk_level'):
            base_dict['risk_level'] = idea.risk_level
        if hasattr(idea, 'technical_risks'):
            base_dict['technical_risks'] = idea.technical_risks
        if hasattr(idea, 'resource_requirements'):
            base_dict['resource_requirements'] = idea.resource_requirements
        if hasattr(idea, 'implementation_blockers'):
            base_dict['implementation_blockers'] = idea.implementation_blockers
        if hasattr(idea, 'improvement_recommendations'):
            base_dict['improvement_recommendations'] = idea.improvement_recommendations
        
        return base_dict
    
    def _save_evaluation_results(
        self, 
        evaluation_results: Dict[str, Any], 
        implementation_readiness: Dict[str, Any], 
        state: PoCState
    ) -> str:
        """Save evaluation results as artifact."""
        comprehensive_results = {
            "evaluation_results": evaluation_results,
            "implementation_readiness": implementation_readiness,
            "evaluation_metadata": {
                "evaluation_timestamp": datetime.now().isoformat(),
                "evaluation_criteria": [
                    "technical_feasibility",
                    "resource_requirements", 
                    "alignment_analysis",
                    "risk_assessment",
                    "implementation_blockers"
                ]
            }
        }
        
        return self._save_json_artifact(
            comprehensive_results,
            f"idea_reflection_iteration_{state['iteration']}.json",
            state
        )
    
    def _log_evaluation_summary(self, ideas: List[PoCIdea], evaluation_results: Dict[str, Any]) -> None:
        """Log evaluation summary with color formatting."""
        print("\033[95m" + "="*60)  # Magenta color
        print("🤔 IDEA REFLECTION & EVALUATION RESULTS")
        print("="*60)
        print(f"📊 Ideas Evaluated: {len(ideas)}")
        
        ready_count = sum(1 for idea in ideas if getattr(idea, 'implementation_feasible', False))
        print(f"✅ Implementation Ready: {ready_count}/{len(ideas)}")
        
        high_risk_count = sum(1 for idea in ideas if getattr(idea, 'risk_level', 'medium') == 'high')
        print(f"⚠️ High Risk Ideas: {high_risk_count}/{len(ideas)}")
        
        if ideas:
            avg_feasibility = sum(idea.feasibility_score for idea in ideas) / len(ideas)
            print(f"📈 Average Feasibility: {avg_feasibility:.2f}")
            
            print(f"\n🏆 Top 3 Ideas (by evaluation):")
            for i, idea in enumerate(ideas[:3], 1):
                risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(getattr(idea, 'risk_level', 'medium'), "⚪")
                feasible_emoji = "✅" if getattr(idea, 'implementation_feasible', False) else "❌"
                print(f"   {i}. {idea.title}")
                print(f"      Score: {idea.total_score:.2f} | Feasible: {feasible_emoji} | Risk: {risk_emoji}")
        
        print("="*60 + "\033[0m")  # Reset color
    
    def _calculate_evaluation_score(self, evaluation_results: Dict[str, Any], ideas: List[PoCIdea]) -> float:
        """Calculate quality score for evaluation process."""
        if not ideas:
            return 0.0
        
        score = 0.0
        
        # Base score for completing evaluation
        score += 0.3
        
        # Score for having implementation-ready ideas
        ready_ideas = sum(1 for idea in ideas if getattr(idea, 'implementation_feasible', False))
        score += (ready_ideas / len(ideas)) * 0.4
        
        # Score for risk assessment quality
        if evaluation_results.get("risk_summary"):
            score += 0.2
        
        # Score for having actionable recommendations
        if evaluation_results.get("recommendations"):
            score += 0.1
        
        return min(score, 1.0)
    
    def _select_alternative_idea(self, ranked_ideas: List[PoCIdea], idea_change_feedback: Dict[str, Any], 
                               state: PoCState) -> PoCIdea:
        """Select an alternative idea different from the previous failed one."""
        
        # Get the previously selected idea
        previous_selected_idea = state.get("selected_idea")
        change_iteration = idea_change_feedback.get("change_iteration", 1)
        
        # For first change, try second-best idea
        # For second change, try third-best idea, etc.
        target_index = min(change_iteration, len(ranked_ideas) - 1)
        
        # If we have different ideas available
        if len(ranked_ideas) > target_index:
            selected_idea = ranked_ideas[target_index]
        else:
            # Fallback to the best available idea
            selected_idea = ranked_ideas[0]
        
        # Log the change reasoning
        change_reasons = idea_change_feedback.get("change_reasons", [])
        self.logger.info(f"Changing from previous idea due to: {'; '.join(change_reasons)}")
        
        return selected_idea