"""Idea Generation Agent for AI-PoC-Agents-v2."""

from typing import Dict, Any, List
import json
from datetime import datetime

from ..agents.base_agent import BaseAgent
from ..core.state import PoCState, PoCIdea


class IdeaGenerationAgent(BaseAgent):
    """Agent responsible for generating multiple PoC ideas based on problem analysis and search results."""
    
    def __init__(self, config):
        """Initialize IdeaGenerationAgent."""
        super().__init__("idea_generation", config)
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        return """
You are an expert Idea Generation Agent for PoC development.

Your responsibilities:
- IDEA GENERATION: Create multiple innovative and feasible PoC approaches
- TECHNICAL INTEGRATION: Leverage search results, sample code, and technical insights
- DIVERSITY CREATION: Generate ideas with different complexity levels and approaches
- FEASIBILITY CONSIDERATION: Balance innovation with practical implementation constraints
- STRUCTURED OUTPUT: Provide detailed specifications for each idea

Key principles:
- Generate 3-5 distinct approaches with varying complexity
- Leverage technical research and sample code from search results
- Consider different technology stacks and methodologies
- Balance innovation potential with implementation feasibility
- Provide concrete technical details for each approach
- Ensure ideas align with identified problem requirements

Always generate ideas that are both innovative and implementable within PoC constraints.
"""
    
    def execute_phase(self, state: PoCState) -> Dict[str, Any]:
        """Execute idea generation based on problem analysis and search results."""
        
        project = state["project"]
        context = self._get_project_context(state)
        previous_results = self._get_previous_results(state)
        
        # Get inputs from previous agents
        problem_analysis = state["agent_memory"].get("problem_identification", {}).get("problem_analysis", {})
        search_results = state["agent_memory"].get("search_problem", {}).get("search_results", {})
        
        if not problem_analysis:
            return {
                "success": False,
                "score": 0.0,
                "output": {},
                "feedback": "No problem analysis found. Please run problem identification first.",
                "artifacts": [],
                "memory": {}
            }
        
        # Prepare enhanced context from search results
        search_context = self._prepare_search_context(search_results)
        sample_code_context = self._prepare_sample_code_context(search_results)
        
        user_prompt = f"""
{context}

{previous_results}

PROBLEM ANALYSIS:
{json.dumps(problem_analysis, indent=2, ensure_ascii=False)}

{search_context}

{sample_code_context}

TASK: Generate 3-5 innovative and feasible PoC ideas to address the identified problems.

Use the research insights, technical approaches, and sample code to inform your ideas.
Each idea should have a different approach, complexity level, and technology focus.

For each idea, provide:

1. IDEA OVERVIEW
   - Title (concise, descriptive)
   - Core approach/concept
   - Key innovation aspects
   - Alignment with problem requirements

2. TECHNICAL DETAILS
   - Proposed technology stack
   - Architecture approach
   - Key technical components
   - Implementation strategy
   - Integration with sample code patterns

3. FEASIBILITY ASSESSMENT
   - Implementation complexity (1-5 scale)
   - Required skills/expertise
   - Estimated effort in hours
   - Risk factors and mitigation

4. IMPACT EVALUATION  
   - Expected business impact (1-5 scale)
   - Innovation potential (1-5 scale)
   - User experience improvements
   - Scalability potential

5. IMPLEMENTATION ROADMAP
   - Key development phases
   - Critical milestones
   - Dependencies and prerequisites
   - Testing and validation approach

6. PROS AND CONS
   - Key advantages
   - Potential drawbacks
   - Risk mitigation strategies

Focus on practical, implementable solutions that leverage the research findings.
Ensure variety in approaches while maintaining feasibility for PoC development.
"""
        
        schema = {
            "ideas": "list"  # Each idea will be a structured object
        }
        
        response = self._generate_structured_response(
            self.get_system_prompt(),
            user_prompt,
            schema
        )
        
        # Parse and create PoCIdea objects
        ideas = self._create_poc_ideas(response.get("ideas", []), search_results)
        
        # Store ideas in state
        state["ideas"] = ideas
        
        # Save ideas as artifact
        ideas_data = [self._idea_to_dict(idea) for idea in ideas]
        artifact_path = self._save_json_artifact(
            {"ideas": ideas_data, "generation_context": {
                "problem_analysis_summary": problem_analysis.get("core_problem", ""),
                "search_sources": search_results.get("sources_searched", []),
                "sample_code_count": len(search_results.get("sample_code_collection", [])),
                "technical_approaches_count": len(search_results.get("technical_approaches", []))
            }},
            f"idea_generation_iteration_{state['iteration']}.json",
            state
        )
        
        # Log generation results
        self._log_generation_summary(ideas, search_results)
        
        feedback = f"""
Idea Generation Complete:
- Generated Ideas: {len(ideas)}
- Average Complexity: {sum(idea.implementation_complexity for idea in ideas) / len(ideas):.1f}/5
- Average Impact: {sum(idea.expected_impact for idea in ideas) / len(ideas):.1f}/5
- Total Estimated Effort: {sum(idea.estimated_effort_hours for idea in ideas)} hours
- Search Sources Used: {', '.join(search_results.get('sources_searched', []))}
- Sample Code References: {len(search_results.get('sample_code_collection', []))}
"""
        
        score = self._calculate_ideas_score(ideas, search_results)
        
        return {
            "success": True,
            "score": score,
            "output": {"ideas": ideas_data, "ideas_count": len(ideas)},
            "feedback": feedback,
            "artifacts": [artifact_path],
            "memory": {
                "generated_ideas": ideas_data,
                "generation_count": len(ideas),
                "search_integration_score": self._calculate_search_integration_score(search_results),
                "generated_at": datetime.now().isoformat()
            }
        }
    
    def _prepare_search_context(self, search_results: Dict[str, Any]) -> str:
        """Prepare search results context for idea generation."""
        if not search_results:
            return "No search results available."
        
        context = f"""
PROBLEM DOMAIN RESEARCH RESULTS:

Sources Searched: {', '.join(search_results.get('sources_searched', []))}

Technical Approaches Found:
{json.dumps(search_results.get('technical_approaches', []), indent=2, ensure_ascii=False)}

Implementation Patterns:
{json.dumps(search_results.get('implementation_patterns', []), indent=2, ensure_ascii=False)}

Best Practices:
{json.dumps(search_results.get('best_practices', []), indent=2, ensure_ascii=False)}

Recommended Technologies:
{json.dumps(search_results.get('recommended_technologies', []), indent=2, ensure_ascii=False)}

Potential Challenges:
{json.dumps(search_results.get('potential_challenges', []), indent=2, ensure_ascii=False)}
"""
        return context
    
    def _prepare_sample_code_context(self, search_results: Dict[str, Any]) -> str:
        """Prepare sample code context for idea generation."""
        sample_code = search_results.get('sample_code_collection', [])
        if not sample_code:
            return "No sample code available."
        
        context = f"""
AVAILABLE SAMPLE CODE REFERENCES ({len(sample_code)} snippets):

"""
        
        # Show top 5 sample code snippets
        for i, snippet in enumerate(sample_code[:5], 1):
            context += f"""
Sample {i} - {snippet.get('language', 'unknown')} ({snippet.get('source', 'unknown')}):
Source: {snippet.get('article_title', snippet.get('file_path', 'Unknown'))}
Code Preview: {snippet.get('code_snippet', '')[:200]}...

"""
        
        if len(sample_code) > 5:
            context += f"... and {len(sample_code) - 5} more sample code snippets available.\n"
        
        return context
    
    def _create_poc_ideas(self, raw_ideas: List[Dict], search_results: Dict[str, Any]) -> List[PoCIdea]:
        """Create PoCIdea objects from raw idea data."""
        ideas = []
        
        for i, raw_idea in enumerate(raw_ideas):
            idea = PoCIdea()
            idea.id = f"idea_{i+1}"
            idea.title = raw_idea.get("title", f"PoC Idea {i+1}")
            idea.description = raw_idea.get("description", raw_idea.get("core_approach", ""))
            idea.technical_approach = raw_idea.get("technical_details", {}).get("architecture_approach", "")
            idea.implementation_complexity = raw_idea.get("feasibility_assessment", {}).get("implementation_complexity", 3)
            idea.expected_impact = raw_idea.get("impact_evaluation", {}).get("expected_business_impact", 3)
            idea.innovation_score = raw_idea.get("impact_evaluation", {}).get("innovation_potential", 3)
            idea.estimated_effort_hours = raw_idea.get("feasibility_assessment", {}).get("estimated_effort_hours", 24)
            
            # Calculate composite scores
            idea.feasibility_score = self._calculate_feasibility_score(raw_idea)
            idea.total_score = self._calculate_total_score(idea)
            
            # Add search-related enhancements
            idea.recommended_technologies = search_results.get("recommended_technologies", [])
            idea.sample_code_references = self._find_relevant_sample_code(raw_idea, search_results)
            
            # Additional fields from raw idea
            idea.technology_stack = raw_idea.get("technical_details", {}).get("technology_stack", [])
            idea.implementation_roadmap = raw_idea.get("implementation_roadmap", [])
            idea.pros = raw_idea.get("pros_and_cons", {}).get("advantages", [])
            idea.cons = raw_idea.get("pros_and_cons", {}).get("drawbacks", [])
            idea.risk_mitigation = raw_idea.get("pros_and_cons", {}).get("risk_mitigation", [])
            
            ideas.append(idea)
        
        return ideas
    
    def _calculate_feasibility_score(self, raw_idea: Dict) -> float:
        """Calculate feasibility score for an idea."""
        feasibility = raw_idea.get("feasibility_assessment", {})
        complexity = feasibility.get("implementation_complexity", 3)
        effort = feasibility.get("estimated_effort_hours", 24)
        
        # Lower complexity and effort = higher feasibility
        complexity_score = (6 - complexity) / 5.0  # Invert complexity (1-5 -> 5-1)
        effort_score = max(0, (48 - effort) / 48.0)  # Assume 48h is maximum reasonable effort
        
        return (complexity_score + effort_score) / 2.0
    
    def _calculate_total_score(self, idea: PoCIdea) -> float:
        """Calculate total score for an idea."""
        # Weighted average of different aspects
        feasibility_weight = 0.4
        impact_weight = 0.3
        innovation_weight = 0.3
        
        return (
            idea.feasibility_score * feasibility_weight +
            (idea.expected_impact / 5.0) * impact_weight +
            (idea.innovation_score / 5.0) * innovation_weight
        )
    
    def _find_relevant_sample_code(self, raw_idea: Dict, search_results: Dict[str, Any]) -> List[str]:
        """Find sample code snippets relevant to the idea."""
        sample_code = search_results.get('sample_code_collection', [])
        if not sample_code:
            return []
        
        # Simple relevance matching based on technology stack
        idea_tech_stack = raw_idea.get("technical_details", {}).get("technology_stack", [])
        if isinstance(idea_tech_stack, str):
            idea_tech_stack = [idea_tech_stack]
        
        relevant_snippets = []
        for snippet in sample_code[:10]:  # Check top 10 snippets
            snippet_code = snippet.get('code_snippet', '').lower()
            for tech in idea_tech_stack:
                if tech.lower() in snippet_code:
                    relevant_snippets.append(snippet.get('snippet_id', ''))
                    break
        
        return relevant_snippets[:5]  # Return top 5 relevant snippets
    
    def _idea_to_dict(self, idea: PoCIdea) -> Dict[str, Any]:
        """Convert PoCIdea object to dictionary."""
        return {
            "id": idea.id,
            "title": idea.title,
            "description": idea.description,
            "technical_approach": idea.technical_approach,
            "implementation_complexity": idea.implementation_complexity,
            "expected_impact": idea.expected_impact,
            "innovation_score": idea.innovation_score,
            "feasibility_score": idea.feasibility_score,
            "total_score": idea.total_score,
            "estimated_effort_hours": idea.estimated_effort_hours,
            "technology_stack": getattr(idea, 'technology_stack', []),
            "implementation_roadmap": getattr(idea, 'implementation_roadmap', []),
            "recommended_technologies": getattr(idea, 'recommended_technologies', []),
            "sample_code_references": getattr(idea, 'sample_code_references', []),
            "pros": getattr(idea, 'pros', []),
            "cons": getattr(idea, 'cons', []),
            "risk_mitigation": getattr(idea, 'risk_mitigation', [])
        }
    
    def _log_generation_summary(self, ideas: List[PoCIdea], search_results: Dict[str, Any]) -> None:
        """Log idea generation summary with color formatting."""
        print("\033[94m" + "="*60)  # Blue color
        print("💡 IDEA GENERATION RESULTS")
        print("="*60)
        print(f"🎯 Generated Ideas: {len(ideas)}")
        print(f"📊 Average Complexity: {sum(idea.implementation_complexity for idea in ideas) / len(ideas):.1f}/5")
        print(f"🚀 Average Impact: {sum(idea.expected_impact for idea in ideas) / len(ideas):.1f}/5")
        print(f"⏱️ Total Effort: {sum(idea.estimated_effort_hours for idea in ideas)} hours")
        print(f"🔍 Search Integration: {len(search_results.get('sources_searched', []))} sources")
        print(f"💻 Sample Code Used: {len(search_results.get('sample_code_collection', []))} snippets")
        
        print(f"\n📋 Generated Ideas:")
        for i, idea in enumerate(ideas, 1):
            print(f"   {i}. {idea.title} (Score: {idea.total_score:.2f}, Effort: {idea.estimated_effort_hours}h)")
        
        print("="*60 + "\033[0m")  # Reset color
    
    def _calculate_ideas_score(self, ideas: List[PoCIdea], search_results: Dict[str, Any]) -> float:
        """Calculate quality score for generated ideas."""
        if not ideas:
            return 0.0
        
        # Base score for having ideas
        score = 0.4
        
        # Score for quantity (up to 5 ideas)
        score += min(len(ideas) * 0.1, 0.3)
        
        # Score for diversity in complexity
        complexities = [idea.implementation_complexity for idea in ideas]
        if len(set(complexities)) > 1:
            score += 0.1
        
        # Score for search integration
        score += self._calculate_search_integration_score(search_results) * 0.2
        
        return min(score, 1.0)
    
    def _calculate_search_integration_score(self, search_results: Dict[str, Any]) -> float:
        """Calculate how well search results were integrated."""
        if not search_results:
            return 0.0
        
        integration_score = 0.0
        
        # Score for having search sources
        if search_results.get("sources_searched"):
            integration_score += 0.3
        
        # Score for technical approaches used
        if search_results.get("technical_approaches"):
            integration_score += 0.3
        
        # Score for sample code availability
        if search_results.get("sample_code_collection"):
            integration_score += 0.4
        
        return min(integration_score, 1.0)