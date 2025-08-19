"""Problem Identification & Ideation Agent for AI-PoC-Agents-v2."""

from typing import Dict, Any, List
import json
import os
from dotenv import load_dotenv

from .base_agent import BaseAgent
from ..core.state import PoCState, PoCIdea
from ..tools.qiita_search import QiitaSemanticSearchTool

# Load environment variables
load_dotenv()


class ProblemAgent(BaseAgent):
    """Agent responsible for problem identification, idea generation, and selection."""
    
    def __init__(self, agent_type: str, config):
        """Initialize ProblemAgent with Qiita integration."""
        super().__init__(agent_type, config)
        
        # Initialize Qiita search tool with environment variable
        qiita_access_token = os.getenv("QIITA_ACCESS_TOKEN")
        self.qiita_tool = QiitaSemanticSearchTool(access_token=qiita_access_token)
        self.qiita_enabled = True
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        return """
You are an expert Problem Identification & Ideation Agent for PoC (Proof of Concept) development.

Your responsibilities:
1. PROBLEM IDENTIFICATION: Analyze user themes to identify specific, actionable problems
2. IDEA GENERATION: Create multiple innovative and feasible solution approaches  
3. IDEA SELECTION: Evaluate and select the best PoC candidate

Key principles:
- Focus on practical, implementable solutions
- Consider technical feasibility, business impact, and innovation potential
- Provide clear reasoning for recommendations
- Balance ambition with realistic constraints
- Think from multiple perspectives (technical, business, user)

Always provide structured, actionable outputs that can guide the next phase of PoC development.
"""
    
    def execute_phase(self, state: PoCState) -> Dict[str, Any]:
        """Execute the agent's logic based on current phase."""
        
        current_phase = state["current_phase"]
        
        if current_phase == "problem_identification":
            return self._identify_problem(state)
        elif current_phase == "idea_generation":
            return self._generate_ideas(state)
        elif current_phase == "idea_selection":
            return self._select_idea(state)
        else:
            raise ValueError(f"ProblemAgent cannot handle phase: {current_phase}")
    
    def _identify_problem(self, state: PoCState) -> Dict[str, Any]:
        """Identify and clarify the problem from user theme."""
        
        project = state["project"]
        context = self._get_project_context(state)
        previous_results = self._get_previous_results(state, same_agent_only=True)
        
        user_prompt = f"""
{context}

{previous_results}

TASK: Analyze the user's theme and identify specific, actionable problems for PoC development.

User Theme: "{project.theme}"
User Description: "{project.description}"

Please provide a detailed problem analysis including:

1. CORE PROBLEM IDENTIFICATION
   - What is the main problem to solve?
   - Why is this problem important?
   - Who are the affected stakeholders?

2. PROBLEM DECOMPOSITION  
   - Break down into specific sub-problems
   - Identify critical vs. nice-to-have aspects
   - Highlight potential technical challenges

3. CONTEXT ANALYSIS
   - Domain/industry context
   - Existing solutions and their limitations
   - Market/user needs

4. SUCCESS CRITERIA DEFINITION
   - How will we measure success?
   - What are the key performance indicators?
   - What would constitute a successful PoC?

5. CONSTRAINTS & REQUIREMENTS
   - Technical constraints
   - Resource limitations
   - Timeline considerations
   - Compliance/regulatory requirements

6. TARGET USER ANALYSIS
   - Who are the primary users?
   - What are their pain points?
   - How would they interact with the solution?

7. TASK TYPE CLASSIFICATION
   Based on the problem description and domain, classify the primary task type:
   - IMAGE_PROCESSING: OCR, computer vision, image classification, object detection
   - CLASSIFICATION: Data classification, fraud detection, network intrusion, sentiment analysis
   - REGRESSION: Price prediction, time series forecasting, numerical prediction
   - NLP: Text processing, language translation, document analysis
   - RECOMMENDATION: Content recommendation, collaborative filtering
   - OTHER: Any other specialized task type
   
   Provide reasoning for the classification and identify key technical requirements for this task type.

Provide specific, actionable insights that can guide PoC development.
"""
        
        schema = {
            "core_problem": "string",
            "problem_importance": "string", 
            "stakeholders": "list",
            "sub_problems": "list",
            "critical_aspects": "list",
            "technical_challenges": "list",
            "domain_context": "string",
            "existing_solutions": "list",
            "success_criteria": "list",
            "kpis": "list", 
            "technical_constraints": "list",
            "resource_limitations": "list",
            "timeline_considerations": "string",
            "target_users": "list",
            "user_pain_points": "list",
            "recommendations": "list",
            "task_type": "string",
            "task_type_reasoning": "string",
            "technical_requirements": "list"
        }
        
        response = self._generate_structured_response(
            self.get_system_prompt(),
            user_prompt,
            schema
        )
        
        # Update project with identified requirements
        project.requirements = response.get("success_criteria", [])
        project.constraints = response.get("technical_constraints", [])
        project.target_users = response.get("target_users", [])
        project.success_criteria = response.get("kpis", [])
        project.domain = response.get("domain_context", "")
        
        # Add task type information to project
        project.task_type = response.get("task_type", "OTHER")
        project.task_type_reasoning = response.get("task_type_reasoning", "")
        project.technical_requirements = response.get("technical_requirements", [])
        
        # Log task type detection results with cyan color
        print("\033[96m" + "="*60)  # Cyan color
        print("🎯 TASK TYPE DETECTION RESULTS")
        print("="*60)
        print(f"📊 Detected Task Type: {project.task_type}")
        print(f"💡 Reasoning: {project.task_type_reasoning}")
        print(f"🔧 Technical Requirements:")
        for req in project.technical_requirements:
            print(f"   - {req}")
        print("="*60 + "\033[0m")  # Reset color
        
        # Save analysis as artifact
        analysis_content = json.dumps(response, indent=2, ensure_ascii=False)
        artifact_path = self._save_artifact(
            analysis_content,
            f"problem_analysis_iteration_{state['iteration']}.json",
            state
        )
        
        # Save state for debugging
        state_pkl_path = self._save_state_debug(state, "problem_analysis")
        artifacts = [artifact_path, state_pkl_path]
        
        # Prepare summary for feedback
        feedback = f"""
Problem Analysis Complete:
- Core Problem: {response.get('core_problem', 'Not identified')}
- Stakeholders: {len(response.get('stakeholders', []))} identified
- Sub-problems: {len(response.get('sub_problems', []))} identified  
- Success Criteria: {len(response.get('success_criteria', []))} defined
- Technical Challenges: {len(response.get('technical_challenges', []))} identified
"""
        
        # Calculate score based on completeness and quality
        score = self._calculate_problem_analysis_score(response)
        
        return {
            "success": True,
            "score": score,
            "output": response,
            "feedback": feedback,
            "artifacts": artifacts,
            "memory": {
                "problem_analysis": response,
                "identified_at": state["updated_at"].isoformat() if "updated_at" in state else None
            }
        }
    
    def _generate_ideas(self, state: PoCState) -> Dict[str, Any]:
        """Generate multiple PoC ideas with Qiita integration."""
        
        project = state["project"]
        context = self._get_project_context(state)
        previous_results = self._get_previous_results(state, same_agent_only=True)
        
        # Get problem analysis from memory
        problem_analysis = self._get_agent_memory(state).get("problem_analysis", {})
        
        # Step 0: Search problem domain using Qiita (if not already done)
        problem_search_results = state.get("problem_search", {}).get("qiita", {})
        if not problem_search_results:
            print("🔍 Performing Qiita problem domain search before idea generation...")
            search_result = self._search_problem_qiita(state)
            if search_result.get("success"):
                problem_search_results = search_result.get("output", {})
            else:
                print("⚠️ Problem search failed, continuing with basic idea generation")
        
        # Step 1: Search Qiita for relevant articles and generate ideas
        qiita_ideas = []
        qiita_insights = {}
        
        if self.qiita_enabled:
            try:
                # Extract technical keywords from problem analysis
                technical_keywords = []
                if problem_analysis:
                    technical_keywords.extend(problem_analysis.get("technical_challenges", []))
                    domain_context = problem_analysis.get("domain_context", "")
                    if domain_context:
                        technical_keywords.append(domain_context)
                
                # Search Qiita articles
                qiita_articles = self.qiita_tool.search_relevant_articles(
                    project_theme=project.theme,
                    technical_keywords=technical_keywords,
                    max_articles=20
                )
                
                # Generate ideas from Qiita articles
                if qiita_articles:
                    qiita_ideas = self.qiita_tool.generate_poc_ideas_from_articles(
                        project_theme=project.theme,
                        articles=qiita_articles
                    )
                    qiita_insights = self.qiita_tool.extract_implementation_insights(qiita_articles)
                    
                    # Save Qiita insights as artifact
                    qiita_content = json.dumps({
                        "articles_found": len(qiita_articles),
                        "ideas_generated": len(qiita_ideas),
                        "insights": qiita_insights,
                        "top_articles": qiita_articles[:5]
                    }, indent=2, ensure_ascii=False, default=str)
                    
                    self._save_artifact(
                        qiita_content,
                        f"qiita_research_iteration_{state['iteration']}.json",
                        state
                    )
                    
            except Exception as e:
                print(f"Qiita integration error: {e}")
                # Continue without Qiita integration if it fails
        
        # Step 2: Generate LLM-based ideas with enhanced context
        # Combine problem search results with traditional Qiita insights
        enhanced_qiita_context = ""
        if problem_search_results:
            enhanced_qiita_context += f"""
PROBLEM DOMAIN RESEARCH (QIITA):
- Technical Approaches: {len(problem_search_results.get('technical_approaches', []))} found
- Best Practices: {len(problem_search_results.get('best_practices', []))} identified
- Potential Challenges: {len(problem_search_results.get('potential_challenges', []))} cataloged
- Reference Articles: {len(problem_search_results.get('reference_articles', []))} analyzed
- Top Technologies: {list(problem_search_results.get('common_technologies', {}).keys())[:5]}

TECHNICAL APPROACHES FROM RESEARCH:
{json.dumps(problem_search_results.get('technical_approaches', []), indent=2, ensure_ascii=False)}

BEST PRACTICES IDENTIFIED:
{json.dumps(problem_search_results.get('best_practices', []), indent=2, ensure_ascii=False)}

POTENTIAL CHALLENGES TO CONSIDER:
{json.dumps(problem_search_results.get('potential_challenges', []), indent=2, ensure_ascii=False)}
"""

        qiita_context = ""
        if qiita_ideas:
            qiita_context = f"""
QIITA IDEA GENERATION INSIGHTS:
- Found {len(qiita_articles)} relevant technical articles for idea generation
- Generated {len(qiita_ideas)} ideas from community implementations
- Top technologies used: {list(qiita_insights.get('common_technologies', {}).keys())[:5]}

QIITA-INSPIRED IDEAS:
{json.dumps(qiita_ideas, indent=2, ensure_ascii=False, default=str)}
"""

        user_prompt = f"""
{context}

{previous_results}

PROBLEM ANALYSIS RESULTS:
{json.dumps(problem_analysis, indent=2) if problem_analysis else "No previous analysis available"}

{enhanced_qiita_context}

{qiita_context}

TASK: Generate multiple innovative and feasible PoC ideas to address the identified problems.

NOTE: 
- Consider the problem domain research insights and technical approaches identified
- Learn from the best practices and be aware of potential challenges
- You can build upon, combine, or create variations of the research findings
- Generate completely new approaches that leverage the technical insights

Generate 3-5 distinct PoC approaches, each with different:
- Technical approaches
- Implementation strategies  
- Technology stacks
- Complexity levels
- Innovation aspects

For each idea, provide:

1. IDEA OVERVIEW
   - Title (concise, descriptive)
   - Core approach/concept
   - Key innovation aspects

2. TECHNICAL DETAILS
   - Proposed technology stack
   - Architecture approach
   - Key technical components
   - Implementation strategy

3. FEASIBILITY ASSESSMENT
   - Implementation complexity (1-5 scale)
   - Required skills/expertise
   - Estimated effort in hours
   - Risk factors

4. IMPACT EVALUATION  
   - Expected business impact (1-5 scale)
   - Innovation potential (1-5 scale)
   - User experience improvements
   - Scalability potential

5. PROS AND CONS
   - Key advantages
   - Potential drawbacks
   - Mitigation strategies

Focus on practical, implementable solutions while encouraging innovation.
"""
        
        response = self._generate_response(self.get_system_prompt(), user_prompt)
        
        # Parse ideas from response
        llm_ideas = self._parse_ideas_from_response(response)
        
        # Step 3: Combine Qiita ideas with LLM ideas
        all_ideas = []
        
        # Convert Qiita ideas to PoCIdea objects
        for qiita_idea in qiita_ideas:
            poc_idea = PoCIdea()
            poc_idea.id = qiita_idea.get("id", f"qiita_{len(all_ideas)}")
            poc_idea.title = qiita_idea.get("title", "Qiita-inspired idea")
            poc_idea.description = qiita_idea.get("description", "")
            poc_idea.technical_approach = qiita_idea.get("technical_approach", "")
            poc_idea.implementation_complexity = qiita_idea.get("implementation_complexity", 3)
            poc_idea.expected_impact = qiita_idea.get("expected_impact", 3)
            poc_idea.feasibility_score = qiita_idea.get("feasibility_score", 0.7)
            poc_idea.total_score = qiita_idea.get("feasibility_score", 0.7)
            poc_idea.estimated_effort_hours = qiita_idea.get("estimated_effort_hours", 16)
            poc_idea.inspiration_source = "qiita"
            poc_idea.qiita_reference_articles = qiita_idea.get("reference_articles", [])
            poc_idea.qiita_code_examples = qiita_idea.get("code_examples", [])
            all_ideas.append(poc_idea)
        
        # Add LLM-generated ideas
        all_ideas.extend(llm_ideas)
        
        # Enhance LLM ideas with Qiita insights if available
        if self.qiita_enabled and qiita_insights:
            try:
                # Convert LLM ideas to dict format for enhancement
                llm_ideas_dict = [idea.__dict__ for idea in llm_ideas]
                enhanced_ideas_dict = self.qiita_tool.enhance_existing_ideas(
                    llm_ideas_dict, project.theme
                )
                
                # Update the LLM ideas with enhancements
                for i, enhanced_dict in enumerate(enhanced_ideas_dict):
                    if i < len(llm_ideas):
                        llm_ideas[i].qiita_recommended_technologies = enhanced_dict.get("qiita_recommended_technologies", [])
                        llm_ideas[i].qiita_reference_articles = enhanced_dict.get("qiita_reference_articles", [])
                        llm_ideas[i].qiita_code_examples = enhanced_dict.get("qiita_code_examples", [])
                        
            except Exception as e:
                print(f"Error enhancing ideas with Qiita: {e}")
        
        ideas = all_ideas[:5]  # Limit to 5 ideas total
        
        # Save ideas to state
        state["ideas"] = ideas
        
        # Save as artifact
        ideas_data = [idea.__dict__ for idea in ideas]
        ideas_content = json.dumps(ideas_data, indent=2, ensure_ascii=False, default=str)
        artifact_path = self._save_artifact(
            ideas_content,
            f"generated_ideas_iteration_{state['iteration']}.json", 
            state
        )
        
        # Save state for debugging
        state_pkl_path = self._save_state_debug(state, "idea_generation")
        artifacts = [artifact_path, state_pkl_path]
        
        qiita_idea_count = len([idea for idea in ideas if getattr(idea, 'inspiration_source', '') == 'qiita'])
        
        feedback = f"""
Idea Generation Complete:
- Generated Ideas: {len(ideas)} (including {qiita_idea_count} from Qiita research)
- Average Complexity: {sum(idea.implementation_complexity for idea in ideas) / len(ideas):.1f}/5
- Average Impact: {sum(idea.expected_impact for idea in ideas) / len(ideas):.1f}/5
- Total Estimated Effort: {sum(idea.estimated_effort_hours for idea in ideas)} hours
- Qiita Articles Analyzed: {len(qiita_articles) if qiita_ideas else 0}
"""
        
        score = self._calculate_ideas_generation_score(ideas)
        
        return {
            "success": True,
            "score": score,
            "output": {"ideas": ideas_data, "ideas_count": len(ideas)},
            "feedback": feedback,
            "artifacts": artifacts,
            "memory": {
                "generated_ideas": ideas_data,
                "generation_count": len(ideas),
                "qiita_ideas_count": qiita_idea_count,
                "qiita_articles_analyzed": len(qiita_articles) if qiita_ideas else 0,
                "qiita_insights": qiita_insights,
                "generated_at": state["updated_at"].isoformat() if "updated_at" in state else None
            }
        }
    
    def _select_idea(self, state: PoCState) -> Dict[str, Any]:
        """Select the best PoC idea from generated options."""
        
        ideas = state.get("ideas", [])
        if not ideas:
            raise ValueError("No ideas available for selection")
        
        context = self._get_project_context(state)
        previous_results = self._get_previous_results(state, same_agent_only=True)
        
        # Convert ideas to dict format for prompt
        ideas_data = [idea.__dict__ if hasattr(idea, '__dict__') else idea for idea in ideas]
        
        user_prompt = f"""
{context}

{previous_results}

AVAILABLE IDEAS:
{json.dumps(ideas_data, indent=2, ensure_ascii=False, default=str)}

TASK: Select the best PoC idea based on comprehensive evaluation.

Evaluation Criteria:
1. FEASIBILITY (40% weight)
   - Implementation complexity vs. available resources
   - Technical risk assessment
   - Required skills availability
   - Timeline compatibility

2. IMPACT POTENTIAL (30% weight)  
   - Business value creation
   - User experience improvement
   - Innovation level
   - Scalability potential

3. STRATEGIC ALIGNMENT (20% weight)
   - Alignment with project goals
   - Market relevance
   - Competitive advantage
   - Long-term viability

4. LEARNING VALUE (10% weight)
   - Skills/knowledge development
   - Technology exploration
   - Process improvement

Provide:
1. DETAILED EVALUATION of each idea against criteria
2. SCORING (0.0-1.0) for each criterion per idea
3. FINAL RECOMMENDATION with clear reasoning
4. IMPLEMENTATION ROADMAP for selected idea
5. RISK MITIGATION STRATEGIES
6. SUCCESS METRICS definition

Select the idea that offers the best balance of feasibility, impact, and learning value.
"""
        
        schema = {
            "evaluations": "list",
            "selected_idea_id": "string", 
            "selection_reasoning": "string",
            "feasibility_score": "number",
            "impact_score": "number", 
            "strategic_score": "number",
            "learning_score": "number",
            "total_score": "number",
            "implementation_roadmap": "list",
            "risk_mitigation": "list",
            "success_metrics": "list",
            "next_steps": "list"
        }
        
        response = self._generate_structured_response(
            self.get_system_prompt(),
            user_prompt,
            schema
        )
        
        # Find and set selected idea
        selected_idea_id = response.get("selected_idea_id", "")
        selected_idea = None
        
        for idea in ideas:
            idea_dict = idea.__dict__ if hasattr(idea, '__dict__') else idea
            if idea_dict.get("id") == selected_idea_id:
                selected_idea = idea
                break
        
        if not selected_idea and ideas:
            # Fallback: select highest scoring idea
            if hasattr(ideas[0], '__dict__'):
                selected_idea = max(ideas, key=lambda x: x.total_score)
            else:
                selected_idea = max(ideas, key=lambda x: x.get("total_score", 0))
        
        # Update selected idea with evaluation scores
        if selected_idea:
            if hasattr(selected_idea, '__dict__'):
                selected_idea.feasibility_score = response.get("feasibility_score", 0.0)
                selected_idea.innovation_score = response.get("impact_score", 0.0) 
                selected_idea.total_score = response.get("total_score", 0.0)
            else:
                selected_idea["feasibility_score"] = response.get("feasibility_score", 0.0)
                selected_idea["innovation_score"] = response.get("impact_score", 0.0)
                selected_idea["total_score"] = response.get("total_score", 0.0)
        
        state["selected_idea"] = selected_idea
        
        # Save selection analysis as artifact
        selection_content = json.dumps(response, indent=2, ensure_ascii=False, default=str)
        artifact_path = self._save_artifact(
            selection_content,
            f"idea_selection_iteration_{state['iteration']}.json",
            state  
        )
        
        # Save state for debugging
        state_pkl_path = self._save_state_debug(state, "idea_selection")
        artifacts = [artifact_path, state_pkl_path]
        
        selected_title = ""
        if selected_idea:
            if hasattr(selected_idea, '__dict__'):
                selected_title = selected_idea.title
            else:
                selected_title = selected_idea.get("title", "Unknown")
        
        feedback = f"""
Idea Selection Complete:
- Selected Idea: {selected_title}
- Total Score: {response.get('total_score', 0.0):.3f}
- Feasibility: {response.get('feasibility_score', 0.0):.3f}
- Impact Potential: {response.get('impact_score', 0.0):.3f}
- Implementation Steps: {len(response.get('implementation_roadmap', []))}
"""
        
        score = response.get("total_score", 0.5)
        
        return {
            "success": True,
            "score": score,
            "output": response,
            "feedback": feedback,
            "artifacts": artifacts,
            "memory": {
                "selected_idea": selected_idea.__dict__ if hasattr(selected_idea, '__dict__') else selected_idea,
                "selection_analysis": response,
                "selected_at": state["updated_at"].isoformat() if "updated_at" in state else None
            }
        }
    
    def _parse_ideas_from_response(self, response: str) -> List[PoCIdea]:
        """Parse ideas from LLM response."""
        ideas = []
        
        # Try to extract structured ideas from response
        # This is a simplified parser - in practice, you might want more robust parsing
        
        # Look for common patterns in the response
        sections = response.split("###") if "###" in response else response.split("##")
        
        idea_count = 0
        for section in sections[1:]:  # Skip first section (usually intro)
            if any(keyword in section.lower() for keyword in ["idea", "approach", "solution"]):
                idea_count += 1
                
                idea = PoCIdea()
                idea.id = f"idea_{idea_count}"
                
                # Extract title (usually in first line)
                lines = section.strip().split('\n')
                idea.title = lines[0].strip() if lines else f"Idea {idea_count}"
                
                # Simple scoring (would be improved with more sophisticated parsing)
                idea.implementation_complexity = min(idea_count + 1, 5)
                idea.expected_impact = min(6 - idea_count, 5) 
                idea.feasibility_score = 0.6 + (idea_count * 0.1)
                idea.total_score = (idea.feasibility_score + idea.expected_impact * 0.2) / 1.2
                
                # Extract basic info
                idea.description = section[:200] + "..." if len(section) > 200 else section
                idea.estimated_effort_hours = 8 + (idea_count * 4)
                
                ideas.append(idea)
        
        # Ensure we have at least one idea
        if not ideas:
            default_idea = PoCIdea()
            default_idea.id = "default_idea"
            default_idea.title = "Default PoC Approach"
            default_idea.description = "A basic PoC implementation approach"
            default_idea.feasibility_score = 0.7
            default_idea.total_score = 0.7
            ideas.append(default_idea)
        
        return ideas[:5]  # Limit to 5 ideas max
    
    def _search_problem_qiita(self, state: PoCState) -> Dict[str, Any]:
        """Search and analyze problem domain using Qiita to gather technical background."""
        
        project = state["project"]
        context = self._get_project_context(state)
        previous_results = self._get_previous_results(state, same_agent_only=True)
        
        # Get problem analysis from memory
        problem_analysis = self._get_agent_memory(state).get("problem_analysis", {})
        
        # Extract search keywords from problem analysis and project
        search_keywords = []
        domain_keywords = []
        
        if problem_analysis:
            # Technical keywords from challenges and requirements
            search_keywords.extend(problem_analysis.get("technical_challenges", []))
            search_keywords.extend(problem_analysis.get("technical_requirements", []))
            
            # Domain context
            domain_context = problem_analysis.get("domain_context", "")
            if domain_context:
                domain_keywords.append(domain_context)
        
        # Add project-based keywords
        if hasattr(project, 'task_type'):
            search_keywords.append(project.task_type.lower().replace('_', ' '))
        
        search_keywords.extend([project.theme, project.description])
        
        # Initialize search results
        problem_search_results = {
            "technical_approaches": [],
            "common_technologies": {},
            "implementation_patterns": [],
            "best_practices": [],
            "potential_challenges": [],
            "reference_articles": [],
            "search_keywords_used": search_keywords + domain_keywords,
            "source": "qiita"
        }
        
        # Search Qiita for problem domain insights
        if self.qiita_enabled:
            try:
                # Search for technical approaches and solutions
                technical_articles = self.qiita_tool.search_relevant_articles(
                    project_theme=project.theme,
                    technical_keywords=search_keywords[:10],  # Limit keywords
                    max_articles=30  # More articles for comprehensive search
                )
                
                if technical_articles:
                    # Extract technical approaches and patterns
                    technical_insights = self.qiita_tool.extract_implementation_insights(technical_articles)
                    
                    # Process insights into structured data
                    problem_search_results.update({
                        "common_technologies": technical_insights.get("common_technologies", {}),
                        "implementation_patterns": technical_insights.get("code_patterns", []),
                        "reference_articles": technical_articles[:15]  # Keep top articles
                    })
                    
                    # Extract technical approaches using LLM analysis
                    articles_text = "\n".join([
                        f"Title: {article.get('title', '')}\nTags: {', '.join(article.get('tags', []))}\nContent: {article.get('body', '')[:500]}"
                        for article in technical_articles[:10]
                    ])
                    
                    analysis_prompt = f"""
Based on these Qiita articles about {project.theme}, extract:

1. Technical Approaches (different ways to solve similar problems)
2. Best Practices (recommended patterns and methodologies)  
3. Potential Challenges (common issues and gotchas)

Articles:
{articles_text}

Focus on actionable technical information that can guide PoC development.
"""
                    
                    schema = {
                        "technical_approaches": "list",
                        "best_practices": "list", 
                        "potential_challenges": "list"
                    }
                    
                    qiita_analysis = self._generate_structured_response(
                        self.get_system_prompt(),
                        analysis_prompt,
                        schema
                    )
                    
                    problem_search_results.update({
                        "technical_approaches": qiita_analysis.get("technical_approaches", []),
                        "best_practices": qiita_analysis.get("best_practices", []),
                        "potential_challenges": qiita_analysis.get("potential_challenges", [])
                    })
                    
                # Save comprehensive search results as artifact
                search_content = json.dumps(problem_search_results, indent=2, ensure_ascii=False, default=str)
                artifact_path = self._save_artifact(
                    search_content,
                    f"problem_search_qiita_iteration_{state['iteration']}.json",
                    state
                )
                
                # Log search results with green color
                print("\033[92m" + "="*60)  # Green color
                print("🔍 QIITA PROBLEM DOMAIN SEARCH RESULTS")
                print("="*60)
                print(f"📊 Articles Analyzed: {len(technical_articles)}")
                print(f"🛠️ Technical Approaches: {len(problem_search_results['technical_approaches'])}")
                print(f"📋 Best Practices: {len(problem_search_results['best_practices'])}")
                print(f"⚠️ Potential Challenges: {len(problem_search_results['potential_challenges'])}")
                print(f"🏷️ Top Technologies: {list(problem_search_results['common_technologies'].keys())[:5]}")
                print("="*60 + "\033[0m")  # Reset color
                
            except Exception as e:
                print(f"Qiita problem search error: {e}")
                artifact_path = None
        
        # Store search results in state for use by _generate_ideas
        if "problem_search" not in state:
            state["problem_search"] = {}
        state["problem_search"]["qiita"] = problem_search_results
        
        # Save state for debugging
        state_pkl_path = self._save_state_debug(state, "problem_search_qiita")
        artifacts = [artifact_path, state_pkl_path] if artifact_path else [state_pkl_path]
        
        feedback = f"""
Qiita Problem Domain Search Complete:
- Technical Articles: {len(problem_search_results.get('reference_articles', []))} analyzed
- Technical Approaches: {len(problem_search_results.get('technical_approaches', []))} identified
- Best Practices: {len(problem_search_results.get('best_practices', []))} found
- Technologies: {len(problem_search_results.get('common_technologies', {}))} cataloged
- Search Keywords: {len(problem_search_results.get('search_keywords_used', []))} used
"""
        
        # Calculate score based on search comprehensiveness
        score = self._calculate_problem_search_score(problem_search_results)
        
        return {
            "success": True,
            "score": score,
            "output": problem_search_results,
            "feedback": feedback,
            "artifacts": artifacts,
            "memory": {
                "problem_search_qiita": problem_search_results,
                "searched_at": state["updated_at"].isoformat() if "updated_at" in state else None
            }
        }
    
    def _calculate_problem_search_score(self, search_results: Dict[str, Any]) -> float:
        """Calculate quality score for problem domain search."""
        score = 0.0
        
        # Base score for having search results
        if search_results:
            score += 0.3
        
        # Score for technical approaches found
        approaches = search_results.get("technical_approaches", [])
        score += min(len(approaches) * 0.1, 0.3)
        
        # Score for best practices
        practices = search_results.get("best_practices", [])
        score += min(len(practices) * 0.05, 0.2)
        
        # Score for reference articles
        articles = search_results.get("reference_articles", [])
        score += min(len(articles) * 0.01, 0.2)
        
        return min(score, 1.0)
    
    def _calculate_problem_analysis_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate quality score for problem analysis."""
        score = 0.0
        
        # Check completeness of required fields
        required_fields = [
            "core_problem", "stakeholders", "sub_problems", "success_criteria",
            "technical_challenges", "target_users"
        ]
        
        for field in required_fields:
            if field in analysis and analysis[field]:
                if isinstance(analysis[field], list):
                    score += 0.15 if len(analysis[field]) > 0 else 0
                else:
                    score += 0.15 if len(str(analysis[field])) > 10 else 0.05
        
        # Bonus for depth and quality
        if len(analysis.get("sub_problems", [])) >= 3:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_ideas_generation_score(self, ideas: List[PoCIdea]) -> float:
        """Calculate quality score for idea generation."""
        if not ideas:
            return 0.0
        
        # Base score for having ideas
        score = 0.4
        
        # Score for quantity (up to 5 ideas)
        score += min(len(ideas) * 0.1, 0.3)
        
        # Score for diversity in complexity and approach
        complexities = [idea.implementation_complexity if hasattr(idea, 'implementation_complexity') 
                       else idea.get('implementation_complexity', 3) for idea in ideas]
        if len(set(complexities)) > 1:
            score += 0.1
        
        # Average feasibility bonus
        avg_feasibility = sum(idea.feasibility_score if hasattr(idea, 'feasibility_score')
                            else idea.get('feasibility_score', 0.5) for idea in ideas) / len(ideas)
        score += avg_feasibility * 0.2
        
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
    """ProblemAgentの動作確認用メイン関数"""
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
    from ai_poc_agents_v2.core.state import PoCProject
    from ai_poc_agents_v2.agents.base_agent import BaseAgent
    from ai_poc_agents_v2.tools.qiita_search import QiitaSemanticSearchTool
    
    # Load environment variables
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Test ProblemAgent with Qiita integration")
    parser.add_argument("--theme", type=str, default="OCR文字認識システム", help="Project theme")
    parser.add_argument("--description", type=str, default="画像から文字を認識して抽出するシステムの開発", help="Project description")
    parser.add_argument("--phase", type=str, choices=["problem_identification", "idea_generation", "idea_selection", "all"], 
                      default="all", help="Which phase to test")
    parser.add_argument("--workspace", type=str, default="./tmp/test_problem_agent", help="Workspace directory")
    parser.add_argument("--no-qiita", action="store_true", help="Disable Qiita integration")
    
    args = parser.parse_args()
    
    print("=== ProblemAgent Test with Qiita Integration ===")
    print(f"Theme: {args.theme}")
    print(f"Description: {args.description}")
    print(f"Phase: {args.phase}")
    print(f"Qiita integration: {'Disabled' if args.no_qiita else 'Enabled'}")
    print()
    
    # Check environment
    qiita_token = os.getenv("QIITA_ACCESS_TOKEN")
    if qiita_token and not args.no_qiita:
        print(f"✓ QIITA_ACCESS_TOKEN found (length: {len(qiita_token)})")
    elif not args.no_qiita:
        print("⚠ No QIITA_ACCESS_TOKEN - using anonymous Qiita access")
    else:
        print("ℹ Qiita integration disabled by user")
    
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
    print("\n1. Initializing ProblemAgent...")
    config = Config()
    
    # Create agent
    problem_agent = ProblemAgent("problem_identifier", config)
    
    # Optionally disable Qiita
    if args.no_qiita:
        problem_agent.qiita_enabled = False
    
    # Create project and state
    project = PoCProject(theme=args.theme)
    project.description = args.description
    
    state = {
        "project": project,
        "current_phase": "problem_identification",
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
    
    print("✓ ProblemAgent initialized successfully")
    
    # Test phases based on user selection
    if args.phase in ["problem_identification", "all"]:
        print("\n2. Testing Problem Identification Phase...")
        result = problem_agent._identify_problem(state)
        print(f"✓ Problem identification completed")
        print(f"   Success: {result['success']}")
        print(f"   Score: {result['score']:.3f}")
        print(f"   Artifacts: {len(result['artifacts'])}")
        
        # Show some results
        output = result.get('output', {})
        if output.get('core_problem'):
            print(f"   Core Problem: {output['core_problem'][:100]}...")
        if output.get('stakeholders'):
            print(f"   Stakeholders: {len(output['stakeholders'])} identified")
        if output.get('technical_challenges'):
            print(f"   Technical Challenges: {len(output['technical_challenges'])} identified")
        
        print(f"   Feedback: {result['feedback'][:200]}...")
    
    if args.phase in ["idea_generation", "all"]:
        print("\n3. Testing Idea Generation Phase...")
        state["current_phase"] = "idea_generation"
        
        result = problem_agent._generate_ideas(state)
        print(f"✓ Idea generation completed")
        print(f"   Success: {result['success']}")
        print(f"   Score: {result['score']:.3f}")
        print(f"   Artifacts: {len(result['artifacts'])}")
        
        # Show detailed results
        output = result.get('output', {})
        memory = result.get('memory', {})
        
        print(f"   Generated Ideas: {output.get('ideas_count', 0)}")
        print(f"   Qiita Ideas: {memory.get('qiita_ideas_count', 0)}")
        print(f"   Qiita Articles Analyzed: {memory.get('qiita_articles_analyzed', 0)}")
        
        # Show ideas
        ideas = state.get("ideas", [])
        print(f"\n   Generated Ideas Details:")
        for i, idea in enumerate(ideas, 1):
            print(f"   {i}. {idea.title}")
            print(f"      Description: {idea.description[:80]}...")
            print(f"      Technical Approach: {getattr(idea, 'technical_approach', 'N/A')}")
            print(f"      Complexity: {idea.implementation_complexity}/5")
            print(f"      Impact: {idea.expected_impact}/5")
            print(f"      Feasibility: {idea.feasibility_score:.2f}")
            print(f"      Source: {getattr(idea, 'inspiration_source', 'llm')}")
            
            # Show Qiita enhancements if available
            if hasattr(idea, 'qiita_reference_articles') and idea.qiita_reference_articles:
                print(f"      Qiita References: {len(idea.qiita_reference_articles)}")
            if hasattr(idea, 'qiita_recommended_technologies') and idea.qiita_recommended_technologies:
                print(f"      Recommended Tech: {', '.join(idea.qiita_recommended_technologies[:3])}")
            print()
        
        # Show Qiita insights if available
        qiita_insights = memory.get('qiita_insights', {})
        if qiita_insights and qiita_insights.get('common_technologies'):
            print(f"   Top Qiita Technologies:")
            for tech, count in list(qiita_insights['common_technologies'].items())[:5]:
                print(f"     - {tech}: {count}")
        
        print(f"   Feedback: {result['feedback'][:300]}...")
    
    if args.phase in ["idea_selection", "all"]:
        print("\n4. Testing Idea Selection Phase...")
        state["current_phase"] = "idea_selection"
        
        # Ensure we have ideas to select from
        if not state.get("ideas"):
            print("   No ideas available - skipping idea selection")
        else:
            result = problem_agent._select_idea(state)
            print(f"✓ Idea selection completed")
            print(f"   Success: {result['success']}")
            print(f"   Score: {result['score']:.3f}")
            print(f"   Artifacts: {len(result['artifacts'])}")
            
            # Show selection results
            selected_idea = state.get("selected_idea")
            if selected_idea:
                if hasattr(selected_idea, '__dict__'):
                    print(f"   Selected Idea: {selected_idea.title}")
                    print(f"   Feasibility Score: {getattr(selected_idea, 'feasibility_score', 'N/A')}")
                    print(f"   Innovation Score: {getattr(selected_idea, 'innovation_score', 'N/A')}")
                    print(f"   Total Score: {getattr(selected_idea, 'total_score', 'N/A')}")
                else:
                    print(f"   Selected Idea: {selected_idea.get('title', 'Unknown')}")
                    print(f"   Total Score: {selected_idea.get('total_score', 'N/A')}")
            
            output = result.get('output', {})
            print(f"   Selection Reasoning: {output.get('selection_reasoning', 'N/A')[:200]}...")
            
            if output.get('implementation_roadmap'):
                print(f"   Implementation Roadmap: {len(output['implementation_roadmap'])} steps")
            if output.get('risk_mitigation'):
                print(f"   Risk Mitigation: {len(output['risk_mitigation'])} strategies")
            
            print(f"   Feedback: {result['feedback'][:200]}...")
    
    print("\n5. Test Summary...")
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
    
    print("\n✅ ProblemAgent test completed successfully!")
    print(f"Workspace preserved at: {workspace_dir}")