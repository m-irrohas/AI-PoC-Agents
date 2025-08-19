"""PoC Design Agent for AI-PoC-Agents-v2."""

from typing import Dict, Any, List
import json
from datetime import datetime

from ..agents.base_agent import BaseAgent
from ..core.state import PoCState, PoCImplementation


class PoCDesignAgent(BaseAgent):
    """Agent responsible for technical analysis and requirements definition for the selected PoC idea."""
    
    def __init__(self, config):
        """Initialize PoCDesignAgent."""
        super().__init__("poc_designer", config)
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        return """
You are an expert PoC Design Agent for technical analysis and architecture design.

Your responsibilities:
- TECHNICAL ARCHITECTURE: Design detailed technical architecture for the selected PoC idea
- REQUIREMENTS SPECIFICATION: Define comprehensive functional and non-functional requirements
- TECHNOLOGY STACK: Finalize technology stack and justify choices
- SYSTEM DESIGN: Create system components, data flows, and integration patterns
- IMPLEMENTATION ROADMAP: Plan detailed implementation phases and dependencies
- QUALITY STANDARDS: Define testing, validation, and quality assurance approaches

Key principles:
- Create implementable and realistic technical specifications
- Balance technical excellence with PoC constraints (time, scope, resources)
- Leverage sample code patterns and best practices from research
- Define clear acceptance criteria and success metrics
- Consider scalability, maintainability, and extensibility
- Address security, performance, and reliability requirements
- Provide concrete implementation guidance

Always produce detailed technical specifications that enable smooth implementation.
"""
    
    def execute_phase(self, state: PoCState) -> Dict[str, Any]:
        """Execute PoC design phase based on selected idea."""
        
        project = state["project"]
        context = self._get_project_context(state)
        previous_results = self._get_previous_results(state)
        
        # Get selected idea from previous phases
        selected_idea = state.get("selected_idea")
        if not selected_idea:
            return {
                "success": False,
                "score": 0.0,
                "output": {},
                "feedback": "No selected idea found. Please run idea selection first.",
                "artifacts": [],
                "memory": {}
            }
        
        # Get inputs from previous agents
        idea_reflection = state["agent_memory"].get("idea_reflection", {})
        search_results = state["agent_memory"].get("search_problem", {}).get("search_results", {})
        
        # Prepare technical context
        technical_context = self._prepare_technical_context(selected_idea, search_results, idea_reflection)
        sample_code_context = self._prepare_sample_code_context(search_results)
        
        user_prompt = f"""
{context}

{previous_results}

SELECTED IDEA FOR DESIGN:
{json.dumps(self._idea_to_dict(selected_idea), indent=2, ensure_ascii=False)}

{technical_context}

{sample_code_context}

TASK: Create a comprehensive technical design and architecture specification for implementing this PoC idea.

Provide the following detailed design components:

1. ARCHITECTURE OVERVIEW
   - High-level system architecture diagram (textual description)
   - Core components and their relationships
   - Data flow and control flow
   - Integration points and interfaces
   - Deployment architecture

2. TECHNICAL SPECIFICATIONS
   - Finalized technology stack with justifications
   - Framework and library selections
   - Database and storage design
   - API design and endpoints
   - Security considerations
   - Performance requirements

3. FUNCTIONAL REQUIREMENTS
   - Detailed feature specifications
   - User stories and acceptance criteria
   - Input/output specifications
   - Business logic requirements
   - Integration requirements

4. NON-FUNCTIONAL REQUIREMENTS
   - Performance benchmarks
   - Scalability requirements
   - Security standards
   - Reliability and availability
   - Usability requirements
   - Maintainability standards

5. IMPLEMENTATION ROADMAP
   - Development phases and milestones
   - Task breakdown and dependencies
   - Resource allocation and timeline
   - Risk assessment and mitigation
   - Testing strategy and approach

6. QUALITY ASSURANCE
   - Unit testing requirements
   - Integration testing plan
   - Performance testing criteria
   - Security testing approach
   - Code quality standards
   - Documentation requirements

7. DEPLOYMENT STRATEGY
   - Environment setup and configuration
   - Deployment process and automation
   - Monitoring and logging
   - Backup and recovery
   - Maintenance procedures

8. SUCCESS METRICS
   - Technical KPIs and measurements
   - Business impact indicators
   - Quality metrics
   - Performance benchmarks
   - User satisfaction criteria

Focus on creating implementable specifications that leverage the available sample code and research insights.
Ensure all design decisions are justified and aligned with PoC constraints.
"""
        
        schema = {
            "architecture_overview": "dict",
            "technical_specifications": "dict", 
            "functional_requirements": "dict",
            "non_functional_requirements": "dict",
            "implementation_roadmap": "dict",
            "quality_assurance": "dict",
            "deployment_strategy": "dict",
            "success_metrics": "dict"
        }
        
        response = self._generate_structured_response(
            self.get_system_prompt(),
            user_prompt,
            schema
        )
        
        # Create PoCImplementation object with design specifications
        implementation = self._create_implementation_design(selected_idea, response, search_results)
        
        # Store implementation design in state
        state["implementation"] = implementation
        
        # Save design as artifact
        design_data = self._implementation_to_dict(implementation)
        artifact_path = self._save_json_artifact(
            design_data,
            f"poc_design_iteration_{state['iteration']}.json",
            state
        )
        
        # Create detailed design document
        design_doc = self._create_design_document(implementation, selected_idea, response)
        doc_path = self._save_artifact(
            design_doc,
            f"poc_design_document_iteration_{state['iteration']}.md",
            state
        )
        
        # Log design summary
        self._log_design_summary(implementation, selected_idea)
        
        feedback = f"""
PoC Design Complete:
- Selected Idea: {selected_idea.title}
- Architecture Complexity: {self._assess_architecture_complexity(response)}/5
- Technology Stack: {len(implementation.tech_stack)} technologies
- Implementation Phases: {len(implementation.environment_config.get('implementation_phases', []))}
- Test Cases Defined: {len(implementation.test_cases)}
- Dependencies: {len(implementation.dependencies)}
- Estimated Implementation Time: {self._estimate_implementation_time(response)} hours
"""
        
        score = self._calculate_design_score(response, implementation, selected_idea)
        
        return {
            "success": True,
            "score": score,
            "output": {"design_specification": design_data, "selected_idea_id": selected_idea.id},
            "feedback": feedback,
            "artifacts": [artifact_path, doc_path],
            "memory": {
                "design_specification": design_data,
                "architecture_complexity": self._assess_architecture_complexity(response),
                "implementation_phases": implementation.environment_config.get('implementation_phases', []),
                "technology_decisions": implementation.tech_stack,
                "designed_at": datetime.now().isoformat()
            }
        }
    
    def _prepare_technical_context(self, selected_idea, search_results: Dict[str, Any], idea_reflection: Dict[str, Any]) -> str:
        """Prepare technical context for design phase."""
        context = f"""
TECHNICAL DESIGN CONTEXT:

Selected Idea Details:
- Title: {selected_idea.title}
- Description: {selected_idea.description}
- Technical Approach: {getattr(selected_idea, 'technical_approach', 'Not specified')}
- Implementation Complexity: {selected_idea.implementation_complexity}/5
- Technology Stack: {getattr(selected_idea, 'technology_stack', [])}

Research-Based Technical Approaches:
{json.dumps(search_results.get('technical_approaches', []), indent=2, ensure_ascii=False)}

Best Practices from Research:
{json.dumps(search_results.get('best_practices', []), indent=2, ensure_ascii=False)}

Recommended Technologies:
{json.dumps(search_results.get('recommended_technologies', []), indent=2, ensure_ascii=False)}

Implementation Patterns:
{json.dumps(search_results.get('implementation_patterns', []), indent=2, ensure_ascii=False)}

Idea Reflection Insights:
{json.dumps(idea_reflection.get('reflection_insights', {}), indent=2, ensure_ascii=False)}
"""
        return context
    
    def _prepare_sample_code_context(self, search_results: Dict[str, Any]) -> str:
        """Prepare sample code context for design decisions."""
        sample_code = search_results.get('sample_code_collection', [])
        if not sample_code:
            return "No sample code available for reference."
        
        context = f"""
AVAILABLE SAMPLE CODE FOR DESIGN REFERENCE ({len(sample_code)} snippets):

Code Patterns and Examples:
"""
        
        # Group sample code by language/technology
        code_by_tech = {}
        for snippet in sample_code:
            lang = snippet.get('language', 'unknown')
            if lang not in code_by_tech:
                code_by_tech[lang] = []
            code_by_tech[lang].append(snippet)
        
        for lang, snippets in code_by_tech.items():
            context += f"\n{lang.upper()} Examples ({len(snippets)} snippets):\n"
            for i, snippet in enumerate(snippets[:3], 1):  # Show top 3 per language
                context += f"  {i}. {snippet.get('article_title', 'Unknown source')}\n"
                context += f"     Code: {snippet.get('code_snippet', '')[:150]}...\n"
        
        return context
    
    def _create_implementation_design(self, selected_idea, design_response: Dict[str, Any], search_results: Dict[str, Any]) -> PoCImplementation:
        """Create PoCImplementation object from design response."""
        implementation = PoCImplementation(idea_id=selected_idea.id)
        
        # Architecture and technical specifications
        arch_overview = design_response.get('architecture_overview', {})
        tech_specs = design_response.get('technical_specifications', {})
        
        implementation.architecture = {
            "overview": arch_overview,
            "specifications": tech_specs,
            "components": arch_overview.get('core_components', []),
            "data_flow": arch_overview.get('data_flow', ''),
            "integration_points": arch_overview.get('integration_points', [])
        }
        
        # Technology stack
        implementation.tech_stack = tech_specs.get('technology_stack', [])
        if isinstance(implementation.tech_stack, str):
            implementation.tech_stack = [implementation.tech_stack]
        
        # Dependencies
        implementation.dependencies = tech_specs.get('dependencies', [])
        
        # Environment configuration
        implementation.environment_config = {
            "functional_requirements": design_response.get('functional_requirements', {}),
            "non_functional_requirements": design_response.get('non_functional_requirements', {}),
            "implementation_phases": design_response.get('implementation_roadmap', {}).get('phases', []),
            "deployment_strategy": design_response.get('deployment_strategy', {}),
            "quality_standards": design_response.get('quality_assurance', {})
        }
        
        # Test cases from quality assurance
        qa_specs = design_response.get('quality_assurance', {})
        implementation.test_cases = [
            {"type": "unit", "description": "Unit testing requirements", "criteria": qa_specs.get('unit_testing', '')},
            {"type": "integration", "description": "Integration testing plan", "criteria": qa_specs.get('integration_testing', '')},
            {"type": "performance", "description": "Performance testing criteria", "criteria": qa_specs.get('performance_testing', '')}
        ]
        
        # Deployment instructions
        deploy_strategy = design_response.get('deployment_strategy', {})
        implementation.deployment_instructions = f"""
Environment Setup: {deploy_strategy.get('environment_setup', '')}
Deployment Process: {deploy_strategy.get('deployment_process', '')}
Configuration: {deploy_strategy.get('configuration', '')}
Monitoring: {deploy_strategy.get('monitoring', '')}
"""
        
        # Performance metrics from success criteria
        success_metrics = design_response.get('success_metrics', {})
        implementation.performance_metrics = {
            "technical_kpis": success_metrics.get('technical_kpis', ''),
            "performance_benchmarks": success_metrics.get('performance_benchmarks', ''),
            "quality_metrics": success_metrics.get('quality_metrics', '')
        }
        
        return implementation
    
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
            "code_files": implementation.code_files,
            "execution_logs": implementation.execution_logs
        }
    
    def _create_design_document(self, implementation: PoCImplementation, selected_idea, design_response: Dict[str, Any]) -> str:
        """Create comprehensive design document."""
        doc = f"""# PoC Design Specification

## Project Overview
- **Selected Idea**: {selected_idea.title}
- **Description**: {selected_idea.description}
- **Implementation Complexity**: {selected_idea.implementation_complexity}/5
- **Expected Impact**: {selected_idea.expected_impact}/5

## Architecture Overview
{json.dumps(design_response.get('architecture_overview', {}), indent=2)}

## Technical Specifications
{json.dumps(design_response.get('technical_specifications', {}), indent=2)}

## Functional Requirements
{json.dumps(design_response.get('functional_requirements', {}), indent=2)}

## Non-Functional Requirements
{json.dumps(design_response.get('non_functional_requirements', {}), indent=2)}

## Implementation Roadmap
{json.dumps(design_response.get('implementation_roadmap', {}), indent=2)}

## Quality Assurance Plan
{json.dumps(design_response.get('quality_assurance', {}), indent=2)}

## Deployment Strategy
{json.dumps(design_response.get('deployment_strategy', {}), indent=2)}

## Success Metrics
{json.dumps(design_response.get('success_metrics', {}), indent=2)}

## Technology Stack
{', '.join(implementation.tech_stack)}

## Dependencies
{', '.join(implementation.dependencies)}

## Test Cases
"""
        
        for i, test_case in enumerate(implementation.test_cases, 1):
            doc += f"\n### Test Case {i}: {test_case['type'].title()}\n"
            doc += f"**Description**: {test_case['description']}\n"
            doc += f"**Criteria**: {test_case['criteria']}\n"
        
        doc += f"""

## Deployment Instructions
{implementation.deployment_instructions}

---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return doc
    
    def _idea_to_dict(self, idea) -> Dict[str, Any]:
        """Convert idea object to dictionary for JSON serialization."""
        return {
            "id": idea.id,
            "title": idea.title,
            "description": idea.description,
            "technical_approach": getattr(idea, 'technical_approach', ''),
            "implementation_complexity": idea.implementation_complexity,
            "expected_impact": idea.expected_impact,
            "innovation_score": getattr(idea, 'innovation_score', 0),
            "feasibility_score": getattr(idea, 'feasibility_score', 0.0),
            "total_score": getattr(idea, 'total_score', 0.0),
            "estimated_effort_hours": getattr(idea, 'estimated_effort_hours', 0),
            "technology_stack": getattr(idea, 'technology_stack', [])
        }
    
    def _assess_architecture_complexity(self, design_response: Dict[str, Any]) -> int:
        """Assess the complexity of the proposed architecture (1-5 scale)."""
        complexity_indicators = 0
        
        # Check architecture overview complexity
        arch_overview = design_response.get('architecture_overview', {})
        components = arch_overview.get('core_components', [])
        if len(components) > 5:
            complexity_indicators += 1
        
        # Check technical specifications complexity
        tech_specs = design_response.get('technical_specifications', {})
        tech_stack = tech_specs.get('technology_stack', [])
        if isinstance(tech_stack, list) and len(tech_stack) > 3:
            complexity_indicators += 1
        
        # Check implementation roadmap complexity
        roadmap = design_response.get('implementation_roadmap', {})
        phases = roadmap.get('phases', [])
        if len(phases) > 3:
            complexity_indicators += 1
        
        # Check integration complexity
        integrations = arch_overview.get('integration_points', [])
        if len(integrations) > 2:
            complexity_indicators += 1
        
        # Check non-functional requirements complexity
        nfr = design_response.get('non_functional_requirements', {})
        if len(nfr) > 5:
            complexity_indicators += 1
        
        return min(complexity_indicators + 1, 5)
    
    def _estimate_implementation_time(self, design_response: Dict[str, Any]) -> int:
        """Estimate implementation time in hours."""
        base_hours = 16  # Base PoC implementation time
        
        # Add time based on architecture complexity
        complexity = self._assess_architecture_complexity(design_response)
        complexity_hours = complexity * 8
        
        # Add time based on technology stack size
        tech_specs = design_response.get('technical_specifications', {})
        tech_stack = tech_specs.get('technology_stack', [])
        tech_hours = len(tech_stack) * 4 if isinstance(tech_stack, list) else 4
        
        # Add time based on implementation phases
        roadmap = design_response.get('implementation_roadmap', {})
        phases = roadmap.get('phases', [])
        phase_hours = len(phases) * 6
        
        return base_hours + complexity_hours + tech_hours + phase_hours
    
    def _log_design_summary(self, implementation: PoCImplementation, selected_idea) -> None:
        """Log design summary with color formatting."""
        print("\033[96m" + "="*60)  # Cyan color
        print("🏗️ POC DESIGN SPECIFICATION")
        print("="*60)
        print(f"🎯 Selected Idea: {selected_idea.title}")
        print(f"🏛️ Architecture Components: {len(implementation.architecture.get('components', []))}")
        print(f"🔧 Technology Stack: {len(implementation.tech_stack)} technologies")
        print(f"📋 Test Cases: {len(implementation.test_cases)}")
        print(f"📦 Dependencies: {len(implementation.dependencies)}")
        print(f"🚀 Implementation Phases: {len(implementation.environment_config.get('implementation_phases', []))}")
        
        print(f"\n🛠️ Selected Technologies:")
        for tech in implementation.tech_stack[:5]:  # Show first 5
            print(f"   • {tech}")
        if len(implementation.tech_stack) > 5:
            print(f"   ... and {len(implementation.tech_stack) - 5} more")
        
        print("="*60 + "\033[0m")  # Reset color
    
    def _calculate_design_score(self, design_response: Dict[str, Any], implementation: PoCImplementation, selected_idea) -> float:
        """Calculate quality score for the design specification."""
        score = 0.0
        
        # Base score for having a complete design
        score += 0.3
        
        # Score for architecture completeness
        arch_overview = design_response.get('architecture_overview', {})
        if arch_overview and len(arch_overview) > 3:
            score += 0.15
        
        # Score for technical specifications completeness
        tech_specs = design_response.get('technical_specifications', {})
        if tech_specs and len(tech_specs) > 3:
            score += 0.15
        
        # Score for having test cases
        if len(implementation.test_cases) >= 3:
            score += 0.1
        
        # Score for implementation roadmap
        phases = implementation.environment_config.get('implementation_phases', [])
        if len(phases) >= 3:
            score += 0.1
        
        # Score for technology stack appropriateness
        if len(implementation.tech_stack) >= 2:
            score += 0.1
        
        # Score for deployment strategy
        if implementation.deployment_instructions and len(implementation.deployment_instructions) > 50:
            score += 0.1
        
        return min(score, 1.0)