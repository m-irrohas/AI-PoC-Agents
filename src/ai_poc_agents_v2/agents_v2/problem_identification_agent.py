"""Problem Identification Agent for AI-PoC-Agents-v2."""

from typing import Dict, Any
import json
from datetime import datetime

from ..agents.base_agent import BaseAgent
from ..core.state import PoCState


class ProblemIdentificationAgent(BaseAgent):
    """Agent responsible for problem identification and analysis."""
    
    def __init__(self, config):
        """Initialize ProblemIdentificationAgent."""
        super().__init__("problem_identification", config)
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        return """
You are an expert Problem Identification Agent for PoC (Proof of Concept) development.

Your responsibility:
- PROBLEM IDENTIFICATION: Analyze user themes to identify specific, actionable problems
- PROBLEM ANALYSIS: Break down complex problems into manageable components
- CONTEXT ANALYSIS: Understand domain, stakeholders, and constraints
- SUCCESS CRITERIA: Define measurable success metrics
- TASK TYPE CLASSIFICATION: Identify the primary technical task type

Key principles:
- Focus on practical, implementable solutions
- Consider technical feasibility, business impact, and innovation potential
- Provide clear reasoning for recommendations
- Balance ambition with realistic constraints
- Think from multiple perspectives (technical, business, user)

Always provide structured, actionable outputs that guide the next phase of PoC development.
"""
    
    def execute_phase(self, state: PoCState) -> Dict[str, Any]:
        """Execute problem identification and analysis."""
        
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
        
        # Investigate sample data if provided
        sample_data_info = self._investigate_sample_data(state)
        if sample_data_info:
            response["sample_data_info"] = sample_data_info
        
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
        print("🎯 PROBLEM IDENTIFICATION RESULTS")
        print("="*60)
        print(f"📊 Core Problem: {response.get('core_problem', 'Not identified')}")
        print(f"🏷️ Task Type: {project.task_type}")
        print(f"💡 Reasoning: {project.task_type_reasoning}")
        print(f"👥 Stakeholders: {len(response.get('stakeholders', []))}")
        print(f"🔧 Technical Requirements:")
        for req in project.technical_requirements[:3]:
            print(f"   - {req}")
        print("="*60 + "\033[0m")  # Reset color
        
        # Save analysis as artifact
        analysis_content = json.dumps(response, indent=2, ensure_ascii=False)
        artifact_path = self._save_artifact(
            analysis_content,
            f"problem_identification_iteration_{state['iteration']}.json",
            state
        )
        
        # Prepare feedback for next agent
        feedback = f"""
Problem Identification Complete:
- Core Problem: {response.get('core_problem', 'Not identified')}
- Task Type: {project.task_type}
- Stakeholders: {len(response.get('stakeholders', []))} identified
- Sub-problems: {len(response.get('sub_problems', []))} identified  
- Success Criteria: {len(response.get('success_criteria', []))} defined
- Technical Challenges: {len(response.get('technical_challenges', []))} identified
"""
        
        # Store sample data info in state for later use
        if sample_data_info:
            state["sample_data_info"] = sample_data_info
            print(f"💾 Sample data info stored in state: {sample_data_info['file_name']}")
        
        # Calculate score based on completeness and quality
        score = self._calculate_problem_analysis_score(response)
        
        return {
            "success": True,
            "score": score,
            "output": response,
            "feedback": feedback,
            "artifacts": [artifact_path],
            "memory": {
                "problem_analysis": response,
                "sample_data_info": sample_data_info,
                "identified_at": datetime.now().isoformat()
            }
        }
    
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
    
    def _investigate_sample_data(self, state: PoCState) -> Dict[str, Any]:
        """Investigate sample data and return analysis for use in implementation."""
        sample_data_path = state.get("sample_data_path", "")
        
        if not sample_data_path:
            return {}
        
        from pathlib import Path
        import pandas as pd
        
        path = Path(sample_data_path)
        
        if not path.exists():
            print(f"⚠️ Sample data path does not exist: {sample_data_path}")
            return {}
        
        print(f"📁 Investigating sample data: {sample_data_path}")
        
        # Initialize sample data info
        sample_data_info = {
            "source_path": str(path.resolve()),
            "is_directory": path.is_dir(),
            "file_exists": True
        }
        
        # If it's a directory, investigate all files
        if path.is_dir():
            print(f"📂 Directory detected, investigating all files...")
            files_info = []
            
            for file_path in path.iterdir():
                if file_path.is_file():
                    file_info = self._investigate_single_file(file_path)
                    files_info.append(file_info)
                    print(f"   📄 {file_path.name}: {file_info.get('file_type', 'unknown')}")
            
            sample_data_info["files"] = files_info
            sample_data_info["file_count"] = len(files_info)
            
            # Find overview.txt for problem context
            overview_file = path / "overview.txt"
            if overview_file.exists():
                try:
                    with open(overview_file, 'r', encoding='utf-8') as f:
                        overview_content = f.read()
                    sample_data_info["overview"] = overview_content
                    print(f"✅ Overview file found and read")
                except Exception as e:
                    print(f"⚠️ Could not read overview.txt: {e}")
            
            # Find training data file for implementation
            train_file = None
            for file_info in files_info:
                if 'train' in file_info['file_name'].lower() and file_info.get('file_type') == 'csv':
                    train_file = file_info
                    break
            
            if train_file:
                sample_data_info["train_data"] = train_file
                print(f"✅ Training data identified: {train_file['file_name']}")
            
        else:
            # Single file investigation
            file_info = self._investigate_single_file(path)
            sample_data_info.update(file_info)
        
        return sample_data_info
    
    def _investigate_single_file(self, file_path) -> Dict[str, Any]:
        """Investigate a single file and return its analysis."""
        from pathlib import Path
        import pandas as pd
        
        file_path = Path(file_path)
        
        file_info = {
            "file_path": str(file_path.resolve()),
            "file_name": file_path.name,
            "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
            "file_type": file_path.suffix.lower()[1:] if file_path.suffix else "unknown"
        }
        
        # Special handling for different file types
        if file_path.suffix.lower() == '.csv':
            try:
                # Read first few rows to understand structure
                df_sample = pd.read_csv(file_path, nrows=100)
                
                file_info.update({
                    "file_type": "csv",
                    "num_columns": len(df_sample.columns),
                    "column_names": list(df_sample.columns),
                    "sample_rows": len(df_sample),
                    "data_types": df_sample.dtypes.astype(str).to_dict(),
                    "missing_values": df_sample.isnull().sum().to_dict(),
                    "sample_preview": df_sample.head(3).to_dict('records')
                })
                
                # Try to get total rows
                try:
                    df_full = pd.read_csv(file_path)
                    file_info["total_rows"] = len(df_full)
                    file_info["target_column_candidates"] = [
                        col for col in df_full.columns 
                        if col.lower() in ['label', 'target', 'class', 'category', 'y', 'attack_type']
                    ]
                    
                    # If last column looks like target
                    last_col = df_full.columns[-1]
                    if last_col not in file_info["target_column_candidates"]:
                        file_info["target_column_candidates"].append(last_col)
                        
                except Exception as e:
                    file_info["total_rows"] = "unknown (large file)"
                
            except Exception as e:
                file_info["analysis_error"] = str(e)
        
        elif file_path.suffix.lower() == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                file_info["content"] = content[:1000]  # First 1000 chars
                file_info["content_lines"] = len(content.split('\n'))
            except Exception as e:
                file_info["read_error"] = str(e)
        
        return file_info