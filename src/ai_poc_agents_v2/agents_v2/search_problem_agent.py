"""Search Problem Agent for AI-PoC-Agents-v2."""

from typing import Dict, Any, List, Optional
import json
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from ..agents.base_agent import BaseAgent
from ..core.state import PoCState
from ..tools.qiita_search import QiitaSemanticSearchTool

# Load environment variables
load_dotenv()


class SearchProblemAgent(BaseAgent):
    """Agent responsible for searching problem domain from multiple sources."""
    
    def __init__(self, config):
        """Initialize SearchProblemAgent with multiple search sources."""
        super().__init__("search_problem", config)
        
        # Initialize Qiita search tool
        qiita_access_token = os.getenv("QIITA_ACCESS_TOKEN")
        self.qiita_tool = QiitaSemanticSearchTool(access_token=qiita_access_token)
        self.qiita_enabled = False  # Disabled to focus on local code examples
        
        # Initialize local code search capabilities
        self.local_code_paths = []  # Will be populated from project settings
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        return """
You are an expert Problem Domain Search Agent for PoC development.

Your responsibilities:
- MULTI-SOURCE SEARCH: Search problem domain from Qiita, local codebase, and sample repositories
- TECHNICAL RESEARCH: Find relevant technical approaches, patterns, and best practices
- CODE COLLECTION: Gather sample code snippets and implementation examples
- INSIGHT EXTRACTION: Extract actionable technical insights from search results
- REFERENCE PREPARATION: Prepare structured references for implementation phase

Key principles:
- Cast a wide net for comprehensive research
- Focus on practical, implementable solutions
- Collect concrete code examples and patterns
- Identify proven approaches and common pitfalls
- Prepare materials that directly support implementation

Always provide structured, actionable research results with concrete code examples.
"""
    
    def execute_phase(self, state: PoCState) -> Dict[str, Any]:
        """Execute problem domain search from multiple sources."""
        
        project = state["project"]
        context = self._get_project_context(state)
        previous_results = self._get_previous_results(state)
        
        # Get problem analysis from previous phase
        problem_agent_memory = state["agent_memory"].get("problem_identification", {})
        problem_analysis = problem_agent_memory.get("problem_analysis", {})
        
        if not problem_analysis:
            return {
                "success": False,
                "score": 0.0,
                "output": {},
                "feedback": "No problem analysis found. Please run problem identification first.",
                "artifacts": [],
                "memory": {}
            }
        
        # Extract search keywords
        search_keywords = self._extract_search_keywords(problem_analysis, project)
        
        # Initialize comprehensive search results
        search_results = {
            "qiita_results": {},
            "local_code_results": {},
            "sample_code_collection": [],
            "technical_approaches": [],
            "implementation_patterns": [],
            "best_practices": [],
            "potential_challenges": [],
            "recommended_technologies": [],
            "search_keywords_used": search_keywords,
            "sources_searched": []
        }
        
        # Phase 1: Search Qiita
        qiita_results = self._search_qiita(search_keywords, project)
        if qiita_results:
            search_results["qiita_results"] = qiita_results
            search_results["sources_searched"].append("qiita")
        
        # Phase 2: Search local codebase (if paths provided)
        local_results = self._search_local_code(search_keywords, project, state)
        if local_results:
            search_results["local_code_results"] = local_results
            search_results["sources_searched"].append("local_code")
        
        # Phase 3: Synthesize findings using LLM
        synthesis_results = self._synthesize_search_results(search_results, problem_analysis, project)
        search_results.update(synthesis_results)
        
        # Phase 4: Collect and prepare sample code
        sample_code = self._collect_sample_code(search_results)
        search_results["sample_code_collection"] = sample_code
        
        # Save comprehensive search results
        artifact_path = self._save_search_results(search_results, state)
        
        # Log search results
        self._log_search_summary(search_results)
        
        # Store results for next phases
        state["problem_search"] = search_results

        
        # Ensure sources_searched contains only strings
        sources_list = [str(source) for source in search_results['sources_searched']]
        
        feedback = f"""
Problem Domain Search Complete:
- Sources Searched: {', '.join(sources_list)}
- Qiita Articles: {len(search_results['qiita_results'].get('reference_articles', []))} analyzed
- Local Code Files: {len(search_results['local_code_results'].get('code_files', []))} analyzed
- Technical Approaches: {len(search_results['technical_approaches'])} identified
- Sample Code Snippets: {len(search_results['sample_code_collection'])} collected
- Implementation Patterns: {len(search_results['implementation_patterns'])} found
"""
        
        score = self._calculate_search_score(search_results)
        
        return {
            "success": True,
            "score": score,
            "output": search_results,
            "feedback": feedback,
            "artifacts": [artifact_path],
            "memory": {
                "search_results": search_results,
                "searched_at": datetime.now().isoformat()
            }
        }
    
    def _extract_search_keywords(self, problem_analysis: Dict[str, Any], project) -> List[str]:
        """Extract search keywords from problem analysis and project info."""
        keywords = []
        
        # From problem analysis - ensure all items are strings
        tech_challenges = problem_analysis.get("technical_challenges", [])
        if isinstance(tech_challenges, list):
            keywords.extend([str(item) for item in tech_challenges])
        
        tech_requirements = problem_analysis.get("technical_requirements", [])
        if isinstance(tech_requirements, list):
            keywords.extend([str(item) for item in tech_requirements])
        
        # From project
        if project.theme:
            keywords.append(str(project.theme))
        if project.description:
            keywords.append(str(project.description))
        if hasattr(project, 'task_type') and project.task_type:
            keywords.append(project.task_type.lower().replace('_', ' '))
        
        # Domain specific
        domain_context = problem_analysis.get("domain_context", "")
        if domain_context:
            keywords.append(str(domain_context))
        
        # Filter out empty strings and ensure all are strings
        keywords = [kw for kw in keywords if kw and isinstance(kw, str) and kw.strip()]
        
        return list(set(keywords))  # Remove duplicates
    
    def _search_qiita(self, keywords: List[str], project) -> Dict[str, Any]:
        """Search Qiita for technical approaches and solutions."""
        if not self.qiita_enabled:
            return {}
        
        if not self.qiita_tool:
            print("Qiita tool not initialized")
            return {}
        
        if not keywords or not project.theme:
            print("Missing keywords or project theme for Qiita search")
            return {}
        
        # Search for technical articles
        articles = self.qiita_tool.search_relevant_articles(
            project_theme=project.theme,
            technical_keywords=keywords[:10],
            max_articles=30
        )
        
        if not articles:
            return {}
        
        # Extract insights
        insights = self.qiita_tool.extract_implementation_insights(articles)
        
        if not insights:
            insights = {"common_technologies": {}, "code_patterns": []}
        
        return {
            "reference_articles": articles[:15],
            "common_technologies": insights.get("common_technologies", {}),
            "code_patterns": insights.get("code_patterns", []),
            "articles_analyzed": len(articles)
        }
    
    def _search_local_code(self, keywords: List[str], project, state: PoCState) -> Dict[str, Any]:
        """Search local codebase for relevant examples."""
        
        # Get local code paths from multiple sources
        local_code_paths = getattr(project, 'local_code_paths', [])
        if not local_code_paths:
            # Fallback to orchestrator paths if available
            local_code_paths = getattr(self, 'local_code_paths', [])
        
        if not local_code_paths:
            print("No local code paths available for search")
            return {}
        
        print(f"Searching local code paths: {local_code_paths}")
        
        # Search results
        code_files = []
        code_snippets = []
        dependency_patterns = []
        architectural_patterns = []
        
        # Search through each local path
        for path_str in local_code_paths:
            path = Path(path_str)
            if not path.exists():
                print(f"Path does not exist: {path}")
                continue
                
            print(f"Searching in: {path}")
            
            # Find all Python files
            python_files = list(path.glob("**/*.py"))
            print(f"Found {len(python_files)} Python files")
            
            for file_path in python_files:
                try:
                    # Read file content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check if any keywords match file content
                    content_lower = content.lower()
                    matching_keywords = [kw for kw in keywords if kw.lower() in content_lower]
                    
                    if matching_keywords or 'classification' in content_lower or 'classifier' in content_lower:
                        # Extract file info
                        file_info = {
                            "file_path": str(file_path),
                            "file_name": file_path.name,
                            "size": len(content),
                            "matching_keywords": matching_keywords,
                            "content_preview": content[:500]
                        }
                        code_files.append(file_info)
                        
                        # Extract code snippets (functions and classes)
                        snippets = self._extract_code_snippets(content, str(file_path))
                        code_snippets.extend(snippets)
                        
                        # Extract dependency patterns
                        deps = self._extract_dependencies(content)
                        if deps:
                            dependency_patterns.extend(deps)
                        
                        print(f"Added relevant file: {file_path.name}")
                
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                    continue
        
        # Extract architectural patterns from code structure
        architectural_patterns = self._analyze_architectural_patterns(code_files)
        
        print(f"Local search results: {len(code_files)} files, {len(code_snippets)} snippets")
        
        return {
            "code_files": code_files,
            "code_snippets": code_snippets,
            "dependency_patterns": list(set(dependency_patterns)),
            "architectural_patterns": architectural_patterns
        }
    
    def _synthesize_search_results(
        self, 
        search_results: Dict[str, Any], 
        problem_analysis: Dict[str, Any], 
        project
    ) -> Dict[str, Any]:
        """Synthesize search results using LLM analysis."""
        
        # Prepare synthesis prompt
        qiita_summary = ""
        if search_results.get("qiita_results"):
            qiita_articles = search_results["qiita_results"].get("reference_articles", [])
            for article in qiita_articles[:10]:
                article_summary = ""
                article_summary += f"Title: {article.get('title', '')}\n"
                article_tags = [t.get("name", "") for t in article.get('tags', [])]
                article_summary += f"Tags: {', '.join(article_tags)}\n"
                article_summary += f"{article.get('body', '')[:500]}\n"
                
                qiita_summary += article_summary
        
        synthesis_prompt = f"""
Based on the following research results for {project.theme}, extract and synthesize:

PROBLEM ANALYSIS:
{json.dumps(problem_analysis, indent=2, ensure_ascii=False)}

QIITA RESEARCH RESULTS:
{qiita_summary}

TASK: Synthesize the research findings into actionable technical guidance:

1. TECHNICAL APPROACHES: Different ways to solve similar problems (include specific methodologies and frameworks)
2. IMPLEMENTATION PATTERNS: Common code patterns and architectural approaches
3. BEST PRACTICES: Recommended patterns, methodologies, and coding practices
4. POTENTIAL CHALLENGES: Common issues, gotchas, and mitigation strategies
5. RECOMMENDED TECHNOLOGIES: Specific libraries, frameworks, and tools that are proven effective

Focus on practical, implementable guidance that will directly support PoC development.
Prioritize approaches that match the identified task type: {getattr(project, 'task_type', 'OTHER')}
"""
        
        schema = {
            "technical_approaches": "list",
            "implementation_patterns": "list",
            "best_practices": "list",
            "potential_challenges": "list",
            "recommended_technologies": "list"
        }
        
        synthesis = self._generate_structured_response(
            self.get_system_prompt(),
            synthesis_prompt,
            schema
        )
        
        return synthesis
    
    def _collect_sample_code(self, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect and structure sample code from search results."""
        sample_code = []
        
        # From Qiita articles
        qiita_results = search_results.get("qiita_results", {})
        articles = qiita_results.get("reference_articles", [])
        
        for article in articles[:5]:  # Top 5 articles
            if article.get("body"):
                # Extract code blocks from article body
                code_blocks = self._extract_code_blocks(article.get("body", ""))
                for i, code_block in enumerate(code_blocks):
                    sample_code.append({
                        "source": "qiita",
                        "article_title": article.get("title", ""),
                        "article_url": article.get("url", ""),
                        "code_snippet": code_block,
                        "language": self._detect_language(code_block),
                        "snippet_id": f"qiita_{article.get('id', 'unknown')}_{i}"
                    })
        
        # From local code (if available)
        local_results = search_results.get("local_code_results", {})
        code_snippets = local_results.get("code_snippets", [])
        
        for snippet in code_snippets:
            sample_code.append({
                "source": "local",
                "file_path": snippet.get("file_path", ""),
                "code_snippet": snippet.get("code", ""),
                "language": snippet.get("language", ""),
                "snippet_id": f"local_{snippet.get('file_path', '').replace('/', '_')}"
            })
        
        return sample_code[:20]  # Limit to 20 samples
    
    def _extract_code_blocks(self, text: str) -> List[str]:
        """Extract code blocks from markdown text."""
        import re
        
        # Find code blocks marked with ```
        code_pattern = r'```[\w]*\n(.*?)```'
        matches = re.findall(code_pattern, text, re.DOTALL)
        
        # Filter out very short code blocks
        return [match.strip() for match in matches if len(match.strip()) > 20]
    
    def _detect_language(self, code: str) -> str:
        """Detect programming language from code snippet."""
        # Simple heuristics for language detection
        if 'import ' in code and ('pandas' in code or 'numpy' in code):
            return 'python'
        elif 'def ' in code or 'import ' in code:
            return 'python'
        elif 'function' in code or 'const' in code or 'let' in code:
            return 'javascript'
        elif 'public class' in code or 'import java' in code:
            return 'java'
        else:
            return 'unknown'
    
    def _save_search_results(self, search_results: Dict[str, Any], state: PoCState) -> str:
        """Save comprehensive search results as artifact."""
        content = json.dumps(search_results, indent=2, ensure_ascii=False, default=str)
        return self._save_artifact(
            content,
            f"problem_search_iteration_{state['iteration']}.json",
            state
        )
    
    def _log_search_summary(self, search_results: Dict[str, Any]) -> None:
        """Log search summary with color formatting."""
        print("\033[92m" + "="*60)  # Green color
        print("🔍 PROBLEM DOMAIN SEARCH RESULTS")
        print("="*60)
        print(f"📊 Sources: {', '.join(search_results['sources_searched'])}")
        print(f"📄 Qiita Articles: {len(search_results['qiita_results'].get('reference_articles', []))}")
        print(f"💻 Code Files: {len(search_results['local_code_results'].get('code_files', []))}")
        print(f"🛠️ Technical Approaches: {len(search_results['technical_approaches'])}")
        print(f"📋 Implementation Patterns: {len(search_results['implementation_patterns'])}")
        print(f"🔧 Sample Code Snippets: {len(search_results['sample_code_collection'])}")
        print(f"🏷️ Recommended Technologies: {len(search_results['recommended_technologies'])}")
        print("="*60 + "\033[0m")  # Reset color
    
    def _calculate_search_score(self, search_results: Dict[str, Any]) -> float:
        """Calculate quality score for search results."""
        score = 0.0
        
        # Base score for having any results
        if search_results.get("sources_searched"):
            score += 0.3
        
        # Score for Qiita results
        qiita_articles = len(search_results.get("qiita_results", {}).get("reference_articles", []))
        score += min(qiita_articles * 0.02, 0.3)
        
        # Score for technical approaches
        approaches = len(search_results.get("technical_approaches", []))
        score += min(approaches * 0.05, 0.2)
        
        # Score for sample code collection
        sample_code = len(search_results.get("sample_code_collection", []))
        score += min(sample_code * 0.01, 0.2)
        
        return min(score, 1.0)
    
    def _extract_code_snippets(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Extract code snippets (functions and classes) from file content."""
        import re
        snippets = []
        
        # Extract function definitions
        func_pattern = r'def\s+(\w+)\s*\([^)]*\):\s*\n((?:\s{4,}.*\n)*)'
        functions = re.finditer(func_pattern, content, re.MULTILINE)
        
        for match in functions:
            func_name = match.group(1)
            func_body = match.group(0)
            if len(func_body) > 50:  # Only include substantial functions
                snippets.append({
                    "type": "function",
                    "name": func_name,
                    "code": func_body[:1000],  # Limit size
                    "file_path": file_path,
                    "language": "python"
                })
        
        # Extract class definitions
        class_pattern = r'class\s+(\w+)[\s\S]*?(?=\nclass|\nif __name__|\Z)'
        classes = re.finditer(class_pattern, content, re.MULTILINE)
        
        for match in classes:
            class_content = match.group(0)
            class_name = re.search(r'class\s+(\w+)', class_content).group(1)
            if len(class_content) > 100:  # Only include substantial classes
                snippets.append({
                    "type": "class",
                    "name": class_name,
                    "code": class_content[:1500],  # Limit size
                    "file_path": file_path,
                    "language": "python"
                })
        
        return snippets
    
    def _extract_dependencies(self, content: str) -> List[str]:
        """Extract import dependencies from file content."""
        import re
        dependencies = []
        
        # Extract imports
        import_patterns = [
            r'import\s+([^\s,]+)',
            r'from\s+([^\s,]+)\s+import',
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            dependencies.extend(matches)
        
        # Filter out standard library imports
        external_deps = []
        common_external = ['pandas', 'numpy', 'sklearn', 'tensorflow', 'torch', 'matplotlib', 'seaborn', 'scipy']
        
        for dep in dependencies:
            if any(ext in dep for ext in common_external):
                external_deps.append(dep)
        
        return external_deps
    
    def _analyze_architectural_patterns(self, code_files: List[Dict[str, Any]]) -> List[str]:
        """Analyze architectural patterns from code files."""
        patterns = []
        
        # Check for common patterns based on file names and content
        for file_info in code_files:
            file_name = file_info["file_name"].lower()
            content_preview = file_info.get("content_preview", "").lower()
            
            # Detect common patterns
            if "classifier" in file_name or "classification" in content_preview:
                patterns.append("Classification Model Pattern")
            
            if "generic" in file_name:
                patterns.append("Generic Implementation Pattern")
            
            if "binary" in file_name:
                patterns.append("Binary Classification Pattern")
            
            if "comparison" in file_name or "compare" in content_preview:
                patterns.append("Model Comparison Pattern")
            
            if "argparse" in content_preview:
                patterns.append("Command Line Interface Pattern")
            
            if "train_test_split" in content_preview:
                patterns.append("ML Training Pipeline Pattern")
        
        return list(set(patterns))
    
    def set_local_code_paths(self, paths: List[str]) -> None:
        """Set local code search paths."""
        self.local_code_paths = [Path(path) for path in paths if Path(path).exists()]