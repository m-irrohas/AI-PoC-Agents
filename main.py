#!/usr/bin/env python3
"""Main entry point for AI-PoC-Agents-v2."""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_poc_agents_v2.core.config import Config
from ai_poc_agents_v2.core.state import create_initial_state, PoCProject
from ai_poc_agents_v2.agents_v2.workflow_orchestrator import WorkflowOrchestrator


def setup_logging(debug: bool = False) -> None:
    """Setup logging configuration."""
    
    level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('ai_poc_agents_v2.log')
        ]
    )


def main() -> None:
    """Main function."""
    
    parser = argparse.ArgumentParser(description="AI-PoC-Agents-v2 - Automated PoC Framework")
    
    parser.add_argument(
        "--theme",
        type=str,
        required=True,
        help="PoC theme or problem statement"
    )
    
    parser.add_argument(
        "--description", 
        type=str,
        default="",
        help="Detailed description of the PoC requirements"
    )
    
    parser.add_argument(
        "--domain",
        type=str,
        default="",
        help="Domain/industry context"
    )
    
    parser.add_argument(
        "--sample-data",
        type=str,
        default="",
        help="Path to sample data file or directory for PoC testing"
    )
    
    parser.add_argument(
        "--local-code-paths",
        type=str,
        nargs="*",
        default=[],
        help="Local code paths to search for implementation examples"
    )
    
    parser.add_argument(
        "--workspace",
        type=str,
        default="./examples",
        help="Workspace directory for artifacts"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-nano-2025-08-07",
        help="Default model to use for agents"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum iterations per phase"
    )
    
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.7,
        help="Minimum score threshold to proceed (0.0-1.0)"
    )
    
    parser.add_argument(
        "--timeline-days",
        type=int,
        default=7,
        help="Expected timeline for PoC in days"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--start-from-phase",
        type=str,
        help="Start workflow from specific phase (problem_identification, search_problem, idea_generation, etc.)"
    )
    
    parser.add_argument(
        "--single-phase",
        type=str,
        help="Execute only a single phase"
    )
    
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream workflow execution with real-time updates"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        if args.config:
            config = Config.from_yaml(Path(args.config))
        else:
            # Use default configuration
            config = Config()
            config.validate()
            
        # Override workflow settings
        config.workflow.max_iterations = args.max_iterations
        config.workflow.score_threshold = args.score_threshold
        
        # Setup workspace directory
        workspace_path = Path(args.workspace)
        workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Create project info
        project = PoCProject(
            theme=args.theme,
            description=args.description,
            domain=args.domain,
            timeline_days=args.timeline_days
        )
        
        logger.info(f"Starting PoC project: {project.theme}")
        logger.info(f"Domain: {project.domain}")
        logger.info(f"Timeline: {project.timeline_days} days")
        logger.info(f"Workspace: {workspace_path}")
        
        # Create initial state
        # Process sample data path
        sample_data_path = ""
        if args.sample_data:
            sample_path = Path(args.sample_data)
            if sample_path.exists():
                sample_data_path = str(sample_path.resolve())
                print(f"📁 Sample data path: {sample_data_path}")
            else:
                print(f"⚠️ Sample data path does not exist: {args.sample_data}")
        
        initial_state = create_initial_state(
            project=project,
            workspace_path=str(workspace_path),
            model_config={
                "problem_identification": args.model,
                "search_problem": args.model,
                "idea_generation": args.model,
                "idea_reflection": args.model,
                "poc_design": args.model,
                "implementation": args.model,
                "execute": args.model,
                "reflection": args.model,
                "reporting": args.model
            },
            max_iterations=config.workflow.max_iterations,
            score_threshold=config.workflow.score_threshold,
            sample_data_path=sample_data_path
        )
        
        # Create workflow orchestrator
        orchestrator = WorkflowOrchestrator(config)
        
        # Set local code paths if provided
        if args.local_code_paths:
            valid_paths = [path for path in args.local_code_paths if Path(path).exists()]
            if valid_paths:
                orchestrator.set_local_code_paths(valid_paths)
                print(f"🔍 Local code search paths: {valid_paths}")
            else:
                print("⚠️ No valid local code paths found")
        
        # Run workflow
        logger.info("Starting AI-PoC-Agents-v2 specialized agent workflow...")
        
        if args.single_phase:
            # Execute single phase
            print(f"\n🎯 Executing single phase: {args.single_phase}")
            final_state = orchestrator.execute_single_phase(initial_state, args.single_phase)
        elif args.start_from_phase:
            # Start from specific phase
            print(f"\n▶️ Starting workflow from phase: {args.start_from_phase}")
            final_state = orchestrator.execute_workflow(initial_state, start_from_phase=args.start_from_phase)
        else:
            # Run complete workflow
            print("\n🚀 Starting complete PoC Development Workflow...")
            final_state = orchestrator.execute_workflow(initial_state)
        
        # Report results
        print("\n" + "=" * 60)
        print("🎉 PoC WORKFLOW COMPLETED")
        print("=" * 60)
        
        # Get workflow status from orchestrator
        status = orchestrator.get_workflow_status(final_state)
        
        print(f"\n📈 RESULTS SUMMARY:")
        print(f"  • Project: {project.theme}")
        print(f"  • Completed Phases: {status['completed_phases']}/{status['total_phases']}")
        print(f"  • Overall Score: {status['overall_score']:.3f}/1.0")
        print(f"  • Current Phase: {status['current_phase']}")
        print(f"  • Total Artifacts: {status['total_artifacts']}")
        print(f"  • Workflow Completed: {'Yes' if status['workflow_completed'] else 'No'}")
        
        if status['error_message']:
            print(f"  ⚠️  Error: {status['error_message']}")
        
        print(f"\n📁 PHASE STATUS:")
        for phase_info in status['phase_status']:
            status_emoji = {"completed": "✅", "in_progress": "🔄", "pending": "⏳"}.get(phase_info['status'], "❓")
            score_text = f"{phase_info['score']:.3f}" if phase_info['score'] is not None else "N/A"
            print(f"  {status_emoji} {phase_info['phase']}: {score_text}/1.0 ({len(phase_info['artifacts'])} artifacts)")
        
        print(f"\n📂 WORKSPACE: {workspace_path}")
        print(f"  • Check the workspace directory for generated artifacts")
        print(f"  • Implementation code and reports are available")
        
        # Save final state (create simple JSON-serializable version)
        state_data = {
            "project_theme": project.theme,
            "workspace_path": str(workspace_path),
            "workflow_status": status,
            "completed_at": datetime.now().isoformat(),
            "agent_memory_keys": list(final_state.get("agent_memory", {}).keys()),
            "total_artifacts": len(final_state.get("artifacts", [])),
            "phase_count": len(final_state.get("phase_results", []))
        }
        
        state_path = workspace_path / "workflow_summary.json"
        import json
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
        print(f"  • Workflow summary saved to: {state_path}")
        
        # Determine success
        if final_state.get("error_message"):
            print(f"\n❌ FAILURE: PoC workflow encountered issues")
            print(f"   Error: {final_state['error_message']}")
            return_code = 1
        else:
            if status['overall_score'] >= 0.7:
                print(f"\n✅ SUCCESS: PoC completed successfully!")
                print(f"   Recommendation: Proceed with development")
            elif status['completed_phases'] >= 6:  # Most phases completed
                print(f"\n⚠️  PARTIAL SUCCESS: PoC completed with room for improvement")
                print(f"   Recommendation: Review results and consider refinements")
            else:
                print(f"\n🔄 INCOMPLETE: Workflow stopped early")
                print(f"   Recommendation: Check error logs and restart if needed")
            
            return_code = 0
        
        print("\n" + "=" * 60)
        sys.exit(return_code)
        
    except KeyboardInterrupt:
        logger.info("Workflow interrupted by user")
        print("\n\n⏸️  Workflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"\n\n❌ Unexpected error: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()