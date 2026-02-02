#!/usr/bin/env python3
"""
Unified Evaluation Script for DDR_Bench.

Single entry point for evaluating agent results across all scenarios:
- MIMIC: Evaluate medical insights against QA pairs
- 10-K: Evaluate financial insights against QA pairs
- GLOBEM: Evaluate behavioral insights against QA pairs

Usage:
    python run_evaluation.py --scenario mimic
    python run_evaluation.py --scenario 10k
    python run_evaluation.py --scenario globem

See README.md for detailed usage instructions.
"""

import argparse
import logging
import os
from pathlib import Path

from config import get_config
from evaluate import UnifiedEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="DDR_Bench Unified Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate MIMIC results (using config settings)
  python run_evaluation.py --scenario mimic

  # Evaluate 10-K results with custom logs
  python run_evaluation.py --scenario 10k --log-dir ./10k_logs
        """
    )
    
    # Required arguments
    parser.add_argument("--scenario", required=True, choices=["mimic", "10k", "globem"],
                        help="Evaluation scenario")
    
    # Path overrides
    # parser.add_argument("--qa-file", help="Path to QA file (overrides config)")  # Removed to enforce config usage
    # parser.add_argument("--log-dir", help="Path to agent logs directory (overrides config)") # Removed to enforce config usage
    parser.add_argument("--output", "-o", help="Output file path for results")
    
    # Execution options
    parser.add_argument("--test-mode", "-t", action="store_true",
                        help="Run in test mode (process only first entity)")
    
    # Configuration file
    parser.add_argument("--config", help="Path to config.yaml file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(args.config)
    scenario_config = config.get_scenario(args.scenario)
    
    # Get paths from config (CLI overrides removed)
    qa_file = scenario_config.qa_file
    log_dir = scenario_config.log_dir
    
    if not qa_file:
        parser.error(f"qa_file not found in config.yaml for scenario {args.scenario}. Please check your config.")
    if not log_dir:
        parser.error(f"log_dir not found in config.yaml for scenario {args.scenario}. Please check your config.")
    
    # Determine output file
    output_file = args.output
    if not output_file:
        log_dir_name = Path(log_dir).name
        output_file = f"./{args.scenario}_{log_dir_name}_evaluation_result.json"
    
    # Resolve evaluation parameters from CONFIG (no CLI overrides for these)
    provider = config.evaluation.provider or "azure"
    model = config.evaluation.model or "gpt-5-mini"
    max_retries = config.evaluation.max_retries or 5
    retry_delay = config.evaluation.retry_delay or 2.0
    log_level = config.agent.log_level or "INFO"
    
    # Set log level for current process
    os.environ["DDR_LOG_LEVEL"] = log_level
    logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    vllm_host = "localhost" # Assuming usage of configured VLLM credentials if needed, or default
    vllm_port = config.provider.vllm_port or 8000
    
    # Build vLLM URL
    vllm_url = f"http://{vllm_host}:{vllm_port}/v1/chat/completions"
    
    print(f"\n{'='*60}")
    print(f"DDR_Bench Evaluation")
    print(f"Scenario: {args.scenario}")
    print(f"QA File: {qa_file}")
    print(f"Log Directory: {log_dir}")
    print(f"Output: {output_file}")
    print(f"Judge Provider: {provider}")
    print(f"Judge Model: {model}")
    print(f"Config File: {args.config or 'config.yaml'}")
    if args.test_mode:
        print("Mode: TEST (first entity only)")
    print(f"{'='*60}\n")
    
    # Create unified evaluator
    evaluator = UnifiedEvaluator(
        scenario=args.scenario,
        vllm_url=vllm_url,
        provider=provider,
        openai_model=model,
        azure_model=model,
        max_retries=max_retries,
        retry_delay=retry_delay
    )
    
    # Run evaluation
    evaluator.run_evaluation(
        qa_file=qa_file,
        logs_dir=log_dir,
        output_file=output_file,
        test_mode=args.test_mode
    )
    
    print(f"\nEvaluation complete. Results saved to: {output_file}")


if __name__ == "__main__":
    main()
