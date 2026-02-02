#!/usr/bin/env python3
"""
Unified Agent Runner for DDR_Bench.

Single entry point for running data analysis agents across all scenarios:
- MIMIC: Patient data analysis using MIMIC-IV database
- 10-K: Financial report analysis using SEC 10-K filings
- GLOBEM: Behavioral data analysis using GLOBEM dataset

Usage:
    python run_agent.py --scenario mimic --db-path /path/to/mimic_iv.db --input /path/to/notes.json
    python run_agent.py --scenario 10k --db-path /path/to/10k.db
    python run_agent.py --scenario globem --data-path /path/to/globem/data

See README.md for detailed usage instructions.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from config import Config, get_config
from base_batch_analyzer import BaseBatchAnalyzer


class PatientBatchAnalyzer(BaseBatchAnalyzer):
    """Batch analyzer for MIMIC patient data."""
    
    def __init__(self, base_log_dir: str, target_ids: Optional[Set[str]] = None, overwrite: bool = False):
        super().__init__(base_log_dir, target_ids, overwrite)

    def extract_identifiers(self, source_file: Path) -> List[Dict[str, Any]]:
        """Extract patient identifiers from pre-defined ID list file."""
        try:
            print(f"Reading patient IDs from: {source_file}")
            with open(source_file, 'r', encoding='utf-8') as f:
                patient_ids = json.load(f)
            
            if not isinstance(patient_ids, list):
                print(f"   Error: Expected a list of patient IDs")
                return []
            
            patients_list = []
            for pid in patient_ids:
                subject_id = str(pid)
                patients_list.append({
                    "patient_id": f"patient_{subject_id}",
                    "subject_id": subject_id,
                    "identifier": subject_id,
                    "data": {}
                })
            
            print(f"   Found {len(patients_list)} patients")
            return patients_list
            
        except Exception as e:
            print(f"   Error reading file: {e}")
            return []
    
    def _prepare_analysis_command(self, identifier_info: Dict[str, Any], source_file: Path,
                                  subdir_name: str, **kwargs) -> tuple:
        """Prepare the command for patient analysis."""
        subject_id = identifier_info["subject_id"]
        
        patient_log_dir = self.base_log_dir / subdir_name
        patient_log_dir.mkdir(parents=True, exist_ok=True)
        
        task = f"Analyze patient {subject_id}"
        
        cmd = [
            sys.executable,
            "agent/data_agent.py",
            "--task", task,
            "--log-dir", str(patient_log_dir)
        ]
        
        if not kwargs.get("auto_finish", True):
            cmd.append("--no-auto-finish")
        
        # Pass config info
        if kwargs.get("config_path"):
            cmd.extend(["--config", kwargs.get("config_path")])
        if kwargs.get("scenario"):
            cmd.extend(["--scenario", kwargs.get("scenario")])

        # MCP arguments: Only pass server script, agent will load config for DB path
        cmd.extend(["--sql-server", "tool_server/sqlite_mcp.py"])
        
        # We do NOT pass --data-path override unless specific need, but user said NO overrides.
        # So we trust config.yaml loaded by sqlite_mcp.py via --config
        
        # Pass max_turns if provided
        if kwargs.get("max_turns"):
            cmd.extend(["--max-turns", str(kwargs.get("max_turns"))])
            
        env = os.environ.copy()
        env['CUSTOM_LOG_DIR'] = str(patient_log_dir)
        
        return cmd, env, f"Patient {subject_id}"
    
    def get_subdir_name(self, identifier: str) -> str:
        return f"patient_{identifier}"
    
    def _create_identifier_from_logs(self, identifier: str, dirname: str) -> Optional[Dict[str, Any]]:
        return {
            "patient_id": f"patient_{identifier}",
            "subject_id": identifier,
            "identifier": identifier,
            "data": {}
        }


class CompanyBatchAnalyzer(BaseBatchAnalyzer):
    """Batch analyzer for 10-K company data."""
    
    def __init__(self, base_log_dir: str, target_ids: Optional[Set[str]] = None, overwrite: bool = False):
        super().__init__(base_log_dir, target_ids, overwrite)

    def extract_identifiers(self, source_file: Path) -> List[Dict[str, Any]]:
        """Extract company identifiers (CIKs) from pre-defined ID list file."""
        companies = []
        try:
            print(f"Reading company CIKs from: {source_file}")
            with open(source_file, 'r', encoding='utf-8') as f:
                ciks = json.load(f)
            
            if not isinstance(ciks, list):
                print(f"   Error: Expected a list of company CIKs")
                return []
            
            for cik in ciks:
                companies.append({
                    "cik": str(cik),
                    "identifier": str(cik)
                })
            
            print(f"Found {len(companies)} companies")
            
        except Exception as e:
            print(f"Error reading ID file: {e}")
        
        return companies
    
    def _prepare_analysis_command(self, identifier_info: Dict[str, Any], source_file: Path,
                                  subdir_name: str, **kwargs) -> tuple:
        """Prepare the command for company analysis."""
        cik = identifier_info["cik"]
        
        company_log_dir = self.base_log_dir / subdir_name
        company_log_dir.mkdir(parents=True, exist_ok=True)
        
        task = f"Analyze company with CIK {cik}"
        
        cmd = [
            sys.executable,
            "agent/data_agent.py",
            "--task", task,
            "--log-dir", str(company_log_dir)
        ]
        
        if not kwargs.get("auto_finish", True):
            cmd.append("--no-auto-finish")
        
        if kwargs.get("config_path"):
            cmd.extend(["--config", kwargs.get("config_path")])
        if kwargs.get("scenario"):
            cmd.extend(["--scenario", kwargs.get("scenario")])

        # Setup MCP arguments
        cmd.extend(["--sql-server", "tool_server/sqlite_mcp.py"])
        
        # Pass max_turns if provided
        if kwargs.get("max_turns"):
            cmd.extend(["--max-turns", str(kwargs.get("max_turns"))])
        
        env = os.environ.copy()
        env['CUSTOM_LOG_DIR'] = str(company_log_dir)
        
        return cmd, env, f"Company CIK {cik}"
    
    def get_subdir_name(self, identifier: str) -> str:
        return f"company_{identifier}"
    
    def _create_identifier_from_logs(self, identifier: str, dirname: str) -> Optional[Dict[str, Any]]:
        return {
            "cik": identifier,
            "identifier": identifier
        }


class UserBatchAnalyzer(BaseBatchAnalyzer):
    """Batch analyzer for GLOBEM user data."""
    
    def __init__(self, base_log_dir: str, target_ids: Optional[Set[str]] = None, overwrite: bool = False):
        super().__init__(base_log_dir, target_ids, overwrite)

    def extract_identifiers(self, source_file: Path) -> List[Dict[str, Any]]:
        """Extract user identifiers from pre-defined ID list file."""
        users = []
        try:
            print(f"Reading user IDs from: {source_file}")
            with open(source_file, 'r', encoding='utf-8') as f:
                user_ids = json.load(f)
            
            if not isinstance(user_ids, list):
                print(f"   Error: Expected a list of user IDs")
                return []
            
            for pid in user_ids:
                users.append({
                    "pid": str(pid),
                    "identifier": str(pid)
                })
            
            print(f"Found {len(users)} users")
            
        except Exception as e:
            print(f"Error reading ID file: {e}")
        
        return users
    
    def _prepare_analysis_command(self, identifier_info: Dict[str, Any], source_file: Path,
                                  subdir_name: str, **kwargs) -> tuple:
        """Prepare the command for user analysis."""
        pid = identifier_info["pid"]
        
        user_log_dir = self.base_log_dir / subdir_name
        user_log_dir.mkdir(parents=True, exist_ok=True)
        
        task = f"Analyze user {pid}"
        
        cmd = [
            sys.executable,
            "agent/data_agent.py",
            "--task", task,
            "--log-dir", str(user_log_dir)
        ]
        
        if not kwargs.get("auto_finish", True):
            cmd.append("--no-auto-finish")
        
        if kwargs.get("config_path"):
            cmd.extend(["--config", kwargs.get("config_path")])
        if kwargs.get("scenario"):
            cmd.extend(["--scenario", kwargs.get("scenario")])

        # Setup MCP arguments
        cmd.extend(["--code-server", "tool_server/code_mcp.py"])
        
        # Pass max_turns if provided
        if kwargs.get("max_turns"):
            cmd.extend(["--max-turns", str(kwargs.get("max_turns"))])
        
        env = os.environ.copy()
        env['CUSTOM_LOG_DIR'] = str(user_log_dir)
        
        return cmd, env, f"User {pid}"
    
    def get_subdir_name(self, identifier: str) -> str:
        return f"user_{identifier}"
    
    def _create_identifier_from_logs(self, identifier: str, dirname: str) -> Optional[Dict[str, Any]]:
        return {
            "pid": identifier,
            "identifier": identifier
        }


def main():
    """Main entry point for running the agent."""
    parser = argparse.ArgumentParser(
        description="DDR_Bench Unified Agent Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run MIMIC patient analysis (configured via config.yaml)
  python run_agent.py --scenario mimic

  # Run 10-K company analysis
  python run_agent.py --scenario 10k

  # Run GLOBEM user analysis
  python run_agent.py --scenario globem
        """
    )
    
    # Required arguments
    parser.add_argument("--scenario", required=True, choices=["mimic", "10k", "globem"],
                        help="Analysis scenario to run")
    
    # Configuration file
    parser.add_argument("--config", help="Path to config.yaml file")
    
    # Execution options
    parser.add_argument("--target-ids", help="Comma-separated list of specific IDs to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    parser.add_argument("--retry-only", action="store_true", help="Only retry failed analyses")
    
    args = parser.parse_args()
    
    # Resolve config path
    config_path = args.config
    if not config_path and Path("config.yaml").exists():
        config_path = str(Path("config.yaml").resolve())
    
    # Load configuration
    config = get_config(args.config) # get_config handles default loading too
    scenario_config = config.get_scenario(args.scenario)
    
    # Get settings from config
    log_dir = scenario_config.log_dir
    id_file = scenario_config.id_file
    
    auto_finish = config.agent.auto_finish if hasattr(config.agent, 'auto_finish') else True
    max_retries = config.agent.max_retries or 2
    max_turns = config.agent.max_turns or 100
    log_level = config.agent.log_level or "INFO"
    
    # Set log level for subprocesses and current process
    os.environ["DDR_LOG_LEVEL"] = log_level
    
    # Process target IDs
    target_ids = None
    if args.target_ids:
        target_ids = set(id.strip() for id in args.target_ids.split(',') if id.strip())
        print(f"Target IDs: {sorted(target_ids)}")
    
    # Validate id_file exists
    if not id_file or not Path(id_file).exists():
        parser.error(f"ID file not found: {id_file}. Please check config.yaml.")
    
    # Validate scenario paths are configured (just valid check, not passed via args)
    if args.scenario == "mimic" and not scenario_config.db_path:
        parser.error("db_path for mimic not found in config.yaml")
    if args.scenario == "10k" and not scenario_config.db_path:
        parser.error("db_path for 10k not found in config.yaml")
    if args.scenario == "globem" and not scenario_config.data_path:
        parser.error("data_path for globem not found in config.yaml")
        
    # Create analyzer based on scenario
    if args.scenario == "mimic":
        analyzer = PatientBatchAnalyzer(log_dir, target_ids, args.overwrite)
    elif args.scenario == "10k":
        analyzer = CompanyBatchAnalyzer(log_dir, target_ids, args.overwrite)
    elif args.scenario == "globem":
        analyzer = UserBatchAnalyzer(log_dir, target_ids, args.overwrite)
    
    source_file = Path(id_file)
    run_kwargs = {
        "max_turns": max_turns, 
        "auto_finish": auto_finish
    }
    
    # Run analysis
    print(f"\n{'='*60}")
    print(f"DDR_Bench Agent Runner")
    print(f"Scenario: {args.scenario}")
    print(f"Provider: {config.provider.default_provider}")
    print(f"Model: {config.provider.default_model}")
    print(f"Log Directory: {log_dir}")
    print(f"Config File: {config_path}")
    print(f"{'='*60}\n")
    
    # Add config_path and scenario to run_kwargs
    if config_path:
        run_kwargs["config_path"] = config_path
    run_kwargs["scenario"] = args.scenario
    
    if args.retry_only:
        analyzer.retry_failed_analyses(max_retries=max_retries, **run_kwargs)
    else:
        analyzer.run_batch_analysis(source_file, max_retries=max_retries, **run_kwargs)


if __name__ == "__main__":
    main()
