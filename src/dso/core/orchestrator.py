from __future__ import annotations

import sys
import time
import random
from datetime import datetime
from pathlib import Path
import yaml

from dso.utils.hpc import HPCConfig
from dso.utils.progress import get_progress_paths

from dso.core.optimisation import run_optimisation_pass
from dso.core.collection import run_collection_pass
from dso.core.submission import run_submission_pass

CORE_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = CORE_DIR.parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent

POLL_INTERVAL_S = 300


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str, log_file_path: Path) -> None:
    line = f"[{now_str()}] {msg}"
    print(line, flush=True)
    try:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        with log_file_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def sleep_with_countdown(base_seconds: int) -> None:
    jitter = random.randint(-15, 15)
    total_sleep = max(10, base_seconds + jitter)
    for i in range(total_sleep, 0, -1):
        sys.stdout.write(f"\r[INFO] Next cycle loop in {i:3d}s... Press Ctrl+C to stop.")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\r" + " " * 60 + "\r")
    sys.stdout.flush()


def start_pipeline(username: str, campaign_id: str, campaign_config: dict, campaign_iterations: int) -> None:
    campaign_dir = PROJECT_ROOT / "storage" / username / "campaigns" / campaign_id
    log_file = campaign_dir / "runs" / "orchestrator.log"

    log("=" * 80, log_file)
    log(f"[START] In-Process Orchestration Campaign Initiated: {campaign_id}", log_file)
    log(f"[INFO] Target Evaluated Budget Limits: {campaign_iterations}", log_file)
    log("=" * 80, log_file)

    try:
        with open(PROJECT_ROOT / "config" / "hpc_config.yaml", "r") as f:
            servers_blueprint = yaml.safe_load(f)["servers"]

        user_config_path = PROJECT_ROOT / "storage" / username / "user_config.yaml"
        with open(user_config_path, "r") as f:
            user_profile = yaml.safe_load(f)

        profile_name = campaign_config["optimisation_bounds"]["solver_settings"].get("cluster_target", "iridis6")
        profile_meta = user_profile["cluster_profiles"][profile_name]
        
        server_target = profile_meta.get("server", profile_name)
        server_meta = servers_blueprint[server_target]
        solver_settings = campaign_config["optimisation_bounds"]["solver_settings"]

        cluster_cfg = HPCConfig(
            user=profile_meta["cluster_user"],
            host=server_meta["hostname"],
            remote_base=profile_meta["remote_base"],
            base_dir=profile_meta["base_dir"],
            ssh_key=user_profile.get("ssh_key_path")
        )
        
        max_active_slots = profile_meta["max_parallel_slots"]
        max_retries = solver_settings.get("max_retries", 2)

    except Exception as e:
        log(f"[CRITICAL] Error parsing runtime infrastructure configurations: {e}", log_file)
        sys.exit(1)

    cycle = 0
    try:
        while True:
            cycle += 1
            log("-" * 80, log_file)
            log(f"[CYCLE {cycle}] Starting Sequential Pipeline Loop Passes", log_file)
            log("-" * 80, log_file)

            try:
                log("[PASS] Running Optimisation Agent...", log_file)
                run_optimisation_pass(
                    username=username, 
                    campaign_id=campaign_id, 
                    campaign_config=campaign_config
                )
            except Exception as e:
                log(f"[ERROR] Optimisation Stage Failed: {e}", log_file)

            try:
                log("[PASS] Running Remote Metric Data Collection Pass...", log_file)
                run_collection_pass(
                    username=username,
                    campaign_id=campaign_id,
                    campaign_config=campaign_config,
                    cluster_cfg=cluster_cfg
                )
            except Exception as e:
                log(f"[ERROR] Collection Stage Failed: {e}", log_file)

            try:
                log("[PASS] Running Cluster Workload Queue Dispatch Pass...", log_file)
                run_submission_pass(
                    username=username,
                    campaign_id=campaign_id,
                    campaign_config=campaign_config,
                    cluster_cfg=cluster_cfg,
                    cluster_name=server_target,
                    max_active=max_active_slots,
                    max_retries=max_retries
                )
            except Exception as e:
                log(f"[ERROR] Submission Stage Failed: {e}", log_file)

            progress_csv, progress_xlsx = get_progress_paths(username, campaign_id)
            if progress_csv.exists():
                with open(progress_csv, "r", encoding="utf-8") as f:
                    completed_evaluations = sum(1 for line in f if "COMPLETED" in line)
                
                log(f"[INFO] Progress Check: {completed_evaluations}/{campaign_iterations} evaluations complete.", log_file)
                
                if completed_evaluations >= campaign_iterations:
                    log(f"[SUCCESS] Reached target budget limit of {campaign_iterations} evaluations! Exiting.", log_file)
                    break
            else:
                log("[INFO] Progress Tracker Empty: Waiting for initial candidate convergence lifecycle.", log_file)

            log(f"[INFO] Cycle {cycle} Complete.", log_file)
            sleep_with_countdown(POLL_INTERVAL_S)

    except KeyboardInterrupt:
        log("[STOP] Orchestrator Loop Interrupted Cleanly via User SIGINT.", log_file)