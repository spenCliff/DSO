from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

from ax.exceptions.generation_strategy import MaxParallelismReachedException
from ax.service.ax_client import AxClient, ObjectiveProperties

from dso.utils.progress import load_progress_only, save_progress

DRIVERS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DRIVERS_DIR.parent.parent.parent

NON_AX_FAILURE_STATES = {"SSH", "SCP", "SBATCH", "SUBMIT_UNKNOWN", "POLL_FAILED"}
BAD_BACKEND_STATES = {"SSH", "SCP", "SBATCH", "SUBMIT_UNKNOWN", "POLL_FAILED"}


def safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load_tracking(tracking_csv: Path) -> Dict[int, dict]:
    if not tracking_csv.exists():
        return {}
    tracking: Dict[int, dict] = {}
    with tracking_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tracking[int(row["trial_id"])] = row
    return tracking


def save_tracking(tracking_csv: Path, tracking: Dict[int, dict]) -> None:
    rows = [tracking[k] for k in sorted(tracking)]
    fieldnames = ["trial_id", "ax_trial_index", "reported", "status_at_report"]
    with tracking_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def set_tracking_row(
    tracking: Dict[int, dict],
    trial_id: int,
    ax_trial_index: int | None = None,
    reported: str | None = None,
    status_at_report: str | None = None,
) -> None:
    row = tracking.setdefault(
        trial_id,
        {
            "trial_id": str(trial_id),
            "ax_trial_index": "",
            "reported": "0",
            "status_at_report": "",
        },
    )
    if ax_trial_index is not None:
        row["ax_trial_index"] = str(ax_trial_index)
    if reported is not None:
        row["reported"] = str(reported)
    if status_at_report is not None:
        row["status_at_report"] = str(status_at_report)


def load_or_create_ax_client(
    state_json: Path, 
    experiment_name: str, 
    ax_parameters: list, 
    objectives_config: dict, 
    random_seed: int
) -> AxClient:
    ax_objectives = {
        metric_name: ObjectiveProperties(minimize=props.get("minimize", True))
        for metric_name, props in objectives_config.items()
    }

    if state_json.exists():
        print(f"[AX] Loading existing experiment state: {state_json.name}")
        ax_client = AxClient.load_from_json_file(filepath=str(state_json))
        ax_client.set_optimization_config(objectives=ax_objectives)
        ax_client.save_to_json_file(filepath=str(state_json))
        return ax_client

    print(f"[AX] Constructing a brand new experiment loop: {experiment_name}")
    ax_client = AxClient(random_seed=random_seed)
    ax_client.create_experiment(
        name=experiment_name,
        parameters=ax_parameters,
        objectives=ax_objectives,
    )
    ax_client.save_to_json_file(filepath=str(state_json))
    return ax_client


def next_trial_id(progress: Dict[int, dict]) -> int:
    return 1 if not progress else max(progress) + 1


def ax_trial_is_active(progress_row: dict, tracking_row: dict | None) -> bool:
    if tracking_row is None:
        return False
    status = progress_row.get("status", "")
    state = (progress_row.get("state") or "").strip().upper()
    if tracking_row.get("reported", "0") == "1":
        return False
    if status == "FAILED" and state in NON_AX_FAILURE_STATES:
        return False
    return status in {"PENDING", "CAD_DONE", "SUBMITTED", "RUNNING", "FAILED", "COMPLETED"}


def count_active_ax_trials(progress: Dict[int, dict], tracking: Dict[int, dict]) -> int:
    return sum(1 for tid, r in progress.items() if ax_trial_is_active(r, tracking.get(tid)))


def should_report_failure_to_ax(progress_row: dict) -> bool:
    return progress_row.get("status", "") == "FAILED" and (progress_row.get("state") or "").strip().upper() not in NON_AX_FAILURE_STATES


def backend_unhealthy(progress: Dict[int, dict]) -> bool:
    return any(r.get("status", "") == "FAILED" and (r.get("state") or "").strip().upper() in BAD_BACKEND_STATES for r in progress.values())


def report_completed_trial(
    ax_client: AxClient,
    trial_id: int,
    ax_trial_index: int,
    progress_row: dict,
    objectives_config: dict,
) -> None:
    raw_data = {}
    log_strings = []

    for metric_name, props in objectives_config.items():
        penalty_value = props.get("penalty_value", -1.0e9)
        val = safe_float(progress_row.get(metric_name, ""), penalty_value)
        raw_data[metric_name] = (val, 0.0)
        log_strings.append(f"{metric_name}={val:.4f}")

    ax_client.complete_trial(trial_index=ax_trial_index, raw_data=raw_data)
    print(f"[AX] Reported COMPLETED trial_id={trial_id:04d}, ax_trial_index={ax_trial_index}, {', '.join(log_strings)}")


def report_failed_trial(ax_client: AxClient, trial_id: int, ax_trial_index: int, objectives_config: dict) -> None:
    for method_name in ("log_trial_failure", "mark_trial_failed"):
        method = getattr(ax_client, method_name, None)
        if method is not None:
            try:
                method(ax_trial_index)
                print(f"[AX] Marked FAILED trial_id={trial_id:04d}, ax_trial_index={ax_trial_index}")
                return
            except Exception:
                pass

    raw_data = {metric: (props.get("penalty_value", -1.0e9), 0.0) for metric, props in objectives_config.items()}
    ax_client.complete_trial(trial_index=ax_trial_index, raw_data=raw_data)
    print(f"[AX] Failed fallback tracking penalties enforced for trial_id={trial_id:04d}")


def run_optimisation_pass(username: str, campaign_id: str, campaign_config: dict) -> None:
    campaign_dir = PROJECT_ROOT / "storage" / "campaigns" / username / campaign_id
    database_dir = campaign_dir / "runs"
    runs_ax_dir = campaign_dir / "ax_state"

    database_dir.mkdir(parents=True, exist_ok=True)
    runs_ax_dir.mkdir(parents=True, exist_ok=True)

    progress_csv = database_dir / "progress_live.csv"
    progress_xlsx = database_dir / "progress_open.xlsx"
    ax_state_json = runs_ax_dir / "ax_state.json"
    ax_tracking_csv = runs_ax_dir / "ax_tracking.csv"

    experiment_name = campaign_config.get("campaign_settings", {}).get("name", f"DSO_{campaign_id}")
    bounds_cfg = campaign_config.get("optimization_bounds", {})
    solver_cfg = bounds_cfg.get("solver_settings", {})
    
    random_seed = solver_cfg.get("random_seed", 123)
    max_ax_active = solver_cfg.get("max_parallel_slots", 6)
    max_new_per_pass = solver_cfg.get("max_new_trials_per_pass", 6)

    ax_parameters = bounds_cfg.get("parameters", [])
    objectives_config = bounds_cfg.get("objectives", {})

    full_param_names = [p["name"] for p in ax_parameters]
    metric_names = list(objectives_config.keys())
    baseline_defaults = {p["name"]: p.get("baseline", 0.0) for p in ax_parameters}

    ax_client = load_or_create_ax_client(
        state_json=ax_state_json,
        experiment_name=experiment_name,
        ax_parameters=ax_parameters,
        objectives_config=objectives_config,
        random_seed=random_seed
    )

    progress = load_progress_only(progress_csv, full_param_names, metric_names)
    tracking = load_tracking(ax_tracking_csv)

    reported_now = 0
    for trial_id in sorted(progress):
        row = progress[trial_id]
        tracking_row = tracking.get(trial_id)

        if not tracking_row or tracking_row.get("reported", "0") == "1":
            continue

        ax_trial_index = int(tracking_row["ax_trial_index"])
        status = row.get("status", "")

        try:
            if status == "COMPLETED":
                report_completed_trial(ax_client, trial_id, ax_trial_index, row, objectives_config)
                set_tracking_row(tracking, trial_id, reported="1", status_at_report="COMPLETED")
                reported_now += 1
            elif should_report_failure_to_ax(row):
                report_failed_trial(ax_client, trial_id, ax_trial_index, objectives_config)
                set_tracking_row(tracking, trial_id, reported="1", status_at_report="FAILED")
                reported_now += 1
        except Exception as e:
            print(f"[WARN] Error parsing metrics for trial_id={trial_id:04d}: {e}")

    save_tracking(ax_tracking_csv, tracking)
    ax_client.save_to_json_file(filepath=str(ax_state_json))

    progress = load_progress_only(progress_csv, full_param_names, metric_names)

    if backend_unhealthy(progress):
        print("[INFO] Cluster transport nodes unhealthy. Pausing discovery generations.")
        return

    active_ax = count_active_ax_trials(progress, tracking)
    free_slots = max(0, max_ax_active - active_ax)
    to_create = min(free_slots, max_new_per_pass)

    if to_create <= 0:
        print(f"[INFO] Search pipeline capacity reached ({active_ax}/{max_ax_active} active).")
        return

    created_now = 0
    for _ in range(to_create):
        try:
            params, ax_trial_index = ax_client.get_next_trial()
        except MaxParallelismReachedException:
            break

        full_params = dict(baseline_defaults)
        for name in full_param_names:
            if name in params:
                full_params[name] = float(params[name])

        trial_id = next_trial_id(progress)

        progress[trial_id] = {
            "trial_id": str(trial_id),
            "case_name": f"ax_{ax_trial_index:04d}",
            "status": "PENDING",
            "state": "QUEUED_BY_AX",
            "reason": "Acquisition parameter choice enqueued",
            **{k: "" for k in objectives_config.keys()},
            **{k: str(v) for k, v in full_params.items()},
        }

        set_tracking_row(tracking, trial_id, ax_trial_index=ax_trial_index, reported="0")
        created_now += 1
        print(f"[AX] Discovered Candidate trial_id={trial_id:04d} -> Index {ax_trial_index}")

    save_progress(progress_csv, progress_xlsx, progress, full_param_names, metric_names)
    save_tracking(ax_tracking_csv, tracking)
    ax_client.save_to_json_file(filepath=str(ax_state_json))
    print(f"[INFO] Optimisation check finished. Reported: {reported_now} | Generated New: {created_now}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--campaign-id", required=True)
    args = parser.parse_args()

    runtime_config_path = PROJECT_ROOT / "storage" / "campaigns" / args.username / args.campaign_id / "inputs" / "active_config.json"
    with open(runtime_config_path, "r") as f:
        loaded_config = json.load(f)

    run_optimisation_pass(username=args.username, campaign_id=args.campaign_id, campaign_config=loaded_config)