from __future__ import annotations

import os
import time
from pathlib import Path

from dso.utils.geometry import generate_geom
from dso.utils.hpc import HPCConfig, submit_slurm_trial, check_cluster_health
from dso.utils.progress import load_progress_only, save_progress, set_row, get_progress_paths

RETRYABLE_FAILURE_STATES = {"SSH", "SCP", "SBATCH", "SUBMIT_UNKNOWN"}

CORE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CORE_DIR.parent.parent.parent
CAD_SYS_LOCK = PROJECT_ROOT / "storage" / "sw_cad_engine.lock"


def acquire_cad_lock(lock_path: Path, timeout_s: int = 600) -> bool:
    """Blocks until it successfully creates a lock file, ensuring single-user CAD access."""
    start_time = time.time()
    while True:
        try:
            # 'x' mode is atomic at the OS level: fails if file already exists
            with open(lock_path, "x") as f:
                f.write(f"Locked by PID {os.getpid()} at {time.time()}\n")
            return True
        except FileExistsError:
            if time.time() - start_time > timeout_s:
                raise TimeoutError("Timed out waiting for SolidWorks CAD Engine process lock to release.")
            print("[CAD LOCK] SolidWorks busy processing another campaign. Retrying in 5 seconds...")
            time.sleep(5)


def release_cad_lock(lock_path: Path) -> None:
    """Safely cleans up the lock file."""
    try:
        if lock_path.exists():
            lock_path.unlink()
    except Exception as e:
        print(f"[WARN] Failed to delete CAD lockfile: {e}")


def count_active_by_cluster(progress: dict[int, dict], cluster_name: str) -> dict[str, int]:
    counts = {cluster_name: 0}
    for row in progress.values():
        status = row.get("status", "")
        row_cluster = row.get("cluster_name", "").strip()
        if status in {"SUBMITTED", "RUNNING"} and row_cluster == cluster_name:
            counts[cluster_name] += 1
    return counts


def choose_cluster(progress: dict[int, dict], cluster_name: str, max_active: int) -> str | None:
    counts = count_active_by_cluster(progress, cluster_name)
    active_count = counts.get(cluster_name, 0)
    if active_count < max_active:
        return cluster_name
    return None


def eligible_for_submit(row: dict, max_retries: int) -> bool:
    status = row.get("status", "")
    retry_count = int(row.get("retry_count", "0") or 0)
    state = (row.get("state") or "").strip().upper()

    if status in {"COMPLETED", "SUBMITTED", "RUNNING"}:
        return False
    if status == "FAILED":
        if retry_count > max_retries:
            return False
        return state in RETRYABLE_FAILURE_STATES or state == ""
    return status in {"PENDING", "CAD_DONE"}


def classify_pre_submit_failure(exc: Exception, stage: str) -> tuple[str, str]:
    text = f"{type(exc).__name__}: {exc}".lower()
    if stage == "CAD":
        if "rebuild" in text:
            return "CAD", "CAD_REBUILD"
        if "export" in text or ".x_t" in text or "parasolid" in text:
            return "CAD", "CAD_EXPORT"
        return "CAD", "CAD_UNKNOWN"
    if stage == "SUBMIT":
        if "sbatch" in text:
            return "SUBMIT", "SBATCH"
        if "scp" in text:
            return "SUBMIT", "SCP"
        if "ssh" in text:
            return "SUBMIT", "SSH"
        return "SUBMIT", "SUBMIT_UNKNOWN"
    return stage, "UNKNOWN"


def prepare_geometry(
    username: str, 
    campaign_id: str, 
    master_part_name: str, 
    progress: dict[int, dict], 
    trial_id: int, 
    param_names: list[str], 
    metric_names: list[str],
    csv_paths: tuple[Path, Path],
    reason: str
) -> bool:
    row = progress[trial_id]
    retry_count = int(row.get("retry_count", "0") or 0)
    
    acquire_cad_lock(CAD_SYS_LOCK)
    
    try:
        params = {name: float(row[name]) for name in param_names}
        
        cad_result = generate_geom(
            username=username,
            campaign_id=campaign_id,
            master_part_name=master_part_name,
            iteration=trial_id,
            params=params,
            keep_trial_part=True
        )
        
        exported_path = Path(cad_result["geom_xt_path"]).resolve()
        set_row(
            progress,
            trial_id,
            status="CAD_DONE",
            geom_source_path=str(exported_path),
            geom_xt_path=str(exported_path),
            state="CAD_DONE",
            reason=reason,
        )
        save_progress(csv_paths[0], csv_paths[1], progress, param_names, metric_names)
        return True
        
    except KeyboardInterrupt:
        save_progress(csv_paths[0], csv_paths[1], progress, param_names, metric_names)
        raise
    except Exception as exc:
        retry_count += 1
        failure_stage, failure_type = classify_pre_submit_failure(exc, "CAD")
        set_row(
            progress,
            trial_id,
            status="FAILED",
            retry_count=retry_count,
            success=False,
            state=failure_type,
            reason=f"{failure_stage}: {type(exc).__name__}: {exc}",
        )
        save_progress(csv_paths[0], csv_paths[1], progress, param_names, metric_names)
        print(f"[ERROR] CAD failed for trial {trial_id:04d}: {exc}")
        return False
        
    finally:
        release_cad_lock(CAD_SYS_LOCK)


def run_submission_pass(
    username: str,
    campaign_id: str,
    campaign_config: dict,
    cluster_cfg: HPCConfig,
    cluster_name: str,
    max_active: int,
    max_retries: int,
) -> None:
    param_names = [p["name"] for p in campaign_config["optimization_bounds"].get("parameters", [])]
    metric_names = list(campaign_config["optimization_bounds"].get("objectives", {}).keys())
    master_part_name = campaign_config["optimization_bounds"].get("geometry_settings", {}).get("master_part_name", "SP_Geom.SLDART")

    progress_csv, progress_xlsx = get_progress_paths(username, campaign_id)
    csv_paths = (progress_csv, progress_xlsx)
    
    progress = load_progress_only(progress_csv, param_names, metric_names)
    if not progress:
        print(f"[INFO] No progress entries found for campaign {campaign_id}.")
        return

    cluster_counts = count_active_by_cluster(progress, cluster_name)
    total_active = cluster_counts.get(cluster_name, 0)
    total_free = max(0, max_active - total_active)

    if total_free <= 0:
        print(f"[INFO] Cluster {cluster_name} is currently full ({total_active}/{max_active} slots used).")
        return

    for trial_id in sorted(progress):
        cluster_counts = count_active_by_cluster(progress, cluster_name)
        total_active = cluster_counts.get(cluster_name, 0)
        total_free = max(0, max_active - total_active)

        if total_free <= 0:
            break

        row = progress[trial_id]
        if not eligible_for_submit(row, max_retries):
            continue

        status = row["status"]

        if status == "PENDING":
            if not prepare_geometry(username, campaign_id, master_part_name, progress, trial_id, param_names, metric_names, csv_paths, "CAD export complete"):
                continue
            row = progress[trial_id]
            status = row["status"]

        elif status == "FAILED":
            geom_xt_path = row.get("geom_xt_path", "")
            retry_count = int(row.get("retry_count", "0") or 0) + 1

            if geom_xt_path and Path(geom_xt_path).exists():
                set_row(
                    progress,
                    trial_id,
                    status="CAD_DONE",
                    retry_count=retry_count,
                    state="RETRY_READY",
                    reason="Retrying from existing CAD export",
                )
                save_progress(progress_csv, progress_xlsx, progress, param_names, metric_names)
                row = progress[trial_id]
                status = row["status"]
            else:
                set_row(progress, trial_id, retry_count=retry_count)
                save_progress(progress_csv, progress_xlsx, progress, param_names, metric_names)
                if not prepare_geometry(username, campaign_id, master_part_name, progress, trial_id, param_names, metric_names, csv_paths, "CAD regenerated after failure"):
                    continue
                row = progress[trial_id]
                status = row["status"]

        if status != "CAD_DONE":
            continue

        chosen = choose_cluster(progress, cluster_name, max_active)
        if chosen is None:
            break

        try:
            geom_xt = Path(row["geom_xt_path"])

            if not check_cluster_health(cluster_cfg):
                print(f"[WARN] Cluster {cluster_name} failed active health check. Postponing submission.")
                break

            remote_dir, job_id, jobname = submit_slurm_trial(
                cfg=cluster_cfg,
                trial_id=trial_id,
                local_payload=geom_xt,
                remote_payload_name="geom.x_t"
            )

            set_row(
                progress,
                trial_id,
                status="SUBMITTED",
                remote_dir=remote_dir,
                job_id=job_id,
                jobname=jobname,
                state="SUBMITTED",
                reason=f"Submitted to {cluster_name}",
                cluster_name=cluster_name,
                cluster_user=cluster_cfg.user,
                cluster_host=cluster_cfg.host,
                cluster_remote_base=cluster_cfg.remote_base,
                cluster_base_dir=cluster_cfg.base_dir,
            )
            save_progress(progress_csv, progress_xlsx, progress, param_names, metric_names)
            print(f"[OK] Successfully submitted Trial {trial_id:04d} to {cluster_name} (Job ID: {job_id})")

        except KeyboardInterrupt:
            save_progress(progress_csv, progress_xlsx, progress, param_names, metric_names)
            raise
        except Exception as exc:
            retry_count = int(progress[trial_id].get("retry_count", "0") or 0) + 1
            failure_stage, failure_type = classify_pre_submit_failure(exc, "SUBMIT")
            set_row(
                progress,
                trial_id,
                status="FAILED",
                retry_count=retry_count,
                success=False,
                state=failure_type,
                reason=f"{failure_stage}: {type(exc).__name__}: {exc}",
            )
            save_progress(progress_csv, progress_xlsx, progress, param_names, metric_names)
            print(f"[ERROR] Remote submit failed for Trial {trial_id:04d}: {exc}")
            
            if failure_type in {"SSH", "SCP", "SBATCH", "SUBMIT_UNKNOWN"}:
                break