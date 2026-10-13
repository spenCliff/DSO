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
    start_time = time.time()
    while True:
        try:
            with open(lock_path, "x") as f:
                f.write(f"Locked by PID {os.getpid()} at {time.time()}\n")
            return True
        except FileExistsError:
            if time.time() - start_time > timeout_s:
                raise TimeoutError("Timed out waiting for SolidWorks CAD Engine process lock to release.")
            time.sleep(5)


def release_cad_lock(lock_path: Path) -> None:
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


def extract_campaign_parameters(campaign_config: dict) -> tuple[list[dict], dict[str, float]]:
    bounds_cfg = campaign_config.get("optimisation_bounds", {})
    part_params = bounds_cfg.get("part_parameters", {})
    
    flat_parameters = []
    baseline_defaults = {}

    for comp_key, comp_data in part_params.items():
        flat_parameters.extend(comp_data.get("parameters", []))
        baseline_defaults.update(comp_data.get("baselines", {}))
        
    return flat_parameters, baseline_defaults


def get_params_for_component(campaign_config: dict, component_key: str, trial_row: dict) -> dict[str, float]:
    part_params = campaign_config.get("optimisation_bounds", {}).get("part_parameters", {})
    geom_settings = campaign_config.get("optimisation_bounds", {}).get("geometry_settings", {})
    part_mappings = geom_settings.get("part_mappings", {})

    # Invert part_mappings to find the config key (e.g. "side_pod" -> "RPLC_SP")
    param_block_key = component_key
    for rplc_key, comp_name in part_mappings.items():
        if comp_name == component_key or rplc_key == component_key:
            param_block_key = rplc_key
            break

    comp_cfg = part_params.get(param_block_key, {})
    param_defs = comp_cfg.get("parameters", [])
    
    comp_params = {}
    for pdef in param_defs:
        pname = pdef["name"]
        if pname in trial_row and trial_row[pname] != "":
            comp_params[pname] = float(trial_row[pname])
            
    return comp_params


def prepare_geometry(
    username: str, 
    campaign_id: str, 
    campaign_config: dict, 
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
        geom_settings = campaign_config.get("optimisation_bounds", {}).get("geometry_settings", {})
        master_parts = geom_settings.get("master_parts", {})
        
        exported_xt_paths = []
        exported_part_paths = []
        
        for comp_key, master_part_name in master_parts.items():
            comp_params = get_params_for_component(campaign_config, comp_key, row)
            
            cad_result = generate_geom(
                username=username,
                campaign_id=campaign_id,
                master_part_name=master_part_name,
                iteration=trial_id,
                params=comp_params,
                keep_trial_part=True
            )
            
            orig_xt = Path(cad_result["geom_xt_path"])
            target_xt = orig_xt.parent / f"{comp_key}.x_t"
            if orig_xt.exists():
                orig_xt.replace(target_xt)
                
            exported_xt_paths.append(str(target_xt.resolve()))
            exported_part_paths.append(cad_result["trial_part_path"])
        
        set_row(
            progress,
            trial_id,
            status="CAD_DONE",
            geom_source_path=";".join(exported_part_paths),
            geom_xt_path=";".join(exported_xt_paths),
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
    flat_params, _ = extract_campaign_parameters(campaign_config)
    param_names = [p["name"] for p in flat_params]
    metric_names = list(campaign_config["optimisation_bounds"].get("objectives", {}).keys())

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
            if not prepare_geometry(
                username=username,
                campaign_id=campaign_id,
                campaign_config=campaign_config,
                progress=progress,
                trial_id=trial_id,
                param_names=param_names,
                metric_names=metric_names,
                csv_paths=csv_paths,
                reason="CAD export complete"
            ):
                continue
            row = progress[trial_id]
            status = row["status"]

        elif status == "FAILED":
            geom_xt_str = row.get("geom_xt_path", "")
            retry_count = int(row.get("retry_count", "0") or 0) + 1

            payload_paths = [Path(p) for p in geom_xt_str.split(";") if p]
            all_exist = len(payload_paths) > 0 and all(p.exists() for p in payload_paths)

            if all_exist:
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
                if not prepare_geometry(
                    username=username,
                    campaign_id=campaign_id,
                    campaign_config=campaign_config,
                    progress=progress,
                    trial_id=trial_id,
                    param_names=param_names,
                    metric_names=metric_names,
                    csv_paths=csv_paths,
                    reason="CAD regenerated after failure"
                ):
                    continue
                row = progress[trial_id]
                status = row["status"]

        if status != "CAD_DONE":
            continue

        chosen = choose_cluster(progress, cluster_name, max_active)
        if chosen is None:
            break

        try:
            # 1. Parse CAD file paths
            payload_paths = [Path(p) for p in row["geom_xt_path"].split(";") if p]

            # 2. Append active_config.yaml / active_config.json so Java macro can read part_mappings
            campaign_inputs_dir = PROJECT_ROOT / "storage" / "campaigns" / username / campaign_id / "inputs"
            yaml_cfg = campaign_inputs_dir / "config.yaml"

            if yaml_cfg.exists():
                payload_paths.append(yaml_cfg)

            if not check_cluster_health(cluster_cfg):
                print(f"[WARN] Cluster {cluster_name} failed active health check. Postponing submission.")
                break

            remote_dir, job_id, jobname = submit_slurm_trial(
                cfg=cluster_cfg,
                trial_id=trial_id,
                local_payloads=payload_paths,
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