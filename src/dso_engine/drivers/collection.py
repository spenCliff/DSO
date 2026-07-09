from __future__ import annotations

from pathlib import Path

from dso.utils.hpc import HPCConfig, _remote_exists, _ssh, _try_sacct_state, try_download
from dso.utils.progress import load_progress_only, save_progress, set_row, get_progress_paths, get_runs_dir
from dso.utils.csv_parser import parse_mean_coefficients_csv  # Adjusted your import wrapper safely


def slurm_out_remote_path(trial_id: int, row: dict, cluster_cfg: HPCConfig) -> tuple[str, str, str]:
    job_id = (row.get("job_id") or "").strip()
    if not job_id:
        raise RuntimeError(f"Trial {trial_id} has no job_id recorded.")
    jobname = row.get("jobname") or f"IP_T{trial_id:04d}"
    
    remote_dir = row.get("remote_dir") or f"{cluster_cfg.remote_base}/trial_{trial_id:04d}"
    slurm_out_remote = f"{remote_dir}/{jobname}_{job_id}.out"
    return jobname, remote_dir, slurm_out_remote


def get_job_state_once(trial_id: int, row: dict, cluster_cfg: HPCConfig) -> tuple[str, str]:
    job_id = row.get("job_id")
    if not job_id:
        raise RuntimeError(f"Trial {trial_id} has no job_id recorded.")

    _, _, slurm_out_remote = slurm_out_remote_path(trial_id, row, cluster_cfg)

    state = _try_sacct_state(cluster_cfg, job_id)
    if state:
        base = state.split("+")[0]
        if base in {"PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}:
            return base, slurm_out_remote

    if _remote_exists(cluster_cfg, slurm_out_remote):
        tail = _ssh(cluster_cfg, f"bash -lc 'tail -n 120 {slurm_out_remote} || true'").lower()

        if "finished at:" in tail:
            return "COMPLETED", slurm_out_remote
        if "cancelled" in tail or "canceled" in tail:
            return "CANCELLED", slurm_out_remote
        if "out of memory" in tail:
            return "OUT_OF_MEMORY", slurm_out_remote
        if "license checkout failed" in tail:
            return "FAILED", slurm_out_remote
        if "error" in tail or "segmentation fault" in tail or "slurmstepd:" in tail:
            return "FAILED", slurm_out_remote

        return "RUNNING", slurm_out_remote

    return "PENDING", slurm_out_remote


def classify_runtime_failure(
    sacct_state: str,
    slurm_text: str,
    solver_text: str,
    results_ok: bool,
    solver_ok: bool,
) -> tuple[str, str]:
    text = (slurm_text + "\n" + solver_text).lower()

    if sacct_state == "OUT_OF_MEMORY" or "out of memory" in text:
        return "OUT_OF_MEMORY", "Job ran out of memory."
    if sacct_state == "TIMEOUT":
        return "TIMEOUT", "Job hit wall-clock limit."
    if sacct_state == "CANCELLED":
        return "CANCELLED", "Job was cancelled."
    if "license checkout failed" in text or "unable to execute mesh operation. license checkout failed" in text:
        return "LICENSE", "STAR-CCM+ license checkout failed during the run."
    if "segmentation fault" in text:
        return "SEGFAULT", "Segmentation fault reported in SLURM or solver output."
    if "floating point exception" in text:
        return "FLOATING_POINT", "Floating point exception reported in output."
    if sacct_state == "FAILED" and not solver_ok and not results_ok:
        return "FAILED_NO_OUTPUT", "Job failed and produced no solver or results output."
    if sacct_state == "COMPLETED" and not results_ok:
        return "POSTPRO_MISSING", "Job completed but Results.csv was missing."

    return sacct_state or "FAILED", (
        f"Job ended with state={sacct_state}. "
        f"Results available={results_ok}, SolverOutput available={solver_ok}."
    )


def collect_one_finished_trial(
    username: str, 
    campaign_id: str, 
    trial_id: int, 
    row: dict, 
    cluster_cfg: HPCConfig, 
    metric_names: list[str], 
    verbose: bool = True
) -> dict:
    sacct_state = row.get("state", "") or "UNKNOWN"
    _, remote_dir, slurm_out_remote = slurm_out_remote_path(trial_id, row, cluster_cfg)

    local_runs_dir = get_runs_dir(username, campaign_id)
    local_trial_dir = local_runs_dir / f"trial_{trial_id:04d}"
    local_trial_dir.mkdir(parents=True, exist_ok=True)

    slurm_local = local_trial_dir / Path(slurm_out_remote).name
    solver_local = local_trial_dir / "SolverOutput.txt"
    results_local = local_trial_dir / "Results.csv"

    slurm_ok = try_download(cluster_cfg, slurm_out_remote, slurm_local, verbose=verbose)
    solver_ok = try_download(cluster_cfg, f"{remote_dir}/SolverOutput.txt", solver_local, verbose=verbose)

    results_remote_a = f"{remote_dir}/PostProOutputs/Results.csv"
    results_remote_b = f"{remote_dir}/Results.csv"

    try:
        results_remote_exists = _remote_exists(cluster_cfg, results_remote_a) or _remote_exists(cluster_cfg, results_remote_b)
    except Exception as exc:
        results_remote_exists = False
        if verbose:
            print(f"[WARN] Could not check remote Results.csv existence for trial {trial_id:04d}: {exc}")

    results_ok = try_download(cluster_cfg, results_remote_a, results_local, verbose=verbose)
    if not results_ok:
        results_ok = try_download(cluster_cfg, results_remote_b, results_local, verbose=verbose)

    slurm_text = slurm_local.read_text(errors="ignore") if slurm_ok and slurm_local.exists() else ""
    solver_text = solver_local.read_text(errors="ignore") if solver_ok and solver_local.exists() else ""

    # Initialize objective result map dynamically using campaign configuration array values
    result = {"success": False, "state": sacct_state, "reason": "", "job_id": row.get("job_id", "")}
    for metric in metric_names:
        result[metric] = 0.0

    if results_ok and results_local.exists():
        try:
            vals = parse_mean_coefficients_csv(results_local)
            # Map values dynamically out of parsed csv via fallback keys
            for metric in metric_names:
                result[metric] = vals.get(metric, vals.get(f"{metric} Total", 0.0))
            result["success"] = True
            result["state"] = "COMPLETED"
            result["reason"] = "Parsed Results.csv"
            return result
        except Exception as exc:
            result["state"] = "RESULTS_PARSE"
            result["reason"] = f"Results.csv downloaded but failed to parse: {exc}"
            return result
    
    if sacct_state == "COMPLETED" and results_remote_exists and not results_ok:
        result["state"] = "LOCAL_DOWNLOAD_FAIL"
        result["reason"] = "Results.csv exists remotely but failed to download locally."
        return result

    fail_state, fail_reason = classify_runtime_failure(
        sacct_state=sacct_state,
        slurm_text=slurm_text,
        solver_text=solver_text,
        results_ok=results_ok,
        solver_ok=solver_ok,
    )
    result["state"] = fail_state
    result["reason"] = fail_reason
    return result


def try_recover_local_results(username: str, campaign_id: str, trial_id: int, metric_names: list[str]) -> dict | None:
    local_runs_dir = get_runs_dir(username, campaign_id)
    local_results = local_runs_dir / f"trial_{trial_id:04d}" / "ExcelFiles/Case_CompletedMaster Plot Averaged.csv"
    if not local_results.exists():
        return None

    vals = parse_mean_coefficients_csv(local_results)
    recovered = {
        "success": True,
        "state": "RECOVERED_LOCAL_RESULTS",
        "reason": "Recovered from existing local Results.csv",
    }
    for metric in metric_names:
        recovered[metric] = vals.get(metric, vals.get(f"{metric} Total", 0.0))
    return recovered


def run_collection_pass(
    username: str,
    campaign_id: str,
    campaign_config: dict,
    cluster_cfg: HPCConfig,
) -> None:
    param_names = [p["name"] for p in campaign_config["optimization_bounds"].get("parameters", [])]
    metric_names = list(campaign_config["optimization_bounds"].get("objectives", {}).keys())

    progress_csv, progress_xlsx = get_progress_paths(username, campaign_id)
    progress = load_progress_only(progress_csv, param_names, metric_names)
    
    if not progress:
        print("[INFO] No progress file rows found. Nothing to collect.")
        return

    polled = queued_now = running_now = completed_now = failed_now = recovered_now = skipped_now = 0

    for trial_id in sorted(progress):
        row = progress[trial_id]
        status = row.get("status", "")

        try:
            recovered = try_recover_local_results(username, campaign_id, trial_id, metric_names)
            if recovered is not None and status != "COMPLETED":
                update_fields = {
                    "status": "COMPLETED",
                    "success": recovered["success"],
                    "state": recovered["state"],
                    "reason": recovered["reason"]
                }
                update_fields.update({m: recovered[m] for m in metric_names})
                set_row(progress, trial_id, **update_fields)
                save_progress(progress_csv, progress_xlsx, progress, param_names, metric_names)
                recovered_now += 1
                continue

            if status in {"COMPLETED", "PENDING", "CAD_DONE"}:
                skipped_now += 1
                continue

            if status == "FAILED" and not row.get("job_id", "").strip():
                skipped_now += 1
                continue

            if status not in {"SUBMITTED", "RUNNING", "FAILED"}:
                skipped_now += 1
                continue

            if status in {"SUBMITTED", "RUNNING"}:
                polled += 1
                try:
                    state, _ = get_job_state_once(trial_id, row, cluster_cfg)
                except Exception as exc:
                    set_row(
                        progress,
                        trial_id,
                        state="POLL_FAILED",
                        reason=f"Polling failed: {type(exc).__name__}: {exc}",
                    )
                    save_progress(progress_csv, progress_xlsx, progress, param_names, metric_names)
                    continue

                if state == "PENDING":
                    set_row(progress, trial_id, status="SUBMITTED", state="PENDING", reason="Job still queued.")
                    save_progress(progress_csv, progress_xlsx, progress, param_names, metric_names)
                    queued_now += 1
                    continue

                if state == "RUNNING":
                    set_row(progress, trial_id, status="RUNNING", state="RUNNING", reason="Job running.")
                    save_progress(progress_csv, progress_xlsx, progress, param_names, metric_names)
                    running_now += 1
                    continue

                set_row(progress, trial_id, state=state, reason=f"Terminal SLURM state: {state}")
                save_progress(progress_csv, progress_xlsx, progress, param_names, metric_names)

            result = collect_one_finished_trial(username, campaign_id, trial_id, progress[trial_id], cluster_cfg, metric_names, verbose=True)
            new_state = result.get("state", "")
            new_success = result.get("success", False)

            if new_success:
                new_status = "COMPLETED"
            elif new_state == "LOCAL_DOWNLOAD_FAIL":
                new_status = "RUNNING"
            else:
                new_status = "FAILED"

            update_fields = {
                "success": new_success,
                "state": new_state,
                "reason": result.get("reason", ""),
                "status": new_status,
            }
            update_fields.update({m: result.get(m, 0.0) for m in metric_names})
            set_row(progress, trial_id, **update_fields)
            save_progress(progress_csv, progress_xlsx, progress, param_names, metric_names)

            if result.get("success", False):
                completed_now += 1
            elif new_state == "LOCAL_DOWNLOAD_FAIL":
                running_now += 1
            else:
                failed_now += 1

        except KeyboardInterrupt:
            save_progress(progress_csv, progress_xlsx, progress, param_names, metric_names)
            raise
        except Exception as exc:
            set_row(progress, trial_id, success=False, state="COLLECT_EXCEPTION", reason=f"{type(exc).__name__}: {exc}")
            save_progress(progress_csv, progress_xlsx, progress, param_names, metric_names)