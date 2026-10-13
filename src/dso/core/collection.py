from __future__ import annotations

import time
from pathlib import Path

from dso.utils.hpc import (
    HPCConfig,
    _remote_exists,
    _ssh,
    _try_sacct_state,
    download_remote_file,
    parse_generic_csv_metrics,
)
from dso.utils.progress import (
    get_progress_paths,
    get_runs_dir,
    load_progress_only,
    save_progress,
    set_row,
)


def extract_campaign_parameters(campaign_config: dict) -> tuple[list[dict], dict[str, float]]:
    bounds_cfg = campaign_config.get("optimisation_bounds", {})
    part_params = bounds_cfg.get("part_parameters", {})

    flat_parameters = []
    baseline_defaults = {}

    for comp_key, comp_data in part_params.items():
        flat_parameters.extend(comp_data.get("parameters", []))
        baseline_defaults.update(comp_data.get("baselines", {}))

    return flat_parameters, baseline_defaults


def slurm_out_remote_path(trial_id: int, row: dict, cluster_cfg: HPCConfig) -> tuple[str, str, str]:
    job_id = (row.get("job_id") or "").strip()
    if not job_id:
        raise RuntimeError(f"Trial {trial_id} has no job_id recorded.")

    jobname = row.get("jobname") or f"DSO_T{trial_id:04d}"
    remote_dir = row.get("remote_dir") or f"{cluster_cfg.remote_base}/trial_{trial_id:04d}"
    slurm_out_remote = f"{remote_dir}/slurm-{job_id}.out"

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

    file_exists = _remote_exists(cluster_cfg, slurm_out_remote)

    if file_exists:
        tail = _ssh(cluster_cfg, f"bash -lc 'tail -n 120 {slurm_out_remote} || true'").lower()

        if "finished at:" in tail or "completed at:" in tail or "running epilogue script" in tail:
            return "COMPLETED", slurm_out_remote

        if "cancelled" in tail or "canceled" in tail:
            return "CANCELLED", slurm_out_remote

        if "out of memory" in tail or "oom-killer" in tail:
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
        return "POSTPRO_MISSING", "Job completed but Results CSV was missing."

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
    campaign_config: dict,
    verbose: bool = True,
) -> dict:
    sacct_state = row.get("state", "") or "UNKNOWN"
    jobname, remote_dir, slurm_out_remote = slurm_out_remote_path(trial_id, row, cluster_cfg)

    local_runs_dir = get_runs_dir(username, campaign_id)
    local_trial_dir = local_runs_dir / f"trial_{trial_id:04d}"
    local_trial_dir.mkdir(parents=True, exist_ok=True)

    slurm_local = local_trial_dir / Path(slurm_out_remote).name
    solver_local = local_trial_dir / "SolverOutput.txt"

    slurm_ok = download_remote_file(cluster_cfg, slurm_out_remote, slurm_local)
    solver_ok = download_remote_file(cluster_cfg, f"{remote_dir}/SolverOutput.txt", solver_local)

    # Dynamic templates directly from parameters.yaml
    templates = campaign_config.get("optimisation_bounds", {}).get("results_file_templates", ["Results.csv"])

    results_ok = False
    results_remote_exists = False
    results_local = None

    for tmpl in templates:
        rel_path = tmpl.format(
            jobname=jobname,
            trial_id=f"{trial_id:04d}",
            trial_name=f"trial_{trial_id:04d}",
        ).lstrip("/")

        remote_csv = f"{remote_dir}/{rel_path}"

        try:
            if _remote_exists(cluster_cfg, remote_csv):
                results_remote_exists = True

                local_results_path = local_trial_dir / rel_path
                local_results_path.parent.mkdir(parents=True, exist_ok=True)

                if download_remote_file(cluster_cfg, remote_csv, local_results_path):
                    results_ok = True
                    results_local = local_results_path
                    if verbose:
                        print(f"[INFO] Downloaded results for Trial {trial_id:04d}: {local_results_path.name}")
                    break
        except Exception as exc:
            if verbose:
                print(f"[WARN] Error checking remote file {remote_csv} for trial {trial_id:04d}: {exc}")

    slurm_text = slurm_local.read_text(errors="ignore") if slurm_ok and slurm_local.exists() else ""
    solver_text = solver_local.read_text(errors="ignore") if solver_ok and solver_local.exists() else ""

    result = {"success": False, "state": sacct_state, "reason": "", "job_id": row.get("job_id", "")}
    for metric in metric_names:
        result[metric] = 0.0

    if results_ok and results_local and results_local.exists():
        try:
            csv_column_mapping = campaign_config.get("optimisation_bounds", {}).get("csv_column_mapping", {})
            vals = parse_generic_csv_metrics(results_local, metric_names, csv_column_mapping)

            for metric in metric_names:
                result[metric] = vals.get(metric, 0.0)

            result["success"] = True
            result["state"] = "COMPLETED"
            result["reason"] = "Parsed Results CSV successfully"

            try:
                if verbose:
                    print(f"[INFO] Cleaning remote simulation files in {remote_dir}...")
                _ssh(
                    cluster_cfg,
                    f"bash -lc 'rm -f {remote_dir}/*.sim {remote_dir}/*.sim~ {remote_dir}/*.x_t {remote_dir}/*.step'",
                )
            except Exception as clean_exc:
                print(f"[WARN] Failed to delete remote sim files for trial {trial_id:04d}: {clean_exc}")

            return result
        except Exception as exc:
            result["state"] = "RESULTS_PARSE"
            result["reason"] = f"Results CSV downloaded but failed to parse: {exc}"
            return result

    if sacct_state == "COMPLETED" and results_remote_exists and not results_ok:
        result["state"] = "LOCAL_DOWNLOAD_FAIL"
        result["reason"] = "Results CSV exists remotely but failed to download locally."
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


def try_recover_local_results(
    username: str, campaign_id: str, trial_id: int, row: dict, metric_names: list[str], campaign_config: dict
) -> dict | None:
    local_runs_dir = get_runs_dir(username, campaign_id)
    local_trial_dir = local_runs_dir / f"trial_{trial_id:04d}"

    jobname = row.get("jobname") or f"DSO_T{trial_id:04d}"
    templates = campaign_config.get("optimisation_bounds", {}).get("results_file_templates", ["Results.csv"])

    found_local = None
    for tmpl in templates:
        rel_path = tmpl.format(
            jobname=jobname,
            trial_id=f"{trial_id:04d}",
            trial_name=f"trial_{trial_id:04d}",
        ).lstrip("/")

        p = local_trial_dir / rel_path
        if p.exists():
            found_local = p
            break

    if not found_local:
        return None

    try:
        csv_column_mapping = campaign_config.get("optimisation_bounds", {}).get("csv_column_mapping", {})
        vals = parse_generic_csv_metrics(found_local, metric_names, csv_column_mapping)
        recovered = {
            "success": True,
            "state": "RECOVERED_LOCAL_RESULTS",
            "reason": f"Recovered metrics from local file: {found_local.name}",
        }
        for metric in metric_names:
            recovered[metric] = vals.get(metric, 0.0)
        return recovered
    except Exception:
        return None


def get_batch_slurm_states(job_ids: list[str], cluster_cfg: HPCConfig) -> dict[str, str]:
    if not job_ids:
        return {}

    valid_jobs = [str(j).strip() for j in job_ids if str(j).strip()]
    if not valid_jobs:
        return {}

    job_csv = ",".join(valid_jobs)
    inner_cmd = f"command -v sacct >/dev/null 2>&1 || exit 0; sacct -j {job_csv} --format=JobID,State --noheader"
    cmd = f"bash -lc '{inner_cmd}'"

    try:
        stdout = _ssh(cluster_cfg, cmd)
    except Exception as exc:
        print(f"[WARN] Batch sacct check failed: {type(exc).__name__}: {exc}")
        return {}

    states = {}
    for line in stdout.strip().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            raw_id, state = parts[0], parts[1]
            clean_id = raw_id.split(".")[0]
            if clean_id in valid_jobs:
                states[clean_id] = state.upper()

    return states


def run_collection_pass(
    username: str,
    campaign_id: str,
    campaign_config: dict,
    cluster_cfg: HPCConfig,
) -> None:
    flat_params, _ = extract_campaign_parameters(campaign_config)
    param_names = [p["name"] for p in flat_params]
    metric_names = list(campaign_config["optimisation_bounds"].get("objectives", {}).keys())

    progress_csv, progress_xlsx = get_progress_paths(username, campaign_id)
    progress = load_progress_only(progress_csv, param_names, metric_names)

    if not progress:
        print("[INFO] No progress file rows found. Nothing to collect.")
        return

    polled = queued_now = running_now = completed_now = failed_now = recovered_now = skipped_now = 0

    active_jobs_map = {}
    for t_id, row in progress.items():
        st = row.get("status", "")
        j_id = str(row.get("job_id", "")).strip()
        if st in {"SUBMITTED", "RUNNING"} and j_id:
            active_jobs_map[j_id] = t_id

    batch_states = get_batch_slurm_states(list(active_jobs_map.keys()), cluster_cfg)

    for trial_id in sorted(progress):
        row = progress[trial_id]
        status = row.get("status", "")

        try:
            metrics_missing = any(float(row.get(m, 0.0) or 0.0) == 0.0 for m in metric_names)

            # Dynamically checks local runs using parameters.yaml templates
            recovered = try_recover_local_results(
                username, campaign_id, trial_id, row, metric_names, campaign_config
            )
            if recovered is not None and (status != "COMPLETED" or metrics_missing):
                update_fields = {
                    "status": "COMPLETED",
                    "success": recovered["success"],
                    "state": recovered["state"],
                    "reason": recovered["reason"],
                }
                update_fields.update({m: recovered[m] for m in metric_names})
                set_row(progress, trial_id, **update_fields)
                save_progress(progress_csv, progress_xlsx, progress, param_names, metric_names)
                recovered_now += 1
                print(f"[INFO] Recovered metrics locally for Trial {trial_id:04d}: {update_fields}")
                continue

            if status == "COMPLETED" and not metrics_missing:
                skipped_now += 1
                continue

            if status in {"PENDING", "CAD_DONE"}:
                skipped_now += 1
                continue

            if status == "FAILED" and not row.get("job_id", "").strip():
                skipped_now += 1
                continue

            if status in {"SUBMITTED", "RUNNING", "COMPLETED", "FAILED"}:
                polled += 1
                job_id = str(row.get("job_id", "")).strip()

                if job_id and job_id in batch_states:
                    state = batch_states[job_id]
                else:
                    try:
                        state, _ = get_job_state_once(trial_id, row, cluster_cfg)
                    except Exception as exc:
                        print(f"[ERROR] Polling failed for Trial {trial_id:04d}: {type(exc).__name__}: {exc}")
                        set_row(
                            progress,
                            trial_id,
                            status="FAILED",
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

            time.sleep(0.5)

            result = collect_one_finished_trial(
                username,
                campaign_id,
                trial_id,
                progress[trial_id],
                cluster_cfg,
                metric_names,
                campaign_config=campaign_config,
                verbose=True,
            )
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
                print(f"[SUCCESS] Trial {trial_id:04d} collected! Metrics: { {m: result[m] for m in metric_names} }")
            elif new_state == "LOCAL_DOWNLOAD_FAIL":
                running_now += 1
            else:
                failed_now += 1

        except KeyboardInterrupt:
            save_progress(progress_csv, progress_xlsx, progress, param_names, metric_names)
            raise
        except Exception as exc:
            print(f"[ERROR] Collection exception for Trial {trial_id:04d}: {exc}")
            set_row(
                progress,
                trial_id,
                success=False,
                status="FAILED",
                state="COLLECT_EXCEPTION",
                reason=f"{type(exc).__name__}: {exc}",
            )
            save_progress(progress_csv, progress_xlsx, progress, param_names, metric_names)

    print(f"[SUMMARY] Polled: {polled} | Completed: {completed_now} | Recovered: {recovered_now} | Skipped: {skipped_now} | Failed: {failed_now}")