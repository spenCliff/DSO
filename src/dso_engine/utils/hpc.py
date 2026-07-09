from __future__ import annotations

import csv
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

@dataclass
class HPCConfig:
    user: str
    host: str
    remote_base: str 
    base_dir: str          
    ssh_key: str | None = None

SSH_COMMON_OPTS = [
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=120",
    "-o", "ConnectionAttempts=2",
    "-o", "ServerAliveInterval=300",
    "-o", "ServerAliveCountMax=2",
    "-o", "StrictHostKeyChecking=accept-new",
]

def _run(cmd: list[str], timeout_s: int | None = 60, retries: int = 2) -> str:
    last_err = None
    for attempt in range(retries + 1):
        try:
            p = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            if p.returncode == 0:
                return p.stdout.strip()

            stderr = (p.stderr or "").lower()
            msg = f"Command failed:\n cmd: {' '.join(cmd)}\n stdout: {p.stdout}\n stderr: {p.stderr}\n"
            last_err = RuntimeError(msg)
            
            transient = any(x in stderr for x in [
                "connection reset",
                "kex_exchange_identification",
                "timed out",
                "connection refused",
                "broken pipe",
            ])
            if not transient or attempt == retries:
                raise last_err
        except subprocess.TimeoutExpired:
            timeout_msg = "no limit" if timeout_s is None else f"{timeout_s}s"
            last_err = TimeoutError(f"Timeout after {timeout_msg}: {' '.join(cmd)}")
            if attempt == retries:
                raise last_err
        time.sleep(5 * (attempt + 1))
    raise last_err

def _scp_to(cfg: HPCConfig, local_path: Path, remote_path: str, timeout_s: int | None = 120, retries: int = 2) -> None:
    if not local_path.exists():
        raise FileNotFoundError(f"Missing local file: {local_path}")
    cmd = ["scp"]
    if cfg.ssh_key:
        cmd += ["-i", cfg.ssh_key]
    cmd += SSH_COMMON_OPTS
    cmd += [str(local_path), f"{cfg.user}@{cfg.host}:{remote_path}"]
    _run(cmd, timeout_s=timeout_s, retries=retries)

def _scp_from(cfg: HPCConfig, remote_path: str, local_path: Path, timeout_s: int | None = 120, retries: int = 2) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_suffix(local_path.suffix + ".part")
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except OSError:
        pass
    cmd = ["scp"]
    if cfg.ssh_key:
        cmd += ["-i", cfg.ssh_key]
    cmd += SSH_COMMON_OPTS
    cmd += [f"{cfg.user}@{cfg.host}:{remote_path}", str(tmp_path)]
    try:
        _run(cmd, timeout_s=timeout_s, retries=retries)
        os.replace(tmp_path, local_path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise

def _ssh(cfg: HPCConfig, remote_cmd: str) -> str:
    cmd = ["ssh"]
    if cfg.ssh_key:
        cmd += ["-i", cfg.ssh_key]
    cmd += SSH_COMMON_OPTS
    cmd += [f"{cfg.user}@{cfg.host}", remote_cmd]
    return _run(cmd, timeout_s=45, retries=2)

def _remote_exists(cfg: HPCConfig, remote_path: str) -> bool:
    out = _ssh(cfg, f"bash -lc 'test -f {remote_path} && echo YES || echo NO'").strip()
    return out == "YES"

def check_cluster_health(cfg: HPCConfig) -> bool:
    try:
        _ssh(cfg, "bash -lc 'echo ok'")
        return True
    except Exception:
        return False

def _try_sacct_state(cfg: HPCConfig, job_id: str) -> str | None:
    cmd = f"bash -lc 'command -v sacct >/dev/null 2>&1 || exit 0; sacct -j {job_id} --format=State --noheader | head -n 1 | tr -d \" \"'"
    out = _ssh(cfg, cmd).strip()
    return out or None

def check_job_state(cfg: HPCConfig, job_id: str, remote_dir: str, jobname: str) -> tuple[str, str | None]:
    slurm_out = f"{remote_dir}/{jobname}_{job_id}.out"
    if not _remote_exists(cfg, slurm_out):
        return "PENDING", None
        
    state = _try_sacct_state(cfg, job_id)
    if state:
        base = state.split("+")[0]
        if base in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}:
            return base, slurm_out

    tail = _ssh(cfg, f"bash -lc 'tail -n 120 {slurm_out} || true'").lower()
    if "finished at:" in tail:
        return "COMPLETED", slurm_out
    if "cancelled" in tail or "canceled" in tail:
        return "CANCELLED", slurm_out
    if "out of memory" in tail:
        return "OUT_OF_MEMORY", slurm_out
    if "error" in tail or "segmentation fault" in tail or "slurmstepd:" in tail:
        return "FAILED", slurm_out
        
    return "RUNNING", slurm_out

def submit_slurm_trial(cfg: HPCConfig, trial_id: int, local_payload: Path, remote_payload_name: str) -> tuple[str, str, str]:
    jobname = f"DSO_T{trial_id:04d}"
    remote_dir = f"{cfg.remote_base}/trial_{trial_id:04d}"
    
    _ssh(cfg, f"bash -lc 'mkdir -p {remote_dir}'")
    _scp_to(cfg, local_payload, f"{remote_dir}/{remote_payload_name}")
    
    base_run = f"{cfg.base_dir}/run.sh"
    out = _ssh(cfg, f"bash -lc 'cd {remote_dir} && sbatch --job-name={jobname} {base_run}'")
    
    m = re.search(r"Submitted batch job (\d+)", out)
    if not m:
        raise RuntimeError(f"Could not parse sbatch output: {out}")
        
    return remote_dir, m.group(1), jobname

def download_remote_file(cfg: HPCConfig, remote_path: str, local_path: Path) -> bool:
    try:
        _scp_from(cfg, remote_path, local_path, timeout_s=120, retries=2)
        return True
    except Exception:
        return False

def parse_generic_csv_metrics(path: Path, target_metrics: list[str], csv_column_mapping: dict[str, str]) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    if not rows:
        raise ValueError(f"Target metric data file empty: {path.name}")
        
    last_row = rows[-1]
    extracted = {}
    for metric in target_metrics:
        csv_col = csv_column_mapping.get(metric, metric)
        if csv_col in last_row:
            try:
                extracted[metric] = float(last_row[csv_col])
            except (ValueError, TypeError):
                extracted[metric] = 0.0
        else:
            extracted[metric] = 0.0
    return extracted