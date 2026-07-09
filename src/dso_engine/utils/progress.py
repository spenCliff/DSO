from __future__ import annotations

import csv
import os
import time
from pathlib import Path

from openpyxl import Workbook

UTILS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UTILS_DIR.parent.parent.parent


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def progress_fieldnames(param_names: list[str], metric_names: list[str]) -> list[str]:
    base_fields = [
        "trial_id",
        "case_name",
        "varied_param",
        "level",
        "status",
        "retry_count",
        "success",
        "state",
        "reason",
        "job_id",
        "jobname",
        "remote_dir",
        "cluster_name",
        "cluster_user",
        "cluster_host",
        "cluster_remote_base",
        "cluster_base_dir",
        "geom_source_path",
        "geom_xt_path",
        "created_at",
        "updated_at",
    ]
    return base_fields + metric_names + param_names


def get_progress_paths(username: str, campaign_id: str) -> tuple[Path, Path]:
    campaign_dir = PROJECT_ROOT / "storage" / "campaigns" / username / campaign_id / "runs"
    campaign_dir.mkdir(parents=True, exist_ok=True)
    return campaign_dir / "progress_live.csv", campaign_dir / "progress_live.xlsx"


def atomic_write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    def _write_csv(target: Path) -> None:
        with target.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    _write_csv(tmp)

    last_err = None
    for _ in range(12):
        try:
            os.replace(tmp, path)
            return
        except (PermissionError, OSError) as exc:
            last_err = exc
            time.sleep(0.5)

    try:
        _write_csv(path)
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        return
    except Exception:
        pass

    if last_err is not None:
        raise last_err


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_xlsx_snapshot(path: Path, sheet_name: str, fieldnames: list[str], rows: list[dict]) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name

    ws.append(fieldnames)
    for row in rows:
        ws.append([row.get(col, "") for col in fieldnames])

    last_err = None
    for _ in range(12):
        try:
            wb.save(path)
            return
        except PermissionError as exc:
            last_err = exc
            time.sleep(0.5)

    if last_err is not None:
        raise last_err


def normalise_job_id(job_id) -> str:
    s = "" if job_id is None else str(job_id).strip()
    if not s or s.lower() == "nan":
        return ""
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def load_progress_only(progress_csv: Path, param_names: list[str], metric_names: list[str]) -> dict[int, dict]:
    rows = read_csv_rows(progress_csv)
    if not rows:
        return {}

    expected_fields = progress_fieldnames(param_names, metric_names)
    progress: dict[int, dict] = {}

    for row in rows:
        trial_id = int(row["trial_id"])
        for field in expected_fields:
            row.setdefault(field, "")
        row["job_id"] = normalise_job_id(row.get("job_id", ""))
        progress[trial_id] = row

    return progress


def save_progress(
    progress_csv: Path,
    progress_xlsx: Path,
    progress: dict[int, dict],
    param_names: list[str],
    metric_names: list[str],
) -> None:
    fieldnames = progress_fieldnames(param_names, metric_names)
    rows = [progress[k] for k in sorted(progress)]

    atomic_write_csv(progress_csv, fieldnames, rows)

    try:
        write_xlsx_snapshot(progress_xlsx, "progress", fieldnames, rows)
    except Exception as exc:
        print(f"[WARN] Could not refresh {progress_xlsx.name}: {exc}")


def set_row(progress: dict[int, dict], trial_id: int, **updates) -> None:
    row = progress[trial_id]
    for key, value in updates.items():
        if key == "job_id":
            value = normalise_job_id(value)
        row[key] = "" if value is None else str(value)
    row["updated_at"] = now_str()


def count_active(progress: dict[int, dict]) -> int:
    return sum(
        1
        for row in progress.values()
        if row.get("status", "") in {"SUBMITTED", "RUNNING"}
    )