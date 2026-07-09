from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pythoncom
import win32com.client as win32

UTILS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UTILS_DIR.parent.parent.parent

SW_DOC_PART = 1
SW_OPEN_SILENT = 1


def _ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _trial_paths(campaign_dir: Path, iteration: int) -> tuple[Path, Path, Path]:
    trial_dir = campaign_dir / "runs" / f"trial_{iteration:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    trial_part_path = trial_dir / f"trial_{iteration:04d}.SLDPRT"
    export_xt_path = trial_dir / f"geom_iter_{iteration:04d}.x_t"

    return trial_dir, trial_part_path, export_xt_path


def _format_equation(name: str, value: float | int) -> str:
    return f'"{name}" = {float(value):.6f}'


def _find_equation_index(eq_mgr, variable_name: str, n_eqs: int) -> int:
    needle = f'"{variable_name}"'
    for i in range(n_eqs):
        eq = str(eq_mgr.Equation(i))
        if needle in eq:
            return i
    raise ValueError(f"Global variable '{variable_name}' not found in Equation Manager.")


def _open_solidworks():
    pythoncom.CoInitialize()
    sw_app = win32.Dispatch("SldWorks.Application")
    try:
        sw_app.Visible = True
    except Exception:
        pass
    return sw_app


def _open_part(sw_app, part_path: Path):
    errors = win32.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
    warnings = win32.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)

    model = sw_app.OpenDoc6(
        str(part_path),
        SW_DOC_PART,
        SW_OPEN_SILENT,
        "",
        errors,
        warnings,
    )

    if model is None:
        raise RuntimeError(f"Failed to open SolidWorks part: {part_path}")

    return model, int(errors.value), int(warnings.value)


def _update_global_variables(model, params: dict[str, float]) -> None:
    eq_mgr = model.GetEquationMgr
    if eq_mgr is None:
        raise RuntimeError("Could not access SolidWorks Equation Manager.")

    n_eqs = int(eq_mgr.GetCount)

    for name, value in params.items():
        idx = _find_equation_index(eq_mgr, name, n_eqs)
        old_eq = str(eq_mgr.Equation(idx))
        new_eq = _format_equation(name, value)

        eq_mgr.Equation(idx, new_eq)
        print(f"[CAD] {name}: {old_eq} -> {new_eq}")


def _rebuild_model(model) -> None:
    try:
        ok = model.EditRebuild3()
    except TypeError:
        ok = model.EditRebuild3

    print(f"[CAD] Rebuild result: {ok}")
    time.sleep(1.0)

    try:
        model.ForceRebuild3(False)
    except Exception:
        pass

    time.sleep(1.0)


def _export_parasolid(model, export_path: Path) -> Path:
    export_path.parent.mkdir(parents=True, exist_ok=True)

    if export_path.exists():
        export_path.unlink()

    save_ok = None

    try:
        save_ok = model.SaveAs3(str(export_path), 0, 0)
    except Exception:
        try:
            save_ok = model.SaveAs2(str(export_path), 0, True, True)
        except Exception as e:
            raise RuntimeError(f"Parasolid export call failed: {e}")

    time.sleep(1.0)

    if not export_path.exists():
        raise RuntimeError(f"Parasolid export failed; file not created: {export_path}")

    print(f"[CAD] Exported Parasolid: {export_path}")
    print(f"[CAD] Save return value: {save_ok}")
    return export_path


def _close_doc(sw_app, model) -> None:
    try:
        title = model.GetTitle()
    except TypeError:
        title = model.GetTitle

    try:
        sw_app.CloseDoc(title)
    except Exception:
        pass


def generate_geom(
    username: str, 
    campaign_id: str, 
    master_part_name: str, 
    iteration: int, 
    params: dict[str, float], 
    keep_trial_part: bool = True
) -> dict:

    campaign_dir = PROJECT_ROOT / "storage" / "campaigns" / username / campaign_id
    master_part_path = campaign_dir / "inputs" / master_part_name

    _ensure_exists(master_part_path, "Master SolidWorks part")

    trial_dir, trial_part_path, export_xt_path = _trial_paths(campaign_dir, iteration)

    print("=" * 80)
    print(f"[CAD] Trial {iteration:04d}")
    print(f"[CAD] Master part : {master_part_path}")
    print(f"[CAD] Trial dir    : {trial_dir}")
    print(f"[CAD] Trial part   : {trial_part_path}")
    print(f"[CAD] Export path  : {export_xt_path}")
    print("=" * 80)

    shutil.copy2(master_part_path, trial_part_path)
    print(f"[CAD] Copied master part -> {trial_part_path}")

    sw_app = None
    model = None

    try:
        sw_app = _open_solidworks()
        model, open_errors, open_warnings = _open_part(sw_app, trial_part_path)

        print(f"[CAD] Open errors   : {open_errors}")
        print(f"[CAD] Open warnings : {open_warnings}")

        _update_global_variables(model, params)
        _rebuild_model(model)

        try:
            model.Save3(1, 0, 0)
        except Exception:
            pass

        export_path = _export_parasolid(model, export_xt_path)

        return {
            "geom_xt_path": str(export_path.resolve()),
            "trial_part_path": str(trial_part_path.resolve()),
        }

    finally:
        if sw_app is not None and model is not None:
            _close_doc(sw_app, model)

        if not keep_trial_part and trial_part_path.exists():
            try:
                trial_part_path.unlink()
                print(f"[CAD] Deleted temporary trial part: {trial_part_path}")
            except Exception as e:
                print(f"[CAD] Warning: could not delete trial part: {e}")