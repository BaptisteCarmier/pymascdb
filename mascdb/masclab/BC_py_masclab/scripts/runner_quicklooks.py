"""
Generate daily quicklook PNGs from classified ROI files under ``label.outdir``.

Equivalent workflow to masclab/processing/MASC_process_classify_quicklooks.m (part 3):
merge predictions over time bins and save under ``Quicklooks/<yyyy.mm.dd>/`` the files
``yyyymmdd_quicklook#1.png`` / ``yyyymmdd_quicklook#2.png`` (same naming as MATLAB ``make_masc_time_series.m``).

Usage (from ``BC_py_masclab/``):
    python scripts/runner_quicklooks.py

Requires matplotlib. Run classification first so ``label_ID`` / riming / melting exist in ROI files.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


def _load_quicklooks_section(yaml_path: Path) -> dict:
    if yaml is None:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("quicklooks") or {}


def _opt_from_dict(d: dict, process_use_triplet: bool):
    from src.analysis.quicklooks import QuicklooksOptions

    return QuicklooksOptions(
        pixres_mm=float(d.get("pixres_mm", 33.5 / 1000.0)),
        xi_thresh=float(d.get("xi_thresh", 9.0)),
        Nmin_interval=int(d.get("Nmin_interval", 30)),
        Nmin_shift=int(d.get("Nmin_shift", 10)),
        Nclasses_masc=int(d.get("Nclasses_masc", 6)),
        MASC_classes=tuple(d.get("MASC_classes", ("SP", "CC", "PC", "AG", "GR", "CPC"))),
        MASC_classes_desired=tuple(d.get("MASC_classes_desired", (1, 2, 3, 4, 5, 6))),
        N_MascSamples_min=int(d.get("N_MascSamples_min", 0)),
        use_triplet=(
            process_use_triplet
            if d.get("use_triplet") is None
            else bool(d.get("use_triplet"))
        ),
        OR_180=bool(d.get("OR_180", True)),
        savefigs=bool(d.get("savefigs", True)),
        overview_classif=bool(d.get("overview_classif", True)),
        overview_microstruct=bool(d.get("overview_microstruct", True)),
        disp_now=bool(d.get("disp_now", False)),
        qualities=tuple(d.get("qualities", ("GOOD",))),
    )


def main() -> None:
    from src.core.config import load_from_yaml
    from src.analysis.quicklooks import (
        discover_time_bounds_from_outdir,
        run_quicklooks_campaign,
    )

    yaml_path = root_dir / "config" / "config.yaml"
    label_cfg, proc_cfg = load_from_yaml(yaml_path)
    ql_raw = _load_quicklooks_section(yaml_path)

    outdir = Path(label_cfg.outdir)
    save_fmt = str(proc_cfg.save_format)

    savepath_raw = ql_raw.get("savepath", "auto")
    if isinstance(savepath_raw, str) and savepath_raw.lower() == "auto":
        save_root = outdir / "Quicklooks"
    else:
        save_root = Path(savepath_raw)

    opt = _opt_from_dict(ql_raw, proc_cfg.use_triplet_algo)

    t_min = label_cfg.starthr_vec
    t_max = label_cfg.endhr_vec
    if t_min.tzinfo is None:
        t_min = t_min.replace(tzinfo=timezone.utc)
    if t_max.tzinfo is None:
        t_max = t_max.replace(tzinfo=timezone.utc)
    t_min = t_min.astimezone(timezone.utc)
    t_max = t_max.astimezone(timezone.utc)

    bounds = None
    if t_min.year < 1900 or t_max.year > 2100:
        bounds = discover_time_bounds_from_outdir(
            outdir,
            save_fmt,
            tuple(q for q in opt.qualities if q in ("GOOD", "BAD")),  # type: ignore[arg-type]
        )
        if bounds is None:
            print("ERROR: No ROI files found under outdir (needed for open-ended time window).")
            sys.exit(1)
        dmin, dmax = bounds
        dmin = dmin.astimezone(timezone.utc)
        dmax = dmax.astimezone(timezone.utc)
        if t_min.year < 1900:
            t_min = dmin
        if t_max.year > 2100:
            t_max = dmax
        print(f"Time bounds (after discovery): {t_min} … {t_max}")

    print("=" * 70)
    print("MASC quicklooks (aggregation + recap figures)")
    print("=" * 70)
    print(f"  outdir     : {outdir}")
    print(f"  save_root  : {save_root}")
    print(f"  use_triplet: {opt.use_triplet}")
    print(f"  qualities  : {opt.qualities}")
    print()

    paths = run_quicklooks_campaign(outdir, save_fmt, t_min, t_max, opt, save_root)
    print(f"Wrote {len(paths)} figure file(s).")
    for p in paths[:20]:
        print(f"  {p}")
    if len(paths) > 20:
        print(f"  … ({len(paths) - 20} more)")


if __name__ == "__main__":
    main()
