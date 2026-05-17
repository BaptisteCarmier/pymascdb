#!/usr/bin/env python3
"""
Compare GT vs pipeline .joblib, one output .txt per matched pair.

Run (no arguments):

    python pyVSmat.py

Each report:
  line 1: n_true / n_variables (n_variables = feuilles communes + feuilles un seul côté)
  line 2: nombre de feuilles présentes uniquement dans un fichier (GT-only + processed-only)
  puis une ligne par variable: TRUE/FALSE, types, valeurs si FALSE.

Writes under pyVSmat/compare_out/<Campaign>/<stem>.txt
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

try:
    import joblib
except ImportError as e:  # pragma: no cover
    print("pip install joblib", file=sys.stderr)
    raise SystemExit(1) from e

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore


def _here() -> Path:
    return Path(__file__).resolve().parent


GT_RESULT_ROOT = _here() / "GT" / "GT_result"
PROCESSED_PARENT = _here() / "GT" / "raw_images"
OUTDIR = _here() / "compare_out"

RTOL = 1e-5
ATOL = 1e-8
VAL_MAX_LEN = 400


def _to_mapping(obj: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(obj, Mapping):
        return obj
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        d = getattr(obj, "__dict__", None)
        if isinstance(d, Mapping):
            return d
    return None


def _leaf_paths(d: Mapping[str, Any], prefix: str = "") -> Set[str]:
    out: Set[str] = set()
    for k, v in d.items():
        p = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, Mapping) and v:
            out |= _leaf_paths(v, p)
        elif isinstance(v, Mapping) and not v:
            out.add(p)
        else:
            out.add(p)
    return out


def _get_at(d: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _type_name(x: Any) -> str:
    return type(x).__name__


def _fmt_val(x: Any) -> str:
    if np is not None and isinstance(x, np.ndarray):
        s = f"ndarray(shape={x.shape}, dtype={x.dtype})"
        if x.size <= 20 and np.issubdtype(x.dtype, np.number):
            s += f" {np.array2string(x, max_line_width=120)}"
        return s[:VAL_MAX_LEN]
    s = repr(x)
    return s[:VAL_MAX_LEN]


def _equal(a: Any, b: Any) -> bool:
    if np is not None:
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            if a.shape != b.shape or a.dtype != b.dtype:
                return False
            if a.size == 0:
                return True
            if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
                return bool(np.allclose(a, b, rtol=RTOL, atol=ATOL))
            return np.array_equal(a, b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if np is not None:
            return bool(np.isclose(float(a), float(b), rtol=RTOL, atol=ATOL))
        return abs(float(a) - float(b)) <= ATOL + RTOL * max(abs(float(a)), abs(float(b)), 1.0)
    try:
        return a == b
    except Exception:
        return False


def collect_joblibs(root: Path) -> Dict[str, Path]:
    buckets: Dict[str, List[Path]] = defaultdict(list)
    if not root.is_dir():
        return {}
    for p in sorted(root.rglob("*.joblib")):
        if p.is_file():
            buckets[p.name].append(p)
    return {name: paths[0] for name, paths in buckets.items()}


def discover_campaigns() -> List[str]:
    if not GT_RESULT_ROOT.is_dir():
        return []
    out: List[str] = []
    for p in sorted(GT_RESULT_ROOT.iterdir()):
        if p.is_dir():
            proc = PROCESSED_PARENT / f"{p.name}_PROCESSED"
            if proc.is_dir():
                out.append(p.name)
    return out


def compare_pair_lines(gt_obj: Any, pr_obj: Any) -> List[str]:
    """Build full .txt body: 2 stat lines, then one line per variable."""
    lines: List[str] = []

    gt_m = _to_mapping(gt_obj)
    pr_m = _to_mapping(pr_obj)

    if gt_m is None or pr_m is None:
        lines.append("0/1")
        lines.append("1")
        tg, tp = _type_name(gt_obj), _type_name(pr_obj)
        lines.append(f"FALSE\troot\t{tg}\t{tp}\t{_fmt_val(gt_obj)}\t{_fmt_val(pr_obj)}")
        return lines

    g_keys = _leaf_paths(gt_m)
    p_keys = _leaf_paths(pr_m)
    common = sorted(g_keys & p_keys)
    only_g = sorted(g_keys - p_keys)
    only_p = sorted(p_keys - g_keys)

    n_only_side = len(only_g) + len(only_p)
    n_common = len(common)
    n_variables = n_common + n_only_side

    n_true = 0
    for path in common:
        gv = _get_at(gt_m, path)
        pv = _get_at(pr_m, path)
        if _equal(gv, pv):
            n_true += 1

    lines.append(f"{n_true}/{n_variables}")
    lines.append(str(n_only_side))

    for path in common:
        gv = _get_at(gt_m, path)
        pv = _get_at(pr_m, path)
        tg, tp = _type_name(gv), _type_name(pv)
        if _equal(gv, pv):
            if tg == tp:
                lines.append(f"TRUE\t{path}\t{tg}")
            else:
                lines.append(f"TRUE\t{path}\t{tg}\t{tp}")
        else:
            lines.append(f"FALSE\t{path}\t{tg}\t{tp}\t{_fmt_val(gv)}\t{_fmt_val(pv)}")

    for path in only_g:
        gv = _get_at(gt_m, path)
        lines.append(f"FALSE\t{path}\tonly_gt\t{_type_name(gv)}\t\t{_fmt_val(gv)}\t")

    for path in only_p:
        pv = _get_at(pr_m, path)
        lines.append(f"FALSE\t{path}\tonly_processed\t{_type_name(pv)}\t\t\t{_fmt_val(pv)}")

    return lines


def write_pair_report(out_path: Path, body_lines: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(body_lines) + "\n", encoding="utf-8")


def main() -> None:
    campaigns = discover_campaigns()
    if not campaigns:
        print("No campaign pairs found (GT/GT_result + GT/raw_images/<name>_PROCESSED).", file=sys.stderr)
        return

    OUTDIR.mkdir(parents=True, exist_ok=True)

    for camp in campaigns:
        gt_dir = GT_RESULT_ROOT / camp
        pr_dir = PROCESSED_PARENT / f"{camp}_PROCESSED"
        gt_map = collect_joblibs(gt_dir)
        pr_map = collect_joblibs(pr_dir)
        names = sorted(set(gt_map) & set(pr_map))

        camp_out = OUTDIR / camp
        camp_out.mkdir(parents=True, exist_ok=True)

        for bn in names:
            stem = Path(bn).stem
            out_txt = camp_out / f"{stem}.txt"
            try:
                g = joblib.load(gt_map[bn])
            except Exception as exc:
                write_pair_report(
                    out_txt,
                    ["0/0", "0", f"FALSE\tload\t\t\t\tGT load error: {exc}"],
                )
                continue
            try:
                p = joblib.load(pr_map[bn])
            except Exception as exc:
                write_pair_report(
                    out_txt,
                    ["0/0", "0", f"FALSE\tload\t\t\t\tPROCESSED load error: {exc}"],
                )
                continue

            write_pair_report(out_txt, compare_pair_lines(g, p))

        print(f"{camp}: {len(names)} pair reports -> {camp_out}")


if __name__ == "__main__":
    main()
