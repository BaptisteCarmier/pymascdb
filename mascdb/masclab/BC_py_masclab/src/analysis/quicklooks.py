"""
Daily quicklooks: aggregate classified ROI data and save recap figures.

Python counterpart to masclab/prediction/merge_predictions_for_campaign.m
plus masclab/analysis/microstructure/make_masc_time_series.m (overview panels).

Reads ROI files produced under ``outdir/**/DATA/{GOOD,BAD}/`` (joblib/pkl/mat),
reuses ``src.dataio.load_roi_data`` and matches the MATLAB time-binning logic
(``Nmin_interval``, ``Nmin_shift``, ``xi_thresh``, class proportions, etc.).

Output layout: ``<save_root>/<yyyy.mm.dd>/yyyymmdd_quicklook#1.png`` (and ``#2``), filenames aligned with MATLAB.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from src.dataio.load_roi_data import load_roi_data

logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover - optional at import time
    raise ImportError(
        "matplotlib is required for quicklooks. Install with: pip install matplotlib"
    ) from e


def _compute_riming_idx(ri_degree: float) -> float:
    """Same as masclab/prediction/compute_riming_idx.m."""
    return 0.5 * (np.sin(np.pi / 4.0 * (ri_degree - 3.0)) + 1.0)


def _triplet_stem(name: str) -> str:
    """Group cam0/cam1/cam2 files for one flake (basename without _cam_*)."""
    stem = Path(name).stem
    return re.sub(r"_cam_?\d+$", "", stem, flags=re.IGNORECASE)


def _parse_roi_datetime(roi: Dict[str, Any]) -> Optional[datetime]:
    """Resolve observation time from ``roi`` (tnum or filename)."""
    tnum = roi.get("tnum")
    if isinstance(tnum, datetime):
        dt = tnum
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    if isinstance(tnum, (int, float)) and not np.isnan(tnum):
        try:
            return datetime.fromordinal(int(tnum) - 366).replace(tzinfo=timezone.utc)
        except Exception:
            pass
    name = roi.get("name") or ""
    if not name:
        return None
    parts = str(name).split("_")
    if len(parts) >= 2:
        date_str, time_str = parts[0], parts[1]
        dp = date_str.split(".")
        tp = time_str.split(".")
        if len(dp) >= 3 and len(tp) >= 3:
            try:
                return datetime(
                    int(dp[0]),
                    int(dp[1]),
                    int(dp[2]),
                    int(tp[0]),
                    int(tp[1]),
                    int(tp[2]),
                    tzinfo=timezone.utc,
                )
            except ValueError:
                return None
    return None


def _as_float(x: Any, default: float = np.nan) -> float:
    try:
        if x is None:
            return default
        return float(np.asarray(x).ravel()[0])
    except Exception:
        return default


def _as_prob_vec(p: Any, n: int) -> np.ndarray:
    if p is None:
        return np.full(n, np.nan)
    a = np.asarray(p, dtype=float).ravel()
    if a.size >= n:
        return a[:n].copy()
    out = np.full(n, np.nan)
    out[: a.size] = a
    return out


def _masc_point_from_roi(roi: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build one numeric record per ROI (per-image row)."""
    dt = _parse_roi_datetime(roi)
    if dt is None:
        return None

    e = roi.get("E") or {}
    a_e = _as_float(e.get("a"), np.nan)
    b_e = _as_float(e.get("b"), np.nan)
    ar_e = (b_e / a_e) if a_e and not np.isnan(a_e) and a_e != 0 else np.nan
    theta_e = _as_float(e.get("theta"), np.nan)

    rprobs = _as_prob_vec(roi.get("riming_probs"), 5)
    if np.all(np.isnan(rprobs)):
        ri_degree = np.nan
        ri_idx = np.nan
    else:
        ri_degree = float(np.dot(np.arange(1, 6), rprobs / (np.nansum(rprobs) + 1e-12)))
        ri_idx = _compute_riming_idx(ri_degree)

    mprob = _as_float(roi.get("melting_probs"), np.nan)

    return {
        "dt": dt,
        "name": str(roi.get("name") or ""),
        "label_id": _as_float(roi.get("label_ID"), np.nan),
        "area": _as_float(roi.get("area"), np.nan),
        "dmax": _as_float(roi.get("Dmax"), np.nan),
        "dmean": _as_float(roi.get("Dmean"), np.nan),
        "complex": _as_float(roi.get("complex"), np.nan),
        "xhi": _as_float(roi.get("xhi"), np.nan),
        "fallspeed": _as_float(roi.get("fallspeed"), np.nan),
        "ar": ar_e,
        "orientation": theta_e,
        "n_roi": _as_float(roi.get("n_roi"), np.nan),
        "roundness": _as_float(roi.get("roundness"), np.nan),
        "riming_index": ri_idx,
        "melting_id": _as_float(roi.get("melting_ID"), np.nan),
        "melting_prob": mprob,
        "label_probs": _as_prob_vec(roi.get("label_probs"), 6),
    }


def _merge_triplet_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge same-flake multi-cam rows (probabilities like merge_predictions_for_campaign.m)."""
    if len(rows) == 1:
        return rows[0].copy()

    base = rows[0].copy()
    lp = np.nansum(np.stack([r["label_probs"] for r in rows], axis=0), axis=0)
    s = np.nansum(lp)
    if s > 0:
        lp = lp / s
    lid = int(np.nanargmax(lp) + 1)
    base["label_probs"] = lp
    base["label_id"] = float(lid)

    keys_mean = (
        "area",
        "dmean",
        "fallspeed",
        "ar",
        "roundness",
        "riming_index",
        "melting_prob",
        "melting_id",
        "n_roi",
    )
    for k in keys_mean:
        base[k] = float(np.nanmean([r[k] for r in rows]))

    base["dmax"] = float(np.nanmax([r["dmax"] for r in rows]))
    base["complex"] = float(np.nanmax([r["complex"] for r in rows]))
    base["ar"] = float(np.nanmin([r["ar"] for r in rows]))
    base["orientation"] = float(np.nanmin(np.abs([r["orientation"] for r in rows])))
    base["xhi"] = float(np.nanmean([r["xhi"] for r in rows]))
    if base["melting_prob"] > 0.5:
        base["melting_id"] = 1.0
    else:
        base["melting_id"] = 0.0
    return base


def _collect_roi_paths(
    outdir: Path,
    save_format: str,
    qualities: Sequence[Literal["GOOD", "BAD"]],
) -> List[Path]:
    ext = {"joblib": ".joblib", "pkl": ".pkl", "mat": ".mat"}.get(save_format, ".joblib")
    paths: List[Path] = []
    for q in qualities:
        for p in outdir.rglob(f"DATA/{q}/*{ext}"):
            if p.is_file() and p.stem and p.stem[0].isdigit():
                paths.append(p)
    return sorted(set(paths))


def discover_time_bounds_from_outdir(
    outdir: Path, save_format: str, qualities: Sequence[Literal["GOOD", "BAD"]] = ("GOOD",)
) -> Optional[Tuple[datetime, datetime]]:
    """Min/max timestamps inferred from ROI files (for ``starthr_vec: all``)."""
    tmin: Optional[datetime] = None
    tmax: Optional[datetime] = None
    for path in _collect_roi_paths(outdir, save_format, qualities):
        try:
            roi = load_roi_data(path, save_format)
            dt = _parse_roi_datetime(roi)
            if dt is None:
                continue
            tmin = dt if tmin is None or dt < tmin else tmin
            tmax = dt if tmax is None or dt > tmax else tmax
        except Exception as ex:
            logger.debug("skip %s: %s", path, ex)
    if tmin is None or tmax is None:
        return None
    return tmin, tmax


@dataclass
class QuicklooksOptions:
    pixres_mm: float = 33.5 / 1000.0
    xi_thresh: float = 9.0
    Nmin_interval: int = 30
    Nmin_shift: int = 10
    Nclasses_masc: int = 6
    MASC_classes: Tuple[str, ...] = ("SP", "CC", "PC", "AG", "GR", "CPC")
    MASC_classes_desired: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    N_MascSamples_min: int = 0
    use_triplet: bool = False
    OR_180: bool = True
    savefigs: bool = True
    overview_classif: bool = True
    overview_microstruct: bool = True
    disp_now: bool = False
    qualities: Tuple[str, ...] = ("GOOD",)


def _hsv_cmasc() -> np.ndarray:
    """Approximate MATLAB cmasc (hsv(6) with permutations / green tweak)."""
    hues = np.linspace(0.0, 5.0 / 6.0, 6, endpoint=True)
    cm = np.zeros((6, 3))
    for i, h in enumerate(hues):
        cm[i] = matplotlib.colors.hsv_to_rgb((h, 0.85, 0.95))
    cm[3] = matplotlib.colors.hsv_to_rgb((5.0 / 6.0, 0.85, 0.95))
    cm[5] = np.array([49, 163, 84], dtype=float) / 255.0
    return cm


def _aggregate_bins(
    rows: List[Dict[str, Any]],
    t0: datetime,
    t1: datetime,
    opt: QuicklooksOptions,
) -> Dict[str, Any]:
    """Time bins ``t0`` inclusive to ``t1`` exclusive (same construction as MATLAB)."""
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=timezone.utc)
    if t1.tzinfo is None:
        t1 = t1.replace(tzinfo=timezone.utc)
    t0 = t0.astimezone(timezone.utc)
    t1 = t1.astimezone(timezone.utc)

    shift = timedelta(minutes=opt.Nmin_shift)
    width = timedelta(minutes=opt.Nmin_interval)
    tgrid: List[datetime] = []
    t = t0
    while t < t1:
        tgrid.append(t)
        t += shift
    if not tgrid:
        return {}

    nbin = len(tgrid)
    tgrid2 = [tg + width for tg in tgrid]

    classes_des = np.array(opt.MASC_classes_desired, dtype=int)
    nclass = opt.Nclasses_masc

    Nmasc_all = np.zeros(nbin)
    Nmasc_filtered = np.zeros(nbin)
    dom_class = np.full(nbin, np.nan)
    sclass = np.full((nbin, nclass), np.nan)
    melting = np.full(nbin, np.nan)
    riming = np.full(nbin, np.nan)

    dmax_m = np.full(nbin, np.nan)
    dmax_med = np.full(nbin, np.nan)
    dmax_s = np.full(nbin, np.nan)
    nroi_m = np.full(nbin, np.nan)
    nroi_med = np.full(nbin, np.nan)
    nroi_s = np.full(nbin, np.nan)
    ar_m = np.full(nbin, np.nan)
    ar_med = np.full(nbin, np.nan)
    ar_s = np.full(nbin, np.nan)
    arr_m = np.full(nbin, np.nan)
    arr_med = np.full(nbin, np.nan)
    arr_s = np.full(nbin, np.nan)
    or_m = np.full(nbin, np.nan)
    or_med = np.full(nbin, np.nan)
    or_s = np.full(nbin, np.nan)
    cplx_m = np.full(nbin, np.nan)
    cplx_med = np.full(nbin, np.nan)
    cplx_s = np.full(nbin, np.nan)
    fs_m = np.full(nbin, np.nan)
    fs_med = np.full(nbin, np.nan)
    fs_s = np.full(nbin, np.nan)

    xs = np.array([r["dmax"] for r in rows], dtype=float)

    for i, (ta, tb) in enumerate(zip(tgrid, tgrid2)):
        idx_all = [j for j, r in enumerate(rows) if ta <= r["dt"] < tb]
        if not idx_all:
            continue
        idx_f = [
            j
            for j in idx_all
            if not np.isnan(rows[j]["xhi"]) and rows[j]["xhi"] >= opt.xi_thresh
        ]
        idx_f_nosp = [j for j in idx_f if rows[j]["label_id"] != 1.0]

        Nmasc_all[i] = len(idx_all)
        Nmasc_filtered[i] = len(idx_f)

        xa = xs[idx_all]

        if idx_f:
            labs = [int(rows[j]["label_id"]) for j in idx_f if not np.isnan(rows[j]["label_id"])]
            if labs:
                dom_class[i] = float(max(set(labs), key=labs.count))
            counts = np.zeros(nclass)
            for k in range(1, nclass + 1):
                counts[k - 1] = sum(1 for j in idx_f if rows[j]["label_id"] == float(k))
            sclass[i, :] = counts
            melting[i] = float(np.nanmean([rows[j]["melting_id"] for j in idx_f]))
            if idx_f_nosp:
                riming[i] = float(np.nanmean([rows[j]["riming_index"] for j in idx_f_nosp]))
            else:
                riming[i] = np.nan

        if idx_all:
            dmax_m[i] = float(np.nanmean(xa))
            dmax_med[i] = float(np.nanmedian(xa))
            dmax_s[i] = float(np.nanstd(xa))
            nroi_m[i] = float(np.nanmean([rows[j]["n_roi"] for j in idx_all]))
            nroi_med[i] = float(np.nanmedian([rows[j]["n_roi"] for j in idx_all]))
            nroi_s[i] = float(np.nanstd([rows[j]["n_roi"] for j in idx_all]))
            ar_m[i] = float(np.nanmean([rows[j]["ar"] for j in idx_all]))
            ar_med[i] = float(np.nanmedian([rows[j]["ar"] for j in idx_all]))
            ar_s[i] = float(np.nanstd([rows[j]["ar"] for j in idx_all]))
            arr_m[i] = float(np.nanmean([rows[j]["roundness"] for j in idx_all]))
            arr_med[i] = float(np.nanmedian([rows[j]["roundness"] for j in idx_all]))
            arr_s[i] = float(np.nanstd([rows[j]["roundness"] for j in idx_all]))
            orv = np.array([rows[j]["orientation"] for j in idx_all], dtype=float)
            if not opt.OR_180:
                orv = np.abs(orv)
            or_m[i] = float(np.nanmean(orv))
            or_med[i] = float(np.nanmedian(orv))
            or_s[i] = float(np.nanstd(orv))
            cplx_m[i] = float(np.nanmean([rows[j]["complex"] for j in idx_all]))
            cplx_med[i] = float(np.nanmedian([rows[j]["complex"] for j in idx_all]))
            cplx_s[i] = float(np.nanstd([rows[j]["complex"] for j in idx_all]))
            fs_a = np.array([rows[j]["fallspeed"] for j in idx_all], dtype=float)
            fs_a[(fs_a >= 15.0) | (fs_a < 0.1)] = np.nan
            fs_m[i] = float(np.nanmean(fs_a))
            fs_med[i] = float(np.nanmedian(fs_a))
            fs_s[i] = float(np.nanstd(fs_a))

    sclass_norm = np.zeros((nbin, len(classes_des)))
    for i in range(nbin):
        sub = sclass[i, classes_des - 1]
        ssum = np.nansum(sub)
        if ssum > 0:
            sclass_norm[i, :] = sub / ssum
        else:
            sclass_norm[i, :] = np.nan

    mask_low = Nmasc_filtered < opt.N_MascSamples_min
    dom_class[mask_low] = np.nan
    sclass[mask_low, :] = np.nan
    melting[mask_low] = np.nan
    riming[mask_low] = np.nan

    return {
        "tgrid": np.array(tgrid, dtype="datetime64[ns]"),
        "Nmasc_all": Nmasc_all,
        "Nmasc_filtered": Nmasc_filtered,
        "dom_class": dom_class,
        "sclass": sclass,
        "sclass_norm": sclass_norm,
        "melting": melting,
        "riming": riming,
        "Dmax": (dmax_m, dmax_med, dmax_s),
        "Nroi": (nroi_m, nroi_med, nroi_s),
        "AR": (ar_m, ar_med, ar_s),
        "ArR": (arr_m, arr_med, arr_s),
        "OR": (or_m, or_med, or_s),
        "cplx": (cplx_m, cplx_med, cplx_s),
        "fs": (fs_m, fs_med, fs_s),
    }


def _fill_band(ax, t, mean, std, color, alpha=0.35):
    lo = mean - std
    hi = mean + std
    ax.fill_between(t, lo, hi, color=color, alpha=alpha, linewidth=0)
    ax.plot(t, mean, color=color, linewidth=1.5)


def _save_overview_classif(
    data: Dict[str, Any],
    t0: datetime,
    t1: datetime,
    opt: QuicklooksOptions,
    savedir: Path,
    date_tag: str,
) -> None:
    cmasc = _hsv_cmasc()
    tgrid = data["tgrid"]
    tnum = mdates.date2num(tgrid)

    date_title = f"{t0.strftime('%Y.%m.%d %H:%M')} - {t1.strftime('%Y.%m.%d %H:%M')}"
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("white")

    ax1 = fig.add_subplot(4, 1, 1)
    ax1.plot(tnum, data["Nmasc_all"], "k-", lw=2, label="total")
    if opt.xi_thresh > 0:
        ax1.plot(tnum, data["Nmasc_filtered"], "k--", lw=2, label=f"filtered xi>={opt.xi_thresh:g}")
    ax1.set_ylabel("# images" if not opt.use_triplet else "# triplets")
    ax1.set_title(date_title)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")
    ax1.xaxis_date()
    ax1.set_xlim(tnum[0], tnum[-1])

    ax2 = fig.add_subplot(4, 1, (2, 3))
    y_stack = np.nan_to_num(data["sclass_norm"], nan=0.0)
    labs = [opt.MASC_classes[k - 1] for k in opt.MASC_classes_desired]
    colors = [cmasc[k - 1] for k in opt.MASC_classes_desired]
    ax2.stackplot(tnum, y_stack.T, labels=labs, colors=colors, alpha=0.9)
    ax2.plot(tnum, data["riming"], color="white", ls="-.", lw=3, label="R_i")
    ax2.set_ylabel("Proportions")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", ncol=2, fontsize=9)
    ax2.xaxis_date()
    ax2.set_xlim(tnum[0], tnum[-1])

    ax3 = fig.add_subplot(4, 1, 4)
    ax3.plot(tnum, data["riming"], "b-", lw=2, label="R_i")
    ax3.plot(tnum, data["melting"], color=(241 / 255, 105 / 255, 19 / 255), lw=2, label="% wet")
    ax3.set_ylabel("[-]")
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right")
    ax3.xaxis_date()
    ax3.set_xlim(tnum[0], tnum[-1])
    fig.autofmt_xdate()
    fig.tight_layout()
    out = savedir / f"{date_tag}_quicklook#1.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _save_overview_microstruct(
    data: Dict[str, Any],
    t0: datetime,
    t1: datetime,
    opt: QuicklooksOptions,
    savedir: Path,
    date_tag: str,
) -> None:
    tgrid = data["tgrid"]
    tnum = mdates.date2num(tgrid)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    cmap15 = (0.15, 0.35, 0.75, 1.0)

    dmax_m, _, dmax_s = data["Dmax"]
    fs_m, _, fs_s = data["fs"]
    ar_m, _, ar_s = data["AR"]
    or_m, _, or_s = data["OR"]
    arr_m, _, arr_s = data["ArR"]
    cplx_m, _, cplx_s = data["cplx"]

    pix = opt.pixres_mm
    dmax_mm = dmax_m * pix
    dmax_sd = dmax_s * pix

    panels = [
        (axes[0, 0], dmax_mm, dmax_sd, "Dmax", "[mm]", "r"),
        (axes[0, 1], fs_m, fs_s, "Fallspeed", "[m/s]", "b"),
        (axes[1, 0], ar_m, ar_s, "Aspect Ratio", "[-]", cmap15),
        (axes[1, 1], or_m, or_s, "Orientation" + (" [-90; 90]" if opt.OR_180 else " [0; 90]"), "[°]", "r"),
        (axes[2, 0], arr_m, arr_s, "Area Ratio", "[-]", cmap15),
        (axes[2, 1], cplx_m, cplx_s, "Complexity", "[-]", "b"),
    ]
    for ax, m, s, title, ylab, col in panels:
        _fill_band(ax, tnum, m, s, col, 0.35)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(tnum[0], tnum[-1])
        ax.xaxis_date()
    fig.suptitle(
        f"{t0.strftime('%Y.%m.%d %H:%M')} - {t1.strftime('%Y.%m.%d %H:%M')}",
        fontsize=12,
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    out = savedir / f"{date_tag}_quicklook#2.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)


def load_masc_points_for_interval(
    outdir: Path,
    save_format: str,
    t_start: datetime,
    t_stop: datetime,
    use_triplet: bool,
    qualities: Sequence[str] = ("GOOD",),
) -> List[Dict[str, Any]]:
    """Load ROI files with ``t_start <= dt < t_stop`` (half-open time interval)."""
    if t_start.tzinfo is None:
        t_start = t_start.replace(tzinfo=timezone.utc)
    if t_stop.tzinfo is None:
        t_stop = t_stop.replace(tzinfo=timezone.utc)
    t_start = t_start.astimezone(timezone.utc)
    t_stop = t_stop.astimezone(timezone.utc)

    qtuple = tuple(q for q in qualities if q in ("GOOD", "BAD"))
    paths = _collect_roi_paths(outdir, save_format, qtuple)  # type: ignore[arg-type]
    raw: List[Dict[str, Any]] = []
    for path in paths:
        try:
            roi = load_roi_data(path, save_format)
            row = _masc_point_from_roi(roi)
            if row is None:
                continue
            if not (t_start <= row["dt"] < t_stop):
                continue
            raw.append(row)
        except Exception as ex:
            logger.warning("Could not load %s: %s", path, ex)

    if not use_triplet:
        return raw

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in raw:
        stem = _triplet_stem(str(row["name"]))
        groups.setdefault(stem, []).append(row)
    merged: List[Dict[str, Any]] = []
    for _stem, g in groups.items():
        merged.append(_merge_triplet_group(g))
    return merged


def run_quicklooks_for_range(
    outdir: Path,
    save_format: str,
    day_start: datetime,
    day_end: datetime,
    opt: QuicklooksOptions,
    save_root: Path,
) -> List[Path]:
    """
    Build recap figures for ``[day_start, day_end)`` (typically one UTC day).

    Figures are written under ``save_root / 'yyyy.mm.dd' /`` (one folder per UTC calendar day).
    Filenames match MATLAB ``make_masc_time_series.m``: ``yyyymmdd_quicklook#1.png`` and
    ``yyyymmdd_quicklook#2.png`` (``datestr(tgrid(1),'yyyymmdd')`` + ``_quicklook#…``).
    Returns paths of PNG files created (may be empty if no data).
    """
    if day_start.tzinfo is None:
        day_start = day_start.replace(tzinfo=timezone.utc)
    if day_end.tzinfo is None:
        day_end = day_end.replace(tzinfo=timezone.utc)
    day_start = day_start.astimezone(timezone.utc)
    day_end = day_end.astimezone(timezone.utc)

    qualities = tuple(q for q in opt.qualities if q in ("GOOD", "BAD"))
    if not qualities:
        qualities = ("GOOD",)

    rows = load_masc_points_for_interval(
        outdir,
        save_format,
        day_start,
        day_end,
        use_triplet=opt.use_triplet,
        qualities=qualities,
    )
    created: List[Path] = []
    if not rows:
        logger.warning(
            "No ROI rows in [%s, %s] under %s — skipping quicklooks.",
            day_start,
            day_end,
            outdir,
        )
        return created

    data = _aggregate_bins(rows, day_start, day_end, opt)
    if not data:
        return created

    if float(np.nansum(data["Nmasc_all"])) == 0:
        return created

    subdir = save_root / day_start.strftime("%Y.%m.%d")
    subdir.mkdir(parents=True, exist_ok=True)
    date_tag = day_start.strftime("%Y%m%d")

    if opt.savefigs:
        if opt.overview_classif:
            _save_overview_classif(data, day_start, day_end, opt, subdir, date_tag)
            created.append(subdir / f"{date_tag}_quicklook#1.png")
        if opt.overview_microstruct:
            _save_overview_microstruct(data, day_start, day_end, opt, subdir, date_tag)
            created.append(subdir / f"{date_tag}_quicklook#2.png")

    return created


def run_quicklooks_campaign(
    outdir: Path,
    save_format: str,
    t_min: datetime,
    t_max: datetime,
    opt: QuicklooksOptions,
    save_root: Path,
) -> List[Path]:
    """Run :func:`run_quicklooks_for_range` for each calendar day overlapping the window."""
    if t_min.tzinfo is None:
        t_min = t_min.replace(tzinfo=timezone.utc)
    if t_max.tzinfo is None:
        t_max = t_max.replace(tzinfo=timezone.utc)
    t_min = t_min.astimezone(timezone.utc)
    t_max = t_max.astimezone(timezone.utc)

    if t_max <= t_min:
        return []

    d0 = t_min.date()
    d1 = t_max.date()
    all_created: List[Path] = []
    d = d0
    while d <= d1:
        day_start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        day_end = day_start + timedelta(days=1)
        win0 = max(day_start, t_min)
        win1 = min(day_end, t_max)
        if win1 > win0:
            all_created.extend(
                run_quicklooks_for_range(
                    outdir, save_format, win0, win1, opt, save_root
                )
            )
        d = d + timedelta(days=1)
    return all_created
