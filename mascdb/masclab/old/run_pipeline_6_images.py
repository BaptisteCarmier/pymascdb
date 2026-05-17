import csv
import argparse
from pathlib import Path
import time
from typing import Optional

# Ensure project root is on sys.path so "src" package can be imported when running the script directly
import sys
ROOT = Path(__file__).resolve().parents[1]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import ProcessingConfig
from src.processing.single import process_single_image

# Default data folder (relative to repository root)
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "tests" / "data" / "2015.06.20" / "09"


def iter_image_files(d: Path):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
    for ext in exts:
        for p in sorted(d.glob(ext)):
            yield p


def run(dir_path: Path, out_csv: Path, limit: Optional[int] = 6, cam_id: int = 1):
    dir_path = Path(dir_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"Data dir not found: {dir_path}")

    proc = ProcessingConfig()
    summary = []
    counts = {}
    counts_label = {}
    start_all = time.time()

    FLAG_LABELS = {
        2: "GOOD",
        1: "BAD",
        0: "NO_ROI",
        -1: "ERROR"
    }

    for i, img_path in enumerate(iter_image_files(dir_path)):
        if limit is not None and i >= limit:
            break

        pic_info = {
            "filename": img_path.name,
            "cam": cam_id,       # MATLAB convention: 1/2/3
            "id": i + 1,
            "fallspeed": None,
            "time_num": None,
        }

        t0 = time.time()
        try:
            roi, flag, timing = process_single_image(img_path, pic_info, proc, verbose=True)
            error = ""
        except Exception as exc:
            roi, flag, timing = None, -1, {}
            error = str(exc)

        elapsed = time.time() - t0
        n_fields = len(roi) if roi is not None else 0

        label = FLAG_LABELS.get(flag, str(flag))

        

        summary.append({
            "file": str(img_path),
            "flag": int(flag),
            "flag_label": label,
            "n_fields": int(n_fields),
            "t_total": float(elapsed),
            "t_loading": float(timing.get("loading", 0.0)),
            "t_clutter": float(timing.get("clutter", 0.0)),
            "t_edging": float(timing.get("edging", 0.0)),
            "t_roiying": float(timing.get("roiying", 0.0)),
            "t_feature": float(timing.get("feature", 0.0)),
            "error": error,
        })
        counts[flag] = counts.get(flag, 0) + 1
        counts_label[label] = counts_label.get(label, 0) + 1
        print(f"[{i+1}] {img_path.name} flag={flag}({label}) fields={n_fields} err={'-' if not error else error[:80]}")

    total_time = time.time() - start_all
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()) if summary else ["file","flag","flag_label"])
        writer.writeheader()
        for row in summary:
            writer.writerow(row)

    print("roi keys:", list(roi.keys()) if roi else "None")
    print("Done. processed:", len(summary), "total_time(s):", round(total_time, 2))
    print("flag counts (numeric):", counts)
    print("flag counts (labels):", counts_label)
    print("csv:", out_csv)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run MASC pipeline on up to 6 images (no pytest).")
    p.add_argument("--data", "-d", type=Path, default=DEFAULT_DATA, help="Folder with images")
    p.add_argument("--out", "-o", type=Path, default=Path("pipeline_summary.csv"), help="Output CSV path")
    p.add_argument("--limit", "-n", type=int, default=100, help="Max images to process")
    p.add_argument("--cam", type=int, default=1, help="Camera id (MATLAB convention 1/2/3)")
    args = p.parse_args()
    run(args.data, args.out, limit=args.limit, cam_id=args.cam)
