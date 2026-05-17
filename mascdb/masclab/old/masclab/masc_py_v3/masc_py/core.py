
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List
import datetime as dt
import re

from .triplets import MASC_triplet_process
from .ids import parse_timestamp_from_name

DOTDATE = re.compile(r'^(\d{4})\.(\d{2})\.(\d{2})$')
HOUR = re.compile(r'^(\d{2})$')

@dataclass
class LabelParams:
    campaigndir: str
    output_dir: str
    starthr_vec: Tuple[int,int,int,int,int,int]
    endhr_vec: Tuple[int,int,int,int,int,int]

@dataclass
class ProcessParams:
    use_triplet_algo: bool = True

def _dirs_in_window(root: Path, start: dt.datetime, end: dt.datetime) -> List[Path]:
    """Return directories to scan. Supports:
    - YYYY/MM/DD (classic)
    - YYYY.MM.DD/HOUR (user's structure)
    We include an hourly dir if its [hour, hour+1) overlaps [start, end].
    """
    dirs: List[Path] = []

    # 1) YYYY/MM/DD — keep compatibility
    for p in root.rglob("*"):
        if not p.is_dir(): continue
        parts = p.parts[-3:]
        try:
            y,m,d = map(int, parts)
            dte = dt.datetime(y,m,d)
            if start.date() <= dte.date() <= end.date():
                dirs.append(p)
        except Exception:
            pass

    # 2) YYYY.MM.DD/HOUR
    for day_dir in root.iterdir():
        if not day_dir.is_dir(): continue
        m = DOTDATE.match(day_dir.name)
        if not m: continue
        y,mn,d = map(int, m.groups())
        for hour_dir in day_dir.iterdir():
            if not hour_dir.is_dir(): continue
            mh = HOUR.match(hour_dir.name)
            if not mh: continue
            h = int(mh.group(1))
            slot_start = dt.datetime(y,mn,d,h,0,0)
            slot_end   = slot_start + dt.timedelta(hours=1)
            # overlap test
            if slot_end > start and slot_start < end:
                dirs.append(hour_dir)

    # Unique & sorted
    uniq = sorted(set(dirs), key=lambda p: str(p))
    return uniq

def _in_window(path: Path, start: dt.datetime, end: dt.datetime) -> bool:
    """Check if any image file in this directory falls into time window using filename timestamp.
    If no file has parsable timestamp, we accept the directory (conservative) if its hour slot overlaps.
    """
    imgs = [p for p in path.rglob("*") if p.suffix.lower() in (".png",".jpg",".jpeg",".tif",".tiff")]
    saw_ts = False
    for p in imgs:
        ts = parse_timestamp_from_name(p.name)
        if ts:
            saw_ts = True
            tsdt = dt.datetime(*ts)
            if start <= tsdt <= end:
                return True
    if not saw_ts:
        # fallback: already overlapped at dir-level
        return True
    return False

def MASC_process(label: LabelParams, process: ProcessParams):
    root = Path(label.campaigndir)
    start = dt.datetime(*label.starthr_vec)
    end   = dt.datetime(*label.endhr_vec)
    candidate_dirs = _dirs_in_window(root, start, end)
    selected = [d for d in candidate_dirs if _in_window(d, start, end)]
    all_results = []
    for d in selected:
        all_results.extend(MASC_triplet_process(d, process))
    return all_results
