
from pathlib import Path
from typing import Dict, List, Tuple
import re
from .picture import MASC_picture_process
from .ids import get_cam_id, get_snowflake_id

def MASC_triplet_process(container_dir: Path, process):
    imgs = [p for p in container_dir.rglob("*") if p.suffix.lower() in (".png",".jpg",".jpeg",".tif",".tiff")]
    by_flake: Dict[int, Dict[int, Path]] = {}
    for p in imgs:
        cam = get_cam_id(p.name)
        flk = get_snowflake_id(p.name)
        if flk is None or cam is None:
            # skip files that don't encode both ids
            continue
        by_flake.setdefault(flk, {})
        # keep one per cam (first seen)
        by_flake[flk].setdefault(cam, p)

    results = []
    for flk, per_cam in by_flake.items():
        for cam_id, path in per_cam.items():
            results.append(MASC_picture_process(path, process))
    return results
