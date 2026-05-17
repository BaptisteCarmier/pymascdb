"""

"""

import logging
import shutil

from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


def clean_output_dir(outdir: Path) -> None:
    """Remove all contents of the output directory (best effort) before a run."""
    if not outdir.exists():
        return
    for child in outdir.iterdir():
        try:
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
        except Exception:
            logger.warning("Could not remove '%s' while cleaning %s", child, outdir)
