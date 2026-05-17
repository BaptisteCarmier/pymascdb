"""
Classification runner — calls the MASC hydrometeor classification pipeline.

Equivalent of masclab/prediction/make_predictions_for_campaign.m.

Usage (from BC_py_masclab/ directory):
    python scripts/runner_classify.py

Configuration is read from config/config.yaml.
Processing output (roi .mat/.joblib files) must already exist in
label_config.outdir before running this script.

Three classifiers are run in sequence:
  1. Hydrometeor class    (6 classes)
  2. Degree of riming     (5 classes)
  3. Melting / dry snow   (2 classes)

Labels written into each ROI file:
  label_ID   / label_name   / label_probs
  riming_ID  / riming_name  / riming_probs
  melting_ID / melting_name / melting_probs
"""

import sys
import logging
from pathlib import Path

# Ensure BC_py_masclab/ is on the path (run from that directory)
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.core.config import load_from_yaml

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

try:
    import yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False


def _load_classify_config(yaml_path: Path) -> dict:
    """Read the [classification] section from config.yaml."""
    if not _YAML_OK:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    if 'classification' not in cfg:
        raise KeyError(
            "Missing [classification] section in config.yaml. "
            "Please add classifiers_dir, class, riming, melting keys."
        )
    return cfg['classification']


def main():
    print("=" * 70)
    print("MASC Classification Pipeline")
    print("=" * 70)

    # ── Load main config (for outdir and save_format) ─────────────────────────
    yaml_path = root_dir / 'config' / 'config.yaml'
    label_cfg, proc_cfg = load_from_yaml(yaml_path)

    # ── Load classification config ────────────────────────────────────────────
    cl_cfg = _load_classify_config(yaml_path)

    classifiers_dir = Path(cl_cfg['classifiers_dir'])
    if not classifiers_dir.is_absolute():
        classifiers_dir = (root_dir / classifiers_dir).resolve()

    blurry_threshold = float(cl_cfg.get('blurry_threshold', 0.0))
    n_jobs           = int(cl_cfg.get('n_jobs', 1))
    save_format      = proc_cfg.save_format   # from processing section

    outdir = label_cfg.outdir
    print(f"  Output directory : {outdir}")
    print(f"  Classifiers dir  : {classifiers_dir}")
    print(f"  Save format      : {save_format}")
    print(f"  Blurry threshold : {blurry_threshold}")
    print(f"  Workers          : {n_jobs}")
    print()

    if not outdir.exists():
        print(f"ERROR: outdir does not exist: {outdir}")
        print("Run the processing pipeline first (runner_joblib.py).")
        sys.exit(1)

    # ── Import here (after path setup) ───────────────────────────────────────
    from classification.classify import classify_campaign

    # ── 1. Hydrometeor class (6 classes) ─────────────────────────────────────
    cl_class_path = classifiers_dir / cl_cfg['class']
    print(f"[1/3] Hydrometeor class classifier: {cl_cfg['class']}")
    classify_campaign(
        outdir           = outdir,
        classifier_path  = cl_class_path,
        save_format      = save_format,
        blurry_threshold = blurry_threshold,
        label_prefix     = 'label',
        n_jobs           = n_jobs,
    )
    print()

    # ── 2. Degree of riming (5 classes) ──────────────────────────────────────
    cl_riming_path = classifiers_dir / cl_cfg['riming']
    print(f"[2/3] Riming classifier: {cl_cfg['riming']}")
    classify_campaign(
        outdir           = outdir,
        classifier_path  = cl_riming_path,
        save_format      = save_format,
        blurry_threshold = blurry_threshold,
        label_prefix     = 'riming',
        n_jobs           = n_jobs,
    )
    print()

    # ── 3. Melting / dry snow detection (2 classes) ───────────────────────────
    cl_melting_path = classifiers_dir / cl_cfg['melting']
    print(f"[3/3] Melting classifier: {cl_cfg['melting']}")
    classify_campaign(
        outdir           = outdir,
        classifier_path  = cl_melting_path,
        save_format      = save_format,
        blurry_threshold = blurry_threshold,
        label_prefix     = 'melting',
        n_jobs           = n_jobs,
    )
    print()

    print("=" * 70)
    print("Classification completed.")
    print(f"Labels written into ROI files in: {outdir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
