"""
Classification pipeline for MASC snowflake images.

Equivalent of masclab/prediction/predict_snowflakes_class.m,
predict_snowflakes_riming.m, predict_snowflakes_melting.m and
make_predictions_for_campaign.m (Christophe Praz, EPFL).

Pipeline per classifier:
  1. Glob all ROI files in outdir (recursive)
  2. Build feature matrix X  (N × 96)
  3. Select columns with classifier.feat_vec
  4. Transform features according to skewness
  5. Standardize with classifier.mean_ / std_
  6. Predict via logistic softmax (or sigmoid for binary)
  7. Write label_ID / label_name / label_probs back into each ROI file

Labels written back:
  - For class    : roi['label_ID'],     roi['label_name'],     roi['label_probs']
  - For riming   : roi['riming_ID'],    roi['riming_name'],    roi['riming_probs']
  - For melting  : roi['melting_ID'],   roi['melting_name'],   roi['melting_probs']
"""

import logging
import numpy as np

from pathlib import Path
from typing import List, Tuple

import joblib as _joblib

from classification.load_classifier import Classifier, load_classifier
from classification.feature_vector import build_feature_vector
from src.dataio.load_roi_data import load_roi_data, save_roi_data

logger = logging.getLogger(__name__)


def _find_roi_files(outdir: Path, fmt: str) -> List[Path]:
    """Recursively find all ROI files with the right extension."""
    ext_map = {'joblib': '*.joblib',
               'pkl': '*.pkl',
               'mat': '*.mat'}
    pattern = ext_map.get(fmt, '*.mat')
    all_files = sorted(outdir.rglob(pattern))
    # Keep only files whose stem starts with a digit
    roi_files = [f for f in all_files if f.stem[0].isdigit()]
    return roi_files


# ── Inference ─────────────────────────────────────────────────────────────────

def _transform_features(X: np.ndarray, skew: np.ndarray) -> np.ndarray:
    """
    Apply per-feature pre-normalisation transform driven by skewness.
    Matches the MATLAB loop in predict_snowflakes_class.m.
    """
    X = X.copy().astype(float)
    for i, s in enumerate(skew):
        if s > 1.0:
            X[:, i] = np.log(np.abs(X[:, i]) + 1.0)
        elif s > 0.75:
            X[:, i] = np.sqrt(np.abs(X[:, i]))
        elif s < -1.0:
            X[:, i] = np.exp(X[:, i])
        elif s < -0.75:
            X[:, i] = X[:, i] ** 2
    return X


def _predict(X: np.ndarray, cl: Classifier) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference with a logistic classifier.

    Returns:
        pred   : int array (N,) of 1-based class indices (matches MATLAB pred_class)
        scores : float array (N, K) of class probabilities
    """
    if cl.type_ != 'logistic':
        raise NotImplementedError(
            f"Only 'logistic' classifiers are supported; got '{cl.type_}'"
        )

    N = X.shape[0]
    # Prepend bias column  [1 | X]  → (N, D+1)
    tX = np.hstack([np.ones((N, 1)), X])

    if cl.type_classif == 'multiclass':
        # Numerically-stable softmax (equivalent to MATLAB exp(tX*model) / sum)
        logits = tX @ cl.model                                       # (N, K)
        logits -= logits.max(axis=1, keepdims=True)
        exp_l  = np.exp(logits)
        scores = exp_l / exp_l.sum(axis=1, keepdims=True)
        pred   = np.argmax(scores, axis=1) + 1                      # 1-based

    elif cl.type_classif == 'binary':
        # Sigmoid  (MATLAB: round(sigmoid(tX*model)))
        raw    = tX @ cl.model                                       # (N, 1) or (N,)
        scores = 1.0 / (1.0 + np.exp(-raw))
        scores = scores.reshape(N, -1)                               # (N, 1)
        pred   = np.round(scores[:, 0]).astype(int)                  # 0 or 1
        # Match MATLAB 1-based: binary labels are 1 or 2, not 0 or 1
        pred   = pred + 1

    else:
        raise ValueError(f"Unknown type_classif: {cl.type_classif}")

    return pred, scores


# ── Main classification function ──────────────────────────────────────────────

def classify_campaign(
    outdir: Path,
    classifier_path: Path,
    save_format: str = 'mat',
    blurry_threshold: float = 0.0,
    label_prefix: str = 'label',
    n_jobs: int = 1,
) -> None:
    """
    Classify all ROI files in *outdir* using the given classifier and write
    the predicted labels back into each file.

    Equivalent of predict_snowflakes_class.m / predict_snowflakes_riming.m /
    predict_snowflakes_melting.m depending on which classifier is passed.

    Args:
        outdir           : Root output directory (e.g. campaigndir/PROCESSED).
                           Searched recursively for ROI files.
        classifier_path  : Path to the MATLAB .mat classifier file.
        save_format      : File format ('mat', 'joblib', 'pkl').
        blurry_threshold : Minimum xhi value to classify as non-blurry.
                           Images below this threshold get label_ID = -9.
        label_prefix     : Prefix for the saved fields:
                             'label'   → label_ID,  label_name,  label_probs
                             'riming'  → riming_ID, riming_name, riming_probs
                             'melting' → melting_ID, melting_name, melting_probs
        n_jobs           : Number of parallel workers (joblib). 1 = sequential.
    """
    outdir = Path(outdir)

    print(f"  Loading classifier: {classifier_path.name}")
    cl = load_classifier(classifier_path)
    print(f"  Classes: {cl.N_labels}")
    print(f"  Features used: {len(cl.feat_vec)} (out of 96)")

    # ── Find files ────────────────────────────────────────────────────────────
    roi_files = _find_roi_files(outdir, save_format)
    if not roi_files:
        print(f"  No ROI files found in {outdir} (format={save_format})")
        return
    print(f"  Found {len(roi_files)} ROI files")

    # ── Build feature matrix  X_full (N × 96) ────────────────────────────────
    print("  Building feature matrix …", end="", flush=True)

    def _fv(f):
        try:
            return build_feature_vector(load_roi_data(f, save_format))
        except Exception as e:
            logger.warning("Could not extract features from %s: %s", f, e)
            return np.zeros(96)

    if n_jobs == 1:
        rows = [_fv(f) for f in roi_files]
    else:
        rows = _joblib.Parallel(n_jobs=n_jobs)(
            _joblib.delayed(_fv)(f) for f in roi_files
        )

    X_full = np.array(rows, dtype=float)   # (N, 96)
    print(f" done  ({X_full.shape})")

    # ── Select features, transform, standardise ───────────────────────────────
    X = X_full[:, cl.feat_vec]             # (N, D)
    X = _transform_features(X, cl.skew_)

    if cl.normalization == 'standardization':
        denom = np.where(cl.std_ > 0, cl.std_, 1.0)
        X = (X - cl.mean_) / denom
    else:
        print(f"  Warning: normalization type '{cl.normalization}' not implemented; skipping.")

    # ── Predict ───────────────────────────────────────────────────────────────
    print("  Running inference …", end="", flush=True)
    pred, scores = _predict(X, cl)
    print(" done")

    # ── Write labels back into each ROI file ──────────────────────────────────
    print("  Saving labels …", end="", flush=True)
    n_fine = 0
    n_blurry = 0

    id_key    = f'{label_prefix}_ID'
    name_key  = f'{label_prefix}_name'
    probs_key = f'{label_prefix}_probs'

    for i, f in enumerate(roi_files):
        try:
            roi = load_roi_data(f, save_format)
            xhi = float(roi.get('xhi', 0.0)) if isinstance(roi, dict) else 0.0

            if xhi > blurry_threshold:
                roi[id_key]    = int(pred[i])
                roi[name_key]  = cl.N_labels[pred[i] - 1]  # pred is 1-based
                roi[probs_key] = scores[i].ravel()
                n_fine += 1
            else:
                roi[id_key]    = -9
                roi[name_key]  = 'blurry'
                roi[probs_key] = scores[i].ravel()
                n_blurry += 1

            save_roi_data(roi, f, format=save_format, compress=3)
        except Exception as e:
            logger.warning("Could not write labels to %s: %s", f, e)

    print(f" done  ({n_fine} classified, {n_blurry} blurry)")
