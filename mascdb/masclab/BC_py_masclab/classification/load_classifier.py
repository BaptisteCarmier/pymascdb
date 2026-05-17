"""
Load and parse MATLAB logistic regression classifier .mat files.

Translated from predict_snowflakes_class.m / predictClassifier.m (Christophe Praz, EPFL).
"""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from scipy.io import loadmat


@dataclass
class Classifier:
    """Parsed MATLAB classifier struct."""
    model: np.ndarray           # Weight matrix (D+1, K) for logistic
    feat_vec: np.ndarray        # 0-based feature indices into the 96-element vector
    mean_: np.ndarray           # Normalization mean (per feature, after feat_vec selection)
    std_: np.ndarray            # Normalization std  (per feature, after feat_vec selection)
    skew_: np.ndarray           # Skewness (drives pre-transform choice)
    normalization: str          # 'standardization' (only supported type)
    type_: str                  # 'logistic' (only supported type)
    type_classif: str           # 'multiclass' or 'binary'
    N_labels: List[str]         # Ordered list of class-name strings


def load_classifier(path) -> Classifier:
    """
    Load a MATLAB logistic regression classifier saved as a .mat file.

    The .mat file must contain a top-level variable named 'classifier' with
    the fields expected by predict_snowflakes_class.m:
        classifier.model
        classifier.normalization        ('standardization')
        classifier.normalization_params.mean / .std / .skew
        classifier.feat_vec             (1-based column indices)
        classifier.N_labels             (cell array of class name strings)
        classifier.type                 ('logistic')
        classifier.type_classif         ('multiclass' | 'binary')

    Args:
        path: Path to the .mat classifier file.

    Returns:
        Classifier dataclass with all fields parsed and ready for inference.
    """
    data = loadmat(str(path), squeeze_me=True, struct_as_record=False)

    if 'classifier' not in data:
        raise KeyError(
            f"Expected variable 'classifier' in {path}; "
            f"found: {[k for k in data if not k.startswith('_')]}"
        )

    cl = data['classifier']

    # ── feat_vec: 1-based indices in MATLAB → 0-based in Python ──────────────
    feat_vec_raw = np.atleast_1d(np.array(cl.feat_vec, dtype=float))
    feat_vec = np.round(feat_vec_raw - 1).astype(int)

    # ── normalization_params ──────────────────────────────────────────────────
    norm = cl.normalization_params
    mean_ = np.atleast_1d(np.array(norm.mean, dtype=float)).ravel()
    std_  = np.atleast_1d(np.array(norm.std,  dtype=float)).ravel()
    skew_ = np.atleast_1d(np.array(norm.skew, dtype=float)).ravel()

    # ── N_labels: MATLAB cell array of strings ────────────────────────────────
    n_labels = cl.N_labels
    if isinstance(n_labels, np.ndarray):
        labels = [str(s).strip() for s in n_labels.ravel()]
    elif isinstance(n_labels, str):
        labels = [n_labels.strip()]
    else:
        labels = [str(s).strip() for s in n_labels]

    return Classifier(
        model        = np.array(cl.model, dtype=float),
        feat_vec     = feat_vec,
        mean_        = mean_,
        std_         = std_,
        skew_        = skew_,
        normalization= str(cl.normalization).strip(),
        type_        = str(cl.type).strip(),
        type_classif = str(cl.type_classif).strip(),
        N_labels     = labels,
    )
