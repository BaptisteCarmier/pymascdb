
# masc_py_v3 — processing with dotted date/hour structure support

Supports both:
- `YYYY/MM/DD/...`
- `YYYY.MM.DD/HOUR/` with filenames like `YYYY.MM.DD_HH.MM.SS_flake_X_cam_Y.png`

The time window is applied using the timestamp parsed from filenames when present,
and otherwise by the hour directory overlap.

Install deps:
  pip install numpy pillow scikit-image scipy PyWavelets
