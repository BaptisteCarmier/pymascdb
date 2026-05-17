
import re

# Strict patterns tailored to filenames like: YYYY.MM.DD_HH.MM.SS_flake_X_cam_Y.png
CAM_PAT = re.compile(r'(?:^|_)cam[_-]?([0-9]+)(?:_|\.|$)', re.IGNORECASE)
FLAKE_PAT = re.compile(r'(?:^|_)(?:flake|flk|snowflake)[_-]?([0-9]+)(?:_|\.|$)', re.IGNORECASE)
TS_PAT = re.compile(r'(\d{4})\.(\d{2})\.(\d{2})_(\d{2})\.(\d{2})\.(\d{2})')

def get_cam_id(s: str):
    m = CAM_PAT.search(s)
    return int(m.group(1)) if m else None

def get_snowflake_id(s: str):
    m = FLAKE_PAT.search(s)
    return int(m.group(1)) if m else None

def parse_timestamp_from_name(s: str):
    m = TS_PAT.search(s)
    if not m: return None
    y,mn,d,h,mi,se = map(int, m.groups())
    return (y,mn,d,h,mi,se)
