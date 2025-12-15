"""
Upload MASC fallspeeds and image filenames found in dirname.

This function reads in the text files found in dirname and outputs a list
of all images found in pic.files, each with a timestamp pic.time_vec, and
a picture id pic.id, and a camera id pic.cam, and a fallspeed and
fallspeed id pic.fallspeed and pic.fallid.

If the fallspeed information is not available, the function returns a
vector of NaN.

Translated from upload.m
Author: Christophe Praz (christophe.praz@epfl.ch)
Last update: November 2025 (Python)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


@dataclass
class PictureList:
    """Container for picture metadata extracted from imgInfo.txt and dataInfo.txt."""

    time_vec: List[List[int]] = field(default_factory=list)  # [[year, month, day, hour, min, sec], ...]
    id: List[int] = field(default_factory=list)
    cam: List[int] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    time_str: List[str] = field(default_factory=list)
    time_num: List[float] = field(default_factory=list)
    fallspeed: List[float] = field(default_factory=list)
    fallid: List[int] = field(default_factory=list)
    id_unique: List[int] = field(default_factory=list)  # Unique picture IDs (for triplet processing)


def upload(dirname: Path) -> PictureList:
    """
    Upload MASC fallspeeds and image filenames from directory.

    Reads imgInfo.txt and dataInfo.txt from dirname and returns a PictureList
    containing all metadata.

    Args:
        dirname: Path to directory containing imgInfo.txt and dataInfo.txt

    Returns
    -------
        PictureList object with fields:
            - files: list of image filenames
            - id: list of picture IDs
            - cam: list of camera IDs (0, 1, 2)
            - time_vec: list of [year, month, day, hour, min, sec] vectors
            - time_str: list of formatted datetime strings
            - time_num: list of datetime objects as float timestamps
            - fallspeed: list of fallspeed values (m/s)
            - fallid: list of fallspeed IDs (matching to pic IDs)

    MATLAB equivalent: upload(dirname)

    Example imgInfo.txt format:
        ID  Cam  Date        Time      Filename
        1   0    06.20.2015  09:00:01  image_001_0.png
        1   1    06.20.2015  09:00:01  image_001_1.png
        1   2    06.20.2015  09:00:01  image_001_2.png

    Example dataInfo.txt format:
        ID  Fallspeed
        1   1.23
        2   0.98
    """
    dirname = Path(dirname)
    pic = PictureList()

    # ===== Import imagelist and time from imgInfo.txt =====

    imginfo_path = dirname / "imgInfo.txt"

    try:
        with open(imginfo_path) as f:
            lines = f.readlines()

        # Skip header line(s) - assume first line is header
        nheaderlines = 0
        data_lines = lines[nheaderlines:]

        for line in data_lines:
            line = line.strip()
            if not line:
                continue

            # Split by whitespace (tabs or spaces)
            parts = line.split()

            if len(parts) < 5:
                continue

            # Parse fields: ID  Cam  Date  Time  Filename
            pic_id = int(parts[0])
            pic_cam = int(parts[1])
            day = parts[2]  # e.g., "06.20.2015"
            timestamp = parts[3]  # e.g., "09:00:01"
            filename = parts[4]  # e.g., "image_001_0.png"
            # il y'a un 0 en plus mais aucune idée de à quoi il correspond

            # Parse date: MM.DD.YYYY
            d_parts = day.split(".")
            picmo = int(d_parts[0])
            picdd = int(d_parts[1])
            picyr = int(d_parts[2])

            # Parse time: HH:MM:SS
            t_parts = timestamp.split(":")
            pichh = int(t_parts[0])
            picmm = int(t_parts[1])
            picss = int(float(t_parts[2]))  # Handle fractional seconds

            # Store data
            pic.id.append(pic_id)
            pic.cam.append(pic_cam)
            pic.files.append(filename)
            pic.time_vec.append([picyr, picmo, picdd, pichh, picmm, picss])

    except Exception as e:
        logger.error("ERROR : unable to open imgInfo.txt in %s", dirname)
        logger.error("Details: %s", e)
        return pic

    # Convert time_vec to datetime strings and numerical timestamps
    pic.time_str = []
    pic.time_num = []

    for tv in pic.time_vec:
        dt = datetime(tv[0], tv[1], tv[2], tv[3], tv[4], tv[5])
        pic.time_str.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
        # Convert to timestamp (seconds since epoch) for MATLAB datenum compatibility
        pic.time_num.append(dt.timestamp())

    # ===== Import fallspeed data from dataInfo.txt =====

    datainfo_path = dirname / "dataInfo.txt"

    try:
        with open(datainfo_path) as f:
            lines = f.readlines()

        # Skip header line(s) if exists
        nheaderlines = 0  # No header in this format
        data_lines = lines[nheaderlines:]

        for line in data_lines:
            line = line.strip()
            if not line:
                continue

            # Split by whitespace
            parts = line.split()

            if len(parts) < 4:
                continue

            # Parse fields: ID  Date  Time  Fallspeed
            fallid = int(parts[0])
            # parts[1] is date, parts[2] is time - we can ignore these
            fallspeed = float(parts[3])

            pic.fallid.append(fallid)
            pic.fallspeed.append(fallspeed)

    except Exception as e:
        # If dataInfo.txt is missing, fill with NaN
        pic.fallid = sorted(set(pic.id))  # unique IDs
        pic.fallspeed = [np.nan] * len(pic.fallid)
        logger.error("ERROR : unable to open dataInfo.txt in %s", dirname)
        logger.error("Details: %s", e)

    # ===== Compute unique picture IDs (for triplet processing) =====
    pic.id_unique = sorted(set(pic.id))

    return pic
