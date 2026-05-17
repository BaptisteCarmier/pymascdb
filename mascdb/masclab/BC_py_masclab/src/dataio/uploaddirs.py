from pathlib import Path
from datetime import datetime

def uploaddirs(campaigndir: Path, starthr_vec: datetime, endhr_vec: datetime):
    """
    Upload all MASC directories satisfying the time window defined by starthr_vec and endhr_vec.
    """
    alldir_names = [d for d in campaigndir.iterdir() if d.is_dir()]
    dir_list = []

    # Loop around the folders
    for folder in alldir_names:
        try:
            # Check if folder format is yyyy.mm.dd
            folder_datenum = datetime.strptime(folder.name, '%Y.%m.%d')

            # Use starthr_vec and endhr_vec directly
            t_min = starthr_vec
            t_max = endhr_vec

            if t_min.date() <= folder_datenum.date() <= t_max.date():
                allsubdir_names = [d for d in folder.iterdir() if d.is_dir()]

                for subfolder in allsubdir_names:
                    try:
                        # Parse the hour from the subfolder name
                        hour = int(subfolder.name)
                        sub_folder_datenum = folder_datenum.replace(hour=hour)

                        if t_min <= sub_folder_datenum <= t_max:
                            dir_list.append(subfolder)
                            print(f"{sub_folder_datenum} added to the dir. list")
                    except ValueError:
                        # Skip invalid subfolder names
                        continue
        except ValueError:
            # Skip invalid folder names
            continue

    return dir_list