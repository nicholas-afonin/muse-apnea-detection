"""
Configuration for settings and paths that are used by virtually every file.
Use as follows:

import config
if config.running_locally:
    print("code is running locally")

print("the raw data is currently found in:", config.raw_csv_directory)
"""

import os


testing_on_macbook = False


running_locally = os.name == 'nt'
# Can use this for local testing with small subsets of the actual dataset before submitting
# to cedar or niagara with full datasets. os.name == 'nt' means running on windows.


class path:
    if testing_on_macbook:
        raw_csv_directory = None
        synced_csv_directory = None
        ACC_features_directory = None
        EEG_features_directory = None
        EEG_ACC_features = None
        EEG_ACC_features_labelled = "/Users/nicholasafonin/Desktop/Sunnybrook/EEG_ACC_features_labelled_April2025/"

    elif running_locally:
        raw_csv_directory = r"C:\Users\aweso\Desktop\Sunnybrook\MUSE\Data\MUSE-PSG/"
        synced_csv_directory = r"C:\Users\aweso\Desktop\Sunnybrook\MUSE\Data\muse_synced/"
        ACC_features_directory = r"C:\Users\aweso\Desktop\Sunnybrook\MUSE\Data\ACC_features/"
        EEG_features_directory = r"C:\Users\aweso\Desktop\Sunnybrook\MUSE\Data\EEG_features/"
        EEG_ACC_features = r"C:\Users\aweso\Desktop\Sunnybrook\MUSE\Data\EEG_ACC_features_June2025/"
        EEG_ACC_features_labelled = r"C:\Users\aweso\Desktop\Sunnybrook\MUSE\Data\EEG_ACC_features_labelled/"

    else:
        raw_csv_directory = "/scratch/a/alim/afoninni/muse/MUSE-PSG/"
        synced_csv_directory = "/scratch/a/alim/afoninni/muse/muse_synced/"
        ACC_features_directory = "/scratch/a/alim/afoninni/muse/ACC_features/"
        EEG_features_directory = "/scratch/a/alim/afoninni/muse/EEG_features/"
        EEG_ACC_features = "/scratch/a/alim/afoninni/muse/EEG_ACC_features_June2025/"
        EEG_ACC_features_labelled = "/scratch/a/alim/afoninni/muse/EEG_ACC_features_labelled_June2025/"