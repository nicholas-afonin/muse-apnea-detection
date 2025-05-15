import os

running_locally = os.name == 'nt'
# Can use this for local testing with small subsets of the actual dataset before submitting
# to cedar or niagara with full datasets.

class path:
    if running_locally:
        raw_csv_directory = r"C:\Users\aweso\Desktop\Sunnybrook\MUSE\Data\MUSE-PSG/"
        synced_csv_directory = r"C:\Users\aweso\Desktop\Sunnybrook\MUSE\Data\muse_synced/"
        ACC_features_directory = r"C:\Users\aweso\Desktop\Sunnybrook\MUSE\Data\ACC_features/"
        EEG_features_directory = r"C:\Users\aweso\Desktop\Sunnybrook\MUSE\Data\EEG_features/"
        combined_features = r"C:\Users\aweso\Desktop\Sunnybrook\MUSE\Data\ACC_EEG_features"

    else:
        raw_csv_directory = "/scratch/a/alim/afoninni/muse/MUSE-PSG/"
        synced_csv_directory = "/scratch/a/alim/afoninni/muse/muse_synced/"
        ACC_features_directory = "/scratch/a/alim/afoninni/muse/ACC_features/"
        EEG_features_directory = "/scratch/a/alim/afoninni/muse/EEG_features/"
        combined_features = "/scratch/a/alim/afoninni/muse/ACC_EEG_features/"
