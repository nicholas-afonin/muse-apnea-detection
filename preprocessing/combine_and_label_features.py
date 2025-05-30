# Combines EEG features, ACC features, and ground truth labels (note feature files already contain
# sleep stage labels under 'name' column)
# Assign all epochs in data with labels according to the relevant ground truth events
# i.e. add a column denoting the relevant events taking place in each epoch/window

import config
import os
import glob
import pandas as pd


def check_files(ACC_path, EEG_path, events_path):
    """This func basically only used when you're trying to combine ACC and EEG features along with
    events like apneas. Checks to ensure the shapes of all the directories you're combining match,
    and simply returns the strings denoting those directories if yes"""

    # Get all CSV files with ACC features, EEG features, and events
    acc_files = sorted(glob.glob(ACC_path + '*_acc_features.csv'))
    eeg_files = sorted(glob.glob(EEG_path + '*_eeg_features.csv'))
    event_files = sorted(glob.glob(events_path + '*_events.csv'))

    # Extract recording IDs
    acc_prefixes = [os.path.basename(f).split('_synced_acc_features.csv')[0] for f in acc_files]
    eeg_prefixes = [os.path.basename(f).split('_synced_eeg_features.csv')[0] for f in eeg_files]
    event_prefixes = [os.path.basename(f).split('_events.csv')[0] for f in event_files]

    # Ensure that for each acc file, there is an eeg file and event file that match the ID
    if len(acc_files) != len(eeg_files) != len(event_files) or acc_prefixes != eeg_prefixes != event_prefixes:
        raise ValueError("Mismatch in the number acc, eeg and event files, or their prefixes")

    return acc_files, eeg_files, event_files, acc_prefixes


def remove_wake_rows(df):
    """
    Removes all the rows in the dataframe where the 'name' column (sleep stage) indicates wake or unknown
    """
    df = df[df['name'] != 0]  # keeps all rows that are not wake
    df = df[df['name'] != 9]  # keeps all rows that are not unknown

    return df


def label_apnea_events(df_features, df_events, min_overlap_fraction, window_length=30):
    """
    Labels each row in the features table according to the events file (ex. apnea or no apnea)
    """

    # Labels for the final table, and which 'input labels' are mapped to them
    # in this case, all apnea-related events are simply mapped to the 'apnea_event' column
    event_map = {'apnea_event': ['Hypopnea', 'Obstructive Apnea', 'Mixed Apnea', 'Central Apnea'],
                 'arousal_event': ['RERA', 'Arousal (ARO RES)', 'Arousal (ARO SPONT)']}

    # Calculate event end-times (from start times and durations)
    df_events = df_events.copy()
    df_events["end"] = df_events["start"] + df_events["duration"]

    # Determine minimum threshold for overlap (if the event takes up more than x seconds
    # of an epoch, label with 1)
    min_overlap_sec = min_overlap_fraction * window_length

    # Prepare output column(s)
    df_features = df_features.copy()
    for column in event_map:
        df_features[column] = 0  # initialize new column with 0s
    df_features.sort_values("ts", inplace=True, ignore_index=True)  # ensure rows are in chron. order

    # Main loop to check overlap between events and epochs
    for i, row in df_features.iterrows():
        window_start = row['ts']
        window_end = window_start + window_length

        # select all the events that potentially overlap
        mask = (df_events["start"] < window_end) & (df_events["end"] > window_start)
        overlapping_events = df_events[mask]

        if overlapping_events.empty:
            continue

        # For each new column we hope to add
        for column in event_map:
            # for each overlapping event
            for _, event in overlapping_events.iterrows():
                # check if the overlapping event is one we care about
                if event["name"] in event_map[column]:
                    # calculate the duration of overlap
                    overlap_sec = min(event["end"], window_end) - max(event["start"], window_start)
                    if overlap_sec >= min_overlap_sec:
                        df_features.at[i, column] = 1
                        break

    return df_features


def combine_features(ACC_feature_files, EEG_feature_files, output_directory):
    for acc_file, eeg_file in zip(ACC_feature_files, EEG_feature_files):
        # Initialize all data for recording
        df_acc = pd.read_csv(acc_file)
        df_eeg = pd.read_csv(eeg_file)

        # drop sleep stage and time column (which already exist in EEG files)
        df_acc = df_acc.drop(columns=['name', 'ts'])

        print(f"Now combining > {acc_file.split('/')[-1]}: {len(df_acc)} rows")

        # Combine acc and eeg features
        df_acc_eeg = pd.concat([df_acc, df_eeg], axis=1)

        # Save output
        filename = os.path.basename(acc_file).replace('_synced_acc_features.csv',
                                                      '_synced_features.csv')
        full_out = os.path.join(output_directory, filename)
        df_acc_eeg.to_csv(full_out, index=False)


def process_features(ACC_EEG_feature_files: [str], event_label_files: [str], output_directory: str, apnea_epoch_threshold: float, arousal_epoch_threshold: float, window_length: int=30):
    output_directory = (output_directory + 'EEG_ACC_features_labelled_ApneaThreshold' + str(apnea_epoch_threshold) +
                        '_ArousalThreshold' + str(arousal_epoch_threshold) + '_window' + str(window_length) + '/')
    os.makedirs(output_directory, exist_ok=True)

    for acc_eeg_file, event_file in zip(ACC_EEG_feature_files, event_label_files):
        # Initialize all data for recording
        df_acc_eeg = pd.read_csv(acc_eeg_file)
        df_events = pd.read_csv(event_file)

        # drop extra or unneeded rows (dependent on the ACC_EEG_feature_files)
        # if generated using the function in this file, nothing needs to be removed
        # df_acc_eeg = df_acc_eeg.drop(columns=['name_y'])

        print(f"Now processing > {acc_eeg_file.split('/')[-1]}: {len(df_acc_eeg)} rows, with {len(df_events)} events")

        # Filter out wake rows
        df_acc_eeg = remove_wake_rows(df_acc_eeg)

        # Add columns labelling apnea events (or other events)
        df_acc_eeg = label_apnea_events(df_acc_eeg, df_events, min_overlap_fraction=apnea_epoch_threshold)

        # save file to output location
        filename = os.path.basename(acc_eeg_file).replace('_synced_features.csv',
                                                      '_synced_features.csv')
        full_out = os.path.join(output_directory, filename)
        df_acc_eeg.to_csv(full_out, index=False)


if __name__ == '__main__':
    """OPTION 1 - combine features"""
    window_size = 10

    acc_features_directory = os.path.join(config.path.ACC_features_directory, f'ACC_features_window{window_size}_strideNA/')
    eeg_features_directory = os.path.join(config.path.EEG_features_directory, f'EEG_features_window{window_size}_strideNA/')
    acc_files = sorted(glob.glob(acc_features_directory + '*_acc_features.csv'))
    eeg_files = sorted(glob.glob(eeg_features_directory + '*_eeg_features.csv'))

    output_folder = os.path.join(config.path.EEG_ACC_features, f"{window_size}s_windows/")
    os.makedirs(output_folder, exist_ok=True)

    combine_features(acc_files, eeg_files, output_folder)

    """OPTION 2 - only process already combined features"""
    features_path = os.path.join(config.path.EEG_ACC_features, "30s_windows/")
    eeg_acc_files = sorted(glob.glob(features_path + '*_synced_features.csv'))
    event_files = sorted(glob.glob(config.path.raw_csv_directory + '*_events.csv'))

    output_folder = config.path.EEG_ACC_features_labelled
    os.makedirs(output_folder, exist_ok=True)

    process_features(eeg_acc_files, event_files, output_folder, 0.30, 0.30, window_length=1)