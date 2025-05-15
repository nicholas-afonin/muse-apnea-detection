# Combines EEG features, ACC features, and ground truth labels (note feature files already contain
# sleep stage labels under 'name' column)
# Assign all epochs in data with labels according to the relevant ground truth events
# i.e. add a column denoting the relevant events taking place in each epoch/window

import config
import os
import glob
import pandas as pd


APNEA_EPOCH_THRESHOLD = 0.01


def path_to_file(ACC_path, EEG_path, events_path):

    # Get all CSV files with ACC features, EEG features, and events
    acc_files = sorted(glob.glob(ACC_path + '*_acc_features.csv'))  # , reverse=True)
    eeg_files = sorted(glob.glob(EEG_path + '*_eeg_features.csv'))  # , reverse=True)
    event_files = sorted(glob.glob(events_path + '*_events.csv'))  # , reverse=True)

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


def label_apnea_events(df_features, df_events, window_length=30, min_overlap_fraction = 0.01):
    """
    Labels each row in the features table according to the events file (ex. apnea or no apnea)
    """

    # Labels for the final table, and which 'input labels' are mapped to them
    # in this case, all apnea-related events are simply mapped to the 'apnea_event' column
    event_map = {'apnea_event': ['Hypopnea', 'Obstructive Apnea', 'Mixed Apnea', 'Central Apnea']}

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



def combine_features_and_label(ACC_feature_files, EEG_feature_files, event_label_files, output_directory):
    for acc_file, eeg_file, event_file in zip(ACC_feature_files, EEG_feature_files, event_label_files):
        # Initialize all data for recording
        df_acc = pd.read_csv(acc_file)
        df_eeg = pd.read_csv(eeg_file)
        df_events = pd.read_csv(event_file)

        df_acc = df_acc.drop(columns=['name', 'ts'])  # drop sleep stage and time column, since eeg has one already
                                                      # and they line up

        print(f"Now processing > {acc_file.split('/')[-1]}: {len(df_acc)} rows/epochs, with {len(df_events)} events")

        # Combine acc and eeg features
        df_acc_eeg = pd.concat([df_acc, df_eeg], axis=1)

        # Filter out wake rows
        df_acc_eeg = remove_wake_rows(df_acc_eeg)

        # Add columns labelling apnea events (or other events)
        df_acc_eeg = label_apnea_events(df_acc_eeg, df_events)

        # save file to output location
        filename = os.path.basename(acc_file).replace('_synced_acc_features.csv',
                                               '_combined_features.csv')
        full_out = os.path.join(output_directory, filename)
        df_acc_eeg.to_csv(full_out, index=False)


def just_label(ACC_EEG_feature_files, event_label_files):
    pass


if __name__ == '__main__':
    # Check if the directory exists
    if not os.path.exists(config.path.combined_features):
        # If it doesn't exist, create it
        os.makedirs(config.path.combined_features)

    acc_files, eeg_files, event_files, prefixes = path_to_file(config.path.ACC_features_directory, config.path.EEG_features_directory, config.path.raw_csv_directory)
    combine_features_and_label(acc_files, eeg_files, event_files, config.path.combined_features)