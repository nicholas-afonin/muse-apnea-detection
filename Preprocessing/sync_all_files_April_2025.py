import os
import pandas as pd
import config
from datetime import datetime


raw_csv_directory = config.path.raw_csv_directory  # Directory where your raw CSV files are located
synced_csv_directory = config.path.synced_csv_directory  # Directory where you want to save the filtered CSV files

# Check if the directory exists
if not os.path.exists(synced_csv_directory):
    # If it doesn't exist, create it
    os.makedirs(synced_csv_directory)

# List all CSV files in the raw directory
raw_csv_files = sorted([f for f in os.listdir(config.path.raw_csv_directory) if f.endswith('.csv')], reverse=True)

# Initialize the total_files and processed_files counters
total_files = len([f for f in raw_csv_files if f.endswith('.csv')])
processed_files = 0

# Initialize dictionaries to store common start and end times for each group
common_start_times = {}
common_end_times = {}

def all_files_present(prefix, files_list):
    """Check if all file types are present for a given prefix."""
    required_suffixes = ['_acc', '_eeg', '_ppg', '_staging2']
    return all(any(file.startswith(prefix) and file.endswith(suffix + '.csv') for file in files_list) for suffix in required_suffixes)

# Loop through the raw CSV files and immediately skip if synced file exists
for raw_file in raw_csv_files:
    if raw_file.endswith('.csv'):
        # Determine the data type based on the file name
        if "_acc" in raw_file:
            data_type = "acc"
        elif "_eeg" in raw_file:
            data_type = "eeg"
        elif "_ppg" in raw_file:
            data_type = "ppg"
        elif "_staging2" in raw_file:
            data_type = "staging2"
        else:
            continue

        common_prefix = raw_file.split("_" + data_type)[0]

        # Check if all file types are present for the given prefix
        if not all_files_present(common_prefix, raw_csv_files):
            print(f"All file types not present for prefix {common_prefix}. Skipping...")
            continue

        # Check if the synced version of the file exists in the output directory
        synced_file_path = os.path.join(synced_csv_directory, f'{common_prefix}_synced_{data_type}.csv')
        if os.path.exists(synced_file_path):
            print(f"Synced file {synced_file_path} already exists. Skipping...")
            processed_files += 1
            print(f"Processed {processed_files} out of {total_files} files.")
            continue
        
        # Print the name of the file being processed
        print(f"Processing file: {raw_file}")
        processed_files += 1
        print(f"Processed {processed_files} out of {total_files} files.")

        # Extract the common prefix before the data type
        common_prefix = raw_file.split("_" + data_type)[0]

        # Read the CSV file into a DataFrame
        df = pd.read_csv(os.path.join(raw_csv_directory, raw_file))

        # Adjust for the timestamp column based on the data type
        timestamp_col = 'start' if data_type == 'staging2' else 'ts'

        # Convert the timestamp column to datetime objects with second precision
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='s').dt.floor('s')

        # Calculate the common start and end times based on the data in this file
        file_start_time = df[timestamp_col].min()
        file_end_time = df[timestamp_col].max()

        # Update the common_start_times and common_end_times dictionaries
        if common_prefix in common_start_times:
            common_start_times[common_prefix] = max(common_start_times[common_prefix], file_start_time)
            common_end_times[common_prefix] = min(common_end_times[common_prefix], file_end_time)
        else:
            common_start_times[common_prefix] = file_start_time
            common_end_times[common_prefix] = file_end_time

    # Loop through the groups and process/sync the files accordingly
    for common_prefix in common_start_times:
        common_start_time = common_start_times[common_prefix]
        common_end_time = common_end_times[common_prefix]

        for raw_file in raw_csv_files:
            if raw_file.startswith(common_prefix):
                # Determine the data type based on the file name
                if "_acc" in raw_file:
                    data_type = "acc"
                elif "_eeg" in raw_file:
                    data_type = "eeg"
                elif "_ppg" in raw_file:
                    data_type = "ppg"
                elif "_staging2" in raw_file:
                    data_type = "staging2"
                else:
                    continue

                # Check if the synced version of the file exists in the output directory
                synced_file_path = os.path.join(synced_csv_directory, f'{common_prefix}_synced_{data_type}.csv')
                if os.path.exists(synced_file_path):
                    print(f"Synced file {synced_file_path} already exists. Skipping...")
                    continue

                # Read the CSV file into a DataFrame
                df = pd.read_csv(os.path.join(raw_csv_directory, raw_file))

                # Adjust for the timestamp column based on the data type
                timestamp_col = 'start' if data_type == 'staging2' else 'ts'

                # Convert the timestamp column to datetime objects with second precision
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='s').dt.floor('s')

                # Filter the data based on the common time range
                df = df[(df[timestamp_col] >= common_start_time) & (df[timestamp_col] <= common_end_time)]

                # Convert the timestamp column back to Unix timestamps
                df[timestamp_col] = df[timestamp_col].astype('int64') // 10**9

                # Save the filtered DataFrame to a new CSV file with the same name and data type
                df.to_csv(os.path.join(synced_csv_directory, f'{common_prefix}_synced_{data_type}.csv'), index=False)
                print(f"Saved synced file: {common_prefix}_synced_{data_type}.csv")
