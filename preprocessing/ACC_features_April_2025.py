"""
This file handles accelerometer feature extraction from synced MUSE sensor data. Can select window size and
stride distance if a sliding window (overlapping windows) is preferred.
Currently, sliding windows are not implemented.
"""

import pandas as pd
import glob
import numpy as np
import os
import config
from scipy.fft import fft
from joblib import Parallel, delayed
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor


if config.running_locally:
    CPU_CORES_AVAILABLE = 10
else:
    CPU_CORES_AVAILABLE = int(os.getenv("SLURM_CPUS_PER_TASK", os.cpu_count()))
    print(f"{CPU_CORES_AVAILABLE} CPU cores available")


def compute_features(df, df_label_time, threshold=0.0, cutoff_freq=0.5, window_size=30):
    # Check if required columns are present in the input dataframe
    required_columns = ['ts', 'ch1', 'ch2', 'ch3']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")

    # Check if 'start' and 'name' columns are present in df_label_time
    if 'start' not in df_label_time.columns or 'name' not in df_label_time.columns:
        raise ValueError(f"Columns 'start' or 'name' not found in the df_label_time DataFrame.")

    feature_list = []

    # Use the 'start' column from df_label_time to determine the chunks
    start_times = df_label_time['start'].tolist()
    names = df_label_time['name'].tolist()

    for i in range(len(start_times) - 1):
        chunk = df[(df['ts'] >= start_times[i]) & (df['ts'] < (start_times[i] + window_size))]

        # Check if the chunk is empty
        if chunk.empty:
            print(f"Warning: No data found for start time {start_times[i]} to {start_times[i + 1]}. Skipping...")
            continue

        # Create a copy of the chunk to avoid SettingWithCopyWarning
        chunk = chunk.copy()
        # Compute ch123 = sqrt(ch1^2 + ch2^2 + ch3^2)
        chunk.loc[:, 'ch123'] = np.sqrt(chunk['ch1']**2 + chunk['ch2']**2 + chunk['ch3']**2)

        # Compute Zero Crossing Rate (ZCR) for ch1, ch2, ch3, and ch123
        zcr_ch1 = np.mean(np.diff(np.sign(chunk['ch1'])) != 0)
        zcr_ch2 = np.mean(np.diff(np.sign(chunk['ch2'])) != 0)
        zcr_ch3 = np.mean(np.diff(np.sign(chunk['ch3'])) != 0)
        zcr_ch123 = np.mean(np.diff(np.sign(chunk['ch123'])) != 0)

        # Compute Total Above Threshold (TAT) for ch1, ch2, ch3, and ch123
        tat_ch1 = np.sum(chunk['ch1'] > threshold)
        tat_ch2 = np.sum(chunk['ch2'] > threshold)
        tat_ch3 = np.sum(chunk['ch3'] > threshold)
        tat_ch123 = np.sum(chunk['ch123'] > threshold)

        # Compute Peak-to-Instantaneous-Mean (PIM) for ch1, ch2, ch3, and ch123
        pim_ch1 = np.max(chunk['ch1']) - np.mean(chunk['ch1'])
        pim_ch2 = np.max(chunk['ch2']) - np.mean(chunk['ch2'])
        pim_ch3 = np.max(chunk['ch3']) - np.mean(chunk['ch3'])
        pim_ch123 = np.max(chunk['ch123']) - np.mean(chunk['ch123'])

        # Compute correlation coefficients between axes
        corr_xy = np.corrcoef(chunk['ch1'], chunk['ch2'])[0, 1]
        corr_xz = np.corrcoef(chunk['ch1'], chunk['ch3'])[0, 1]
        corr_yz = np.corrcoef(chunk['ch2'], chunk['ch3'])[0, 1]

        # Compute dominant frequency and magnitude for each channel
        def get_dominant_freq(data, fs=50, cutoff=cutoff_freq):
            # Compute FFT
            n = len(data)
            yf = fft(data)
            xf = np.linspace(0.0, fs/2, n//2)
            
            # Get magnitudes
            magnitudes = 2.0/n * np.abs(yf[:n//2])
            
            # Filter frequencies above cutoff
            mask = xf >= cutoff
            xf_filtered = xf[mask]
            magnitudes_filtered = magnitudes[mask]
            
            if len(xf_filtered) == 0:
                return np.nan, np.nan
            
            # Find dominant frequency and its magnitude
            idx = np.argmax(magnitudes_filtered)
            dominant_freq = xf_filtered[idx]
            dominant_mag = magnitudes_filtered[idx]
            
            return dominant_freq, dominant_mag

        # Compute for each channel
        dom_freq_ch1, dom_mag_ch1 = get_dominant_freq(chunk['ch1'].values)
        dom_freq_ch2, dom_mag_ch2 = get_dominant_freq(chunk['ch2'].values)
        dom_freq_ch3, dom_mag_ch3 = get_dominant_freq(chunk['ch3'].values)
        dom_freq_ch123, dom_mag_ch123 = get_dominant_freq(chunk['ch123'].values)

        # Compute features for ch1, ch2, ch3, and ch123
        features = {
            'ts': chunk['ts'].iloc[0],
            'mean_ch1': chunk['ch1'].mean(),
            'mean_ch2': chunk['ch2'].mean(),
            'mean_ch3': chunk['ch3'].mean(),
            'mean_ch123': chunk['ch123'].mean(),
            'std_ch1': chunk['ch1'].std(),
            'std_ch2': chunk['ch2'].std(),
            'std_ch3': chunk['ch3'].std(),
            'std_ch123': chunk['ch123'].std(),
            'var_ch1': chunk['ch1'].var(),
            'var_ch2': chunk['ch2'].var(),
            'var_ch3': chunk['ch3'].var(),
            'var_ch123': chunk['ch123'].var(),
            'zcr_ch1': zcr_ch1,
            'zcr_ch2': zcr_ch2,
            'zcr_ch3': zcr_ch3,
            'zcr_ch123': zcr_ch123,
            'tat_ch1': tat_ch1,
            'tat_ch2': tat_ch2,
            'tat_ch3': tat_ch3,
            'tat_ch123': tat_ch123,
            'pim_ch1': pim_ch1,
            'pim_ch2': pim_ch2,
            'pim_ch3': pim_ch3,
            'pim_ch123': pim_ch123,
            'corr_xy': corr_xy,
            'corr_xz': corr_xz,
            'corr_yz': corr_yz,
            'dom_freq_ch1': dom_freq_ch1,
            'dom_mag_ch1': dom_mag_ch1,
            'dom_freq_ch2': dom_freq_ch2,
            'dom_mag_ch2': dom_mag_ch2,
            'dom_freq_ch3': dom_freq_ch3,
            'dom_mag_ch3': dom_mag_ch3,
            'dom_freq_ch123': dom_freq_ch123,
            'dom_mag_ch123': dom_mag_ch123,
            'name': names[i]
        }

        feature_list.append(features)

    # Convert the list of dictionaries into a dataframe
    features_df = pd.DataFrame(feature_list)

    return features_df


def resample_sleep_staging(df, new_window_size, stride):
    """
    Currently the feature extraction is based on the time windows provided by each
    _staging2.csv file, which has 30 second time windows. This function simply resamples it
    to whatever desired window size, keeping the sleep labels assigned at any given point in time.

    Recall that the times in the "starts" column refer to the start of the sleep windows.
    This function ensures that the final "start time" plus the window size does not exceed the range of the data
    """

    if stride is None:  # no sliding window
        # Add a time column in datetime format for better processing and set it as the index
        df['datetime'] = pd.to_datetime(df['start'], unit='s')
        df = df.set_index('datetime')

        # Use built in resampling method
        new_window_size = str(new_window_size) + "s"
        df_resampled = df.resample(new_window_size).ffill()

        # Convert back to original format for consistency
        df_resampled['start'] = df_resampled.index.astype('int64') // 10 ** 9
        df_resampled = df_resampled.reset_index(drop=True)
        df_resampled['name'] = df_resampled['name'].fillna(0).astype(int)

        return df_resampled
    else:
        raise Exception("Does not currently support sliding windows. Slide must be -1")


def combine_acc_features_from_file(file, staging_file, output_path, threshold=0.0, window_size=30, stride=None):
    output_file_name = os.path.join(output_path, f"{os.path.basename(file).replace('_acc.csv', '_acc_features.csv')}")

    # Check if the output file already exists
    if os.path.exists(output_file_name):
        print(f"Skipping {file.split('/')[-1]}: already processed.")
        return

    df = pd.read_csv(file)
    df_label_time = pd.read_csv(staging_file)

    # Print the length of each _acc file
    print(f"Now processing > {file.split('/')[-1]}: {len(df)} rows")

    # Resample staging df to appropriate time windows as requested (extraction is based on it)
    df_label_time = resample_sleep_staging(df_label_time, new_window_size=window_size, stride=stride)

    # Compute features for chunks based on start times from df_label_time
    features = compute_features(df, df_label_time, threshold)

    # Generate an output file name
    output_file_name = os.path.join(output_path, f"{os.path.basename(file).replace('_acc.csv', '_acc_features.csv')}")

    # Save features to CSV
    features.to_csv(output_file_name, index=False)
    print(f"Saved features to: {output_file_name}")


def extract_acc_features(source_files_path, output_path, window_size=30, stride=None):
    """
    Note window size must always be a factor or multiple of 30, otherwise sleep stages shown in
    extracted features may be inaccurate.

    :param source_files_path: path to directory containing synced _acc, _eeg, _ppg, and _staging2 .csv files
    :param output_path: directory where output folder (with extracted features) will be created and saved
    :param window_size: determines the size of the window used to compute features
    :param stride: for overlapping/sliding windows, determines the offset between each sliding window
    """
    # Get all CSV files that end with _acc and _staging from the specified directory
    path = os.path.join(source_files_path, '')
    acc_files = sorted(glob.glob(path + '*_acc.csv'))
    staging_files = sorted(glob.glob(path + '*_staging2.csv'))

    # Extract file prefixes, confirm that they all match up, and define the output directory
    acc_prefixes = [file.split('_acc.csv')[0] for file in acc_files]
    staging_prefixes = [file.split('_staging2.csv')[0] for file in staging_files]

    if len(acc_files) != len(staging_files) or acc_prefixes != staging_prefixes:
        raise ValueError("Mismatch in the number of _acc and _staging files or their prefixes")

    stride_string = "NA" if stride is None else stride  #
    output_path = output_path + "ACC_features_window" + str(window_size) + "_stride" + str(stride_string) + "/"
    os.makedirs(output_path, exist_ok=True)

    # Compute and combine features from all files and save them individually. Leverage joblib parallelism to use all cores.
    print("Extracting features from " + source_files_path + "\nWindow size: " + str(window_size) + "\nStride: " + str(stride) + "\n --- ")
    Parallel(n_jobs=CPU_CORES_AVAILABLE - 1)(delayed(combine_acc_features_from_file)(acc_file, staging_file, output_path, threshold=0.0, window_size=window_size, stride=stride) for acc_file, staging_file in zip(acc_files, staging_files))

if __name__ == '__main__':
    extract_acc_features(config.path.synced_csv_directory, config.path.ACC_features_directory, window_size=20,
                         stride=None)
    extract_acc_features(config.path.synced_csv_directory, config.path.ACC_features_directory, window_size=15,
                         stride=None)
    extract_acc_features(config.path.synced_csv_directory, config.path.ACC_features_directory, window_size=10,
                         stride=None)
    extract_acc_features(config.path.synced_csv_directory, config.path.ACC_features_directory, window_size=5,
                         stride=None)
    extract_acc_features(config.path.synced_csv_directory, config.path.ACC_features_directory, window_size=1,
                         stride=None)
