import pandas as pd
import glob
import numpy as np
import os
import config
from scipy import signal
from scipy.fft import fft


def compute_features_for_30s(df, df_label_time, threshold=0.0, cutoff_freq=0.5):
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
        chunk = df[(df['ts'] >= start_times[i]) & (df['ts'] < start_times[i + 1])]

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

def combine_features_from_files(file_list, staging_file_list, output_path, threshold=0.0):
    for file, staging_file in zip(file_list, staging_file_list):

        output_file_name = os.path.join(output_path, f"{os.path.basename(file).replace('_acc.csv', '_acc_features.csv')}")

        # Check if the output file already exists
        if os.path.exists(output_file_name):
            print(f"Skipping {file.split('/')[-1]}: already processed.")
            continue
    
        df = pd.read_csv(file)
        df_label_time = pd.read_csv(staging_file)

        # Print the length of each _acc file
        print(f"Now processing > {file.split('/')[-1]}: {len(df)} rows")
        
        # Compute features for chunks based on start times from df_label_time
        features_30s = compute_features_for_30s(df, df_label_time, threshold)

        # Generate an output file name
        output_file_name = os.path.join(output_path, f"{os.path.basename(file).replace('_acc.csv', '_acc_features.csv')}")
        
        # Save features to CSV
        features_30s.to_csv(output_file_name, index=False)
        print(f"Saved features to: {output_file_name}")

# Get all CSV files that end with _acc and _staging from the specified directory
# Specify the path
path = os.path.join(config.path.synced_csv_directory, '')

# Get all CSV files that end with _acc and _staging
acc_files = sorted(glob.glob(path + '*_acc.csv'))#, reverse=True)
staging_files = sorted(glob.glob(path + '*_staging2.csv'))#, reverse=True)
print(staging_files)

# Extract prefixes
acc_prefixes = [file.split('_acc.csv')[0] for file in acc_files]
staging_prefixes = [file.split('_staging2.csv')[0] for file in staging_files]

# Ensure that for each _acc file there's a corresponding _staging file
if len(acc_files) != len(staging_files) or acc_prefixes != staging_prefixes:
    raise ValueError("Mismatch in the number of _acc and _staging files or their prefixes")

# Define output path
output_path = config.path.ACC_features_directory

# Create the output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Combine features from all files and save them individually
combine_features_from_files(acc_files, staging_files, output_path, threshold=0.0)
