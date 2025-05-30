import pandas as pd
import numpy as np
from scipy.signal import welch, butter, filtfilt
from scipy.stats import entropy
import glob

from yasa import sw_detect, spindles_detect
from scipy import integrate
import antropy as ant
import scipy.stats as sp_stats
import warnings; warnings.simplefilter('ignore')
import config
import os
import math
import warnings
import time
from joblib import Parallel, delayed

"""
NOTE
Currently only detects sleep spindles and slow waves for 30s windows. Anything less doesn't work due to
the way YASA does the calculation. Ask Nick for more details.
"""

if config.running_locally:
    CPU_CORES_AVAILABLE = 10
else:
    CPU_CORES_AVAILABLE = int(os.getenv("SLURM_CPUS_PER_TASK", os.cpu_count()))
    print(f"{CPU_CORES_AVAILABLE} CPU cores available")

# Define a function to create a bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Define a function to apply the bandpass filter to the EEG data
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    #y_= lfilter(b, a, data)
    y = filtfilt(b, a, data)  # Using filtfilt instead of lfilter
    # # 4. Plot just the filtered signals
    # plt.figure(figsize=(10, 5))
    # plt.plot(data, 'b-', label='lfilter (with delay)')
    # plt.plot(filtered_lfilter, 'g-', label='lfilter (with delay)')
    # plt.plot(filtered_filtfilt, 'r-', label='filtfilt (zero-phase)', alpha=0.75)
    # plt.xlabel('Time [s]')
    # plt.ylabel('Amplitude')
    # plt.title('Filtered Signal Comparison: lfilter vs filtfilt')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return y


def bandpower(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    

    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    if np.abs(integrate.simpson(psd, dx=freq_res)) > 1e-10:  # Check if the denominator is not too close to zero
        bp = integrate.simpson(psd[idx_band], dx=freq_res)
    else:
        bp = 0  # Set bp to a default value or handle the division by zero case in an appropriate way


    if relative:
        total_power = integrate.simpson(psd, dx=freq_res)
        if np.abs(total_power) > 1e-10:  # Check if total_power is not too close to zero
            bp /= total_power
        else:
            bp = 0  # Set bp to a default value or handle the division by zero case in an appropriate way

    return bp


def compute_psd_features_for_channel(data, sp_sw_data, bandpower_data, sampling_rate=256):
    # Compute bandpowers
    band_powers = {}
    band_powers_rel = {}
    # for band_name, band_limits in zip(['Delta','DeltaL','DeltaH',   'Theta','ThetaL','ThetaH',   'Alpha','AlphaL','AlphaH',     'Beta', 'BetaL', 'BetaM','BetaH',         'Gamma', 'GammaL', 'GammaH'], 
    #                                   [[0.5, 4],[0.5, 2],[2, 4],    [4, 8], [4, 6], [6, 8],      [8, 12], [8, 10], [10, 12],    [12, 30], [12, 16], [16, 22], [22, 30],   [30, 40], [30, 35], [35, 40]]):
    for band_name, band_limits in zip(['Delta','DeltaL', 'DeltaH',  'Theta',  'Alpha',    'Beta', 'BetaL', 'BetaM','BetaH', 'Gamma'], [[0.5, 4],[0.5,2], [2,4],   [4, 6],   [6, 10] ,   [12, 30], [9, 16], [16, 22], [22, 30],   [30, 40]]):    
        #print(band_limits)
        # band_powers[band_name] = bandpower(data, sampling_rate, band_limits, 'multitaper')
        # band_powers_rel[band_name] = bandpower(data, sampling_rate, band_limits, 'multitaper', relative=True)
        win_sec = 5
        band_powers[band_name] = bandpower(bandpower_data, sampling_rate, band_limits, win_sec)
        band_powers_rel[band_name] = bandpower(bandpower_data, sampling_rate, band_limits, win_sec, relative=True)


    # Calculate standard descriptive statistics
    hMob, hComp= ant.hjorth_params(data)
    perm = np.apply_along_axis(ant.perm_entropy,  axis=0, arr=data, normalize=True)
    higuchi = np.apply_along_axis(ant.spectral_entropy, sf=256, axis=0, arr=data)   ### Previously it was higuchi
    petrosian = ant.petrosian_fd(data, axis=0)
    katz = ant.katz_fd(data, axis=0)

    # Extract stat features using scipy_stats
    std = np.std(data)
    iqr = sp_stats.iqr(data, rng=(25, 75))
    skew = sp_stats.skew(data)
    kurt = sp_stats.kurtosis(data)

    # Extract slow-wave and sleep spindle counts
    # Uses a different chunk than the rest of the features because yasa.spindles_detect and yasa.sw_detect require
    # >= 30-second windows. See 'obtain_sw_sp_compatible_chunks' function for more details.
    min_len = 30 * sampling_rate  # 30-s ⇒ 7680 samples @256 Hz
    if len(sp_sw_data) < min_len:
        sp_count = 0
        sw_count = 0
    else:
        sp = spindles_detect(sp_sw_data, sampling_rate, verbose="critical")
        sp_count = 0 if sp is None else len(sp.summary())
        sw = sw_detect(sp_sw_data, sampling_rate, verbose="critical")
        sw_count = 0 if sw is None else len(sw.summary())


    return {
        "Delta": band_powers['Delta'],
        "DeltaL": band_powers['DeltaL'],
        "DeltaH": band_powers['DeltaH'],

        "Theta": band_powers['Theta'],
        # "ThetaL": band_powers['ThetaL'],
        # "ThetaH": band_powers['ThetaH'],

        "Alpha": band_powers['Alpha'],
        # "AlphaL": band_powers['AlphaL'],
        # "AlphaH": band_powers['AlphaH'],

        "Beta": band_powers['Beta'],
        "BetaL": band_powers['BetaL'],
        "BetaM": band_powers['BetaM'],
        "BetaH": band_powers['BetaH'],
        
        "Gamma": band_powers['Gamma'],
        # "GammaL": band_powers['GammaL'],
        # "GammaH": band_powers['GammaH'],

        #################
        "Delta_rel": band_powers_rel['Delta'],
        "DeltaL_rel": band_powers_rel['DeltaL'],
        "DeltaH_rel": band_powers_rel['DeltaH'],

        "Theta_rel": band_powers_rel['Theta'],
        # "ThetaL_rel": band_powers_rel['ThetaL'],
        # "ThetaH_rel": band_powers_rel['ThetaH'],

        "Alpha_rel": band_powers_rel['Alpha'],
        # "AlphaL_rel": band_powers_rel['AlphaL'],
        # "AlphaH_rel": band_powers_rel['AlphaH'],

        "Beta_rel": band_powers_rel['Beta'],
        "BetaL_rel": band_powers_rel['BetaL'],
        "BetaM_rel": band_powers_rel['BetaM'],
        "BetaH_rel": band_powers_rel['BetaH'],
        
        "Gamma_rel": band_powers_rel['Gamma'],
        # "GammaL_rel": band_powers_rel['GammaL'],
        # "GammaH_rel": band_powers_rel['GammaH'],

        "std": std,
        "iqr": iqr,
        "skew": skew,
        "kurt": kurt,

        "perm": perm,
        "higuchi": higuchi,
        "petrosian": petrosian,
        "katz": katz,
        "hMob": hMob,
        "hComp": hComp,
        "sp_count": sp_count,
        "sw_count": sw_count
    }


def shannon_entropy(data):
    # Check if data is empty
    if len(data) == 0:
        return 0.0  # Return 0 entropy for empty data

    # Check for NaN values and remove them
    data = data[~np.isnan(data)]

    # Check if data is still non-empty after removing NaN values
    if len(data) == 0:
        return 0.0  # Return 0 entropy if all values were NaN

    return entropy(np.histogram(data, density=True)[0])


def ceildiv(a, b):
    # Simple ceiling division implementation
    return -(a // -b)


def obtain_sw_sp_compatible_chunks(df, start_time, window_size, sampling_rate=256):
    """
    Yasa's sw and sp detection require data chunks of AT LEAST 30 seconds duration (see logic below, sometimes could
    be a second or two longer)
    This function calculates chunks that have the same middle timestamp as the properly windowed chunks, but
    have durations of 30 seconds.
    """

    first_second = df['ts'].iloc[0]
    last_second = df['ts'].iloc[-1]

    half_window = window_size / 2
    middle_of_chunk = start_time + half_window # NOTE THIS CAN BE A FLOAT CURRENTLY

    if middle_of_chunk - 15 < first_second:  # If the start of the 30s window is cut off
        sp_sw_start_time = first_second
        sp_sw_end_time = first_second + 30
    elif middle_of_chunk + 15 > last_second:  # If the end of the 30s window is cut off
        sp_sw_start_time = last_second - 30
        sp_sw_end_time = last_second
    else:  # If nothing is cut off and the 30s window extends cleanly 15s from the middle of the chunk
        sp_sw_start_time = math.floor(middle_of_chunk - 15)
        sp_sw_end_time = math.ceil(middle_of_chunk + 15)

    # Slice out appropriate chunk of data that is at least 30s no matter what
    sp_sw_chunk = df[(df['ts'] >= sp_sw_start_time) & (df['ts'] <= sp_sw_end_time)]

    return sp_sw_chunk


def obtain_bandpower_compatible_chunks(df, start_time, window_size, sampling_rate=256):
    """
    similar issue to sw and sp but for the bandpower calculation, which requires 5s windows.
    note variable names are not updated. this function will just use the normal window size unless if it's
    less than 5s, in which case it always uses minimum 5s -- for appropriate bandpower calculations.
    """

    first_second = df['ts'].iloc[0]
    last_second = df['ts'].iloc[-1]

    half_window = window_size / 2
    middle_of_chunk = start_time + half_window # NOTE THIS CAN BE A FLOAT CURRENTLY

    if window_size < 5:
        bandpower_window_size = 5
    else:
        bandpower_window_size = window_size

    if middle_of_chunk - 15 < first_second:  # If the start of the 30s window is cut off
        sp_sw_start_time = first_second
        sp_sw_end_time = first_second + bandpower_window_size
    elif middle_of_chunk + 15 > last_second:  # If the end of the 30s window is cut off
        sp_sw_start_time = last_second - bandpower_window_size
        sp_sw_end_time = last_second
    else:  # If nothing is cut off and the 30s window extends cleanly 15s from the middle of the chunk
        sp_sw_start_time = math.floor(middle_of_chunk - bandpower_window_size/2)
        sp_sw_end_time = math.ceil(middle_of_chunk + bandpower_window_size/2)

    # Slice out appropriate chunk of data that is at least 30s no matter what
    sp_sw_chunk = df[(df['ts'] >= sp_sw_start_time) & (df['ts'] <= sp_sw_end_time)]

    return sp_sw_chunk



def compute_eeg_features_for_window(df, df_label_time, sampling_rate=256):
    # Check if required columns are present in the input dataframe
    required_columns = ['ts', 'ch1', 'ch2', 'ch3', 'ch4']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")
    
    # Check if 'start' and 'name' columns are present in df_label_time
    if 'start' not in df_label_time.columns or 'name' not in df_label_time.columns:
        raise ValueError(f"Columns 'start' or 'name' not found in the df_label_time DataFrame.")

    # Initialize features list based on sleep staging file
    feature_list = []
    start_times = df_label_time['start'].tolist()
    names = df_label_time['name'].tolist()
    window_size = start_times[1] - start_times[0]  # needs to change if overlapping windows are ever considered

    # Apply butter bandpass filter to all channels of data before processing windows
    filtered_df = df.copy()
    lowcut, highcut = 0.5, 30.0
    for ch in ['ch1', 'ch2', 'ch3', 'ch4']:
        filtered_df[ch] = butter_bandpass_filter(filtered_df[ch], lowcut, highcut, sampling_rate)

    # Compute relevant features for each window of data. Leveraging parallel processing.
    for i in range(len(start_times) - 1):
        chunk = filtered_df[(filtered_df['ts'] >= start_times[i]) & (filtered_df['ts'] < start_times[i + 1])]
        sp_sw_chunk = obtain_sw_sp_compatible_chunks(filtered_df, start_times[i], window_size=window_size)
        bandpower_chunk = obtain_bandpower_compatible_chunks(filtered_df, start_times[i], window_size=window_size)

        # Check if the chunk is empty
        if chunk.empty:
            print(f"Warning: No data found for start time {start_times[i]} to {start_times[i + 1]}. Skipping...")
            continue

        # Initialize features with timestamp and name
        features = {"ts": chunk['ts'].iloc[0], "name": names[i]}

        for ch in ['ch1', 'ch2', 'ch3', 'ch4']:
            ch_features = compute_psd_features_for_channel(chunk[ch], sp_sw_chunk[ch], bandpower_chunk[ch],
                                                           sampling_rate)

            for band, power in ch_features.items():
                features[f"{ch}_{band}"] = power
            features[f"{ch}_ShannonEntropy"] = shannon_entropy(chunk[ch])

        feature_list.append(features)
    
    # Convert the list of dictionaries into a dataframe
    features_df = pd.DataFrame(feature_list)
    
    # Ensure 'ts' is the first column
    cols = ['ts'] + [col for col in features_df if col != 'ts']
    features_df = features_df[cols]
    
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

def save_features_for_subject(features_df, subject_name, output_folder):
    os.makedirs(output_folder, exist_ok=True)  # Ensures the output path actually exists and makes it if not
    output_path = os.path.join(output_folder, f"{subject_name}_eeg_features.csv")
    features_df.to_csv(output_path, index=False)
    print(f"Saved EEG features for {subject_name} to: {output_path}")

def extract_and_save_features_for_recording(file, staging_file, output_folder, existing_files, window_size=30, stride=None):
    time_start = time.time()

    # Extract subject name from file path
    filename_with_extension = os.path.basename(file)
    subject_name = filename_with_extension.split('_eeg.csv')[0]

    # Check if features file already exists for this subject
    if subject_name in existing_files:
        print(f"Features already calculated for {subject_name}. Skipping...")
        return

    df = pd.read_csv(file)
    df_label_time = pd.read_csv(staging_file)

    # Print the length of each _acc file
    print(f"Now processing > {file.split('/')[-1]}: {len(df)} rows")

    # Resample sleep stage labels so that features are extracted for the relevant time windows
    df_label_time = resample_sleep_staging(df_label_time, new_window_size=window_size, stride=stride)

    # Compute features for chunks based on start times from df_label_time
    features_30s = compute_eeg_features_for_window(df, df_label_time)

    # Save features for this subject
    save_features_for_subject(features_30s, subject_name, output_folder)

    print(f"Feature calculation for {file} complete. Took {time.time()-time_start} seconds.")


def extract_eeg_features(source_files_path, output_path, window_size=30, stride=None):
    """
    Note window size must always be a factor or multiple of 30, otherwise sleep stages shown in
    extracted features may be inaccurate.

    :param source_files_path: path to directory containing synced _acc, _eeg, _ppg, and _staging2 .csv files
    :param output_path: directory where output folder (with extracted features) will be created and saved
    :param window_size: determines the size of the window used to compute features
    :param stride: for overlapping/sliding windows, determines the offset between each sliding window
    """

    # Get all CSV files that end with _eeg and _staging from the specified directory
    path = os.path.join(source_files_path, '')  # os.path.join ensures there's a slash at the end

    # Get all files
    eeg_files = sorted(glob.glob(path + '*_eeg.csv'))
    staging_files = sorted(glob.glob(path + '*_staging2.csv'))

    # List of prefixes to skip
    skip_prefixes = [
        '2023-01-23T213736-0500_6002-2MLB-5F48_synced'] #, 'xxx_synced']

    # Filter out files containing any of the skip prefixes
    eeg_files = [f for f in eeg_files if not any(skip_prefix in f for skip_prefix in skip_prefixes)]
    staging_files = [f for f in staging_files if not any(skip_prefix in f for skip_prefix in skip_prefixes)]

    # Extract prefixes
    eeg_prefixes = [file.split('_eeg.csv')[0] for file in eeg_files]
    staging_prefixes = [file.split('_staging2.csv')[0] for file in staging_files]

    # Ensure that for each _eeg file there's a corresponding _staging file
    if len(eeg_files) != len(staging_files) or eeg_prefixes != staging_prefixes:
        raise ValueError("Mismatch in number of _eeg and _staging files or their prefixes")

    # Define the output directory
    stride_string = "NA" if stride is None else stride  #
    output_path = output_path + "EEG_features_window" + str(window_size) + "_stride" + str(stride_string) + "/"
    os.makedirs(output_path, exist_ok=True)

    # Extract features from all files. Leverage joblib parallelism to use all cores.
    print("Extracting features from " + source_files_path + "\nWindow size: " + str(window_size) + "\nStride: " + str(stride) + "\n --- ")
    # Create a set of existing output files for quick lookup
    existing_files = {f.split('_eeg_features.csv')[0]: f for f in os.listdir(output_path) if
                      f.endswith('_eeg_features.csv')}

    # for file, staging_file in zip(eeg_files, staging_files):
    #     extract_and_save_features_for_recording(file, staging_file, output_path, existing_files, window_size=30, stride=stride)

    Parallel(n_jobs=CPU_CORES_AVAILABLE-1)(delayed(extract_and_save_features_for_recording)(eeg_file, staging_file, output_path, existing_files, window_size=window_size, stride=stride) for eeg_file, staging_file in zip(eeg_files, staging_files))


if __name__ == "__main__":
    extract_eeg_features(config.path.synced_csv_directory, config.path.EEG_features_directory, window_size=10, stride=None)