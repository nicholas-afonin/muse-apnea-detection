
import glob
import pandas as pd

print("hello")

path = '/scratch/dgurve/Muse_synced_csv_25_April_2025/'
# Get all files
all_files = sorted(glob.glob(path + '*.csv'))


for file in all_files:
    df = pd.read_csv(file)
    print(df)


print(all_files)

