import os
import pandas as pd
import numpy as np
import config

RAW_DIR  = config.path.raw_csv_directory         # where the raw *_events.csv files live
OUT_DIR  = config.path.dataset_statistics        # where to write merged results
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------
# 1. Read & stack all *_events.csv files
# ----------------------------------------------------------
dfs = []
for f in os.listdir(RAW_DIR):
    if f.endswith('.csv') and '_events' in f:
        df = pd.read_csv(
            os.path.join(RAW_DIR, f),
            usecols=['name', 'start', 'duration']
        )
        df["source_file"] = f  # ← keep origin for later
        dfs.append(df)

if not dfs:
    raise RuntimeError("No *_events.csv files found!")

events = pd.concat(dfs, ignore_index=True)

# If duration field is in ms, convert:
# events['duration'] = events['duration'] / 1000.0

# ----------------------------------------------------------
# 2. Map each event to “family” (apnea vs arousal)
# ----------------------------------------------------------
event_map = {
    'apnea_event'  : ['Hypopnea', 'Obstructive Apnea', 'Mixed Apnea', 'Central Apnea'],
    'arousal_event': ['RERA', 'Arousal (ARO RES)', 'Arousal (ARO SPONT)'],
}

def to_family(label):
    for fam, labels in event_map.items():
        if label.strip() in labels:
            return fam
    return 'other'

events['family'] = events['name'].apply(to_family)
events = events.query("family != 'other'")   # drop events we don’t care about

# ----------------------------------------------------------
# 3. Descriptive statistics on duration
# ----------------------------------------------------------
summary = (
    events.groupby('family')['duration']
          .agg(N='count', Mean='mean', Std='std', Min='min', Max='max')
          .round(2)
)

# Overall
overall = events['duration'].agg(N='count', Mean='mean', Std='std',
                                 Min='min', Max='max').round(2)

print("\nDuration summary by family:\n", summary)
print("\nOverall duration stats:\n", overall)

# ----------------------------------------------------------
# 3b. Histogram of duration for each family
# ----------------------------------------------------------
import matplotlib
matplotlib.use("Agg")          # explicit non-GUI backend
import matplotlib.pyplot as plt

for fam, group in events.groupby("family"):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    ax.hist(group["duration"], bins="auto", edgecolor="black")
    ax.set_title(f"Duration distribution – {fam}")
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"duration_hist_{fam}.png"))
    plt.close(fig)             # important to free memory

# ----------------------------------------------------------
# 3c. Flag events whose duration is below family thresholds
# ----------------------------------------------------------
thresholds = {"apnea_event": 5, "arousal_event": 1}   # seconds

violations = (
    events.loc[
        events.apply(
            lambda r: r["duration"] < thresholds.get(r["family"], np.inf),
            axis=1
        ),
        ["family", "start", "duration", "source_file"]
    ]
)

if violations.empty:
    print("\n✅ No events below thresholds.")
else:
    out_vio = os.path.join(OUT_DIR, "events_below_threshold.csv")
    violations.to_csv(out_vio, index=False)
    print(f"\n⚠️  {len(violations)} events below thresholds → {out_vio}")


# ----------------------------------------------------------
# 4. Save for later
# ----------------------------------------------------------
events.to_csv(os.path.join(OUT_DIR, 'all_events_with_duration.csv'), index=False)
summary.to_csv(os.path.join(OUT_DIR, 'duration_summary_by_family.csv'))
