import matplotlib.pyplot as plt
import config
import os
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import json
import re
from pathlib import Path
import pandas as pd



def plot_training_history(history, save_dir):
    # Plot training & validation loss
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # SAVE TRAINING LOSS PLOT
    loss_plot_path = os.path.join(save_dir, f"training_and_validation_loss.png")
    plt.savefig(loss_plot_path)
    plt.close()


def plot_confusion_matrix(actual, predicted, title, save_dir):
    # Compute raw confusion matrix and normalized one (row-wise)
    conf_mat_raw = confusion_matrix(actual, predicted)
    conf_mat_norm = confusion_matrix(actual, predicted, normalize='true')  # normalize by row so colours make sense

    labels = ["No Apnea", "Apnea"]
    n_classes = len(labels)

    # Set up figure
    fig, ax = plt.subplots()
    im = ax.imshow(conf_mat_norm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    # Set ticks and labels
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels,
           ylabel='Ground Truth',
           xlabel='Predicted',
           title=title)

    # Rotate the tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate each cell with count and percentage
    fmt = '.1f'
    thresh = conf_mat_norm.max() / 2.

    for i in range(n_classes):
        for j in range(n_classes):
            count = conf_mat_raw[i, j]
            percent = conf_mat_norm[i, j] * 100

            # --- pick text colour by luminance of the cell colour ---------
            r, g, b, _ = im.cmap(conf_mat_norm[i, j])       # RGBA from colormap
            luminance  = 0.299 * r + 0.587 * g + 0.114 * b  # ITU-R BT.601
            text_col   = "white" if luminance < 0.5 else "black"
            # ----------------------------------------------------------------

            ax.text(j, i, f"{count}\n({percent:.1f}%)",
                    ha="center", va="center",
                    color=text_col)

    fig.tight_layout()
    plt.ioff()

    # SAVE PLOT instead of displaying
    plot_path = os.path.join(save_dir, f"confusion_matrix.png")
    fig.savefig(plot_path)
    plt.close(fig)


def plot_grid_search_results(
        results_directory: str | Path,
        metrics: tuple[str, ...] = ("f1",),
        save_figures: bool = False,
        dpi: int = 200,
        vmin: float | None = None,
        vmax: float | None = None,
):
    """
    Crawl a directory laid out like
        0.01_thresh_10s_window/
            ├─ metrics.json
            └─ ...
        0.01_thresh_15s_window/
        0.50_thresh_30s_window/
        ...
    and create a heat-map-style grid for each requested metric.

    Parameters
    ----------
    results_directory : str | Path
        Root folder containing all result sub-folders.
    metrics : tuple[str, ...], default ("f1",)
        Which keys from `metrics.json` you’d like to visualise.
    save_figures : bool, default False
        If True, writes a PNG next to `results_directory` for each metric.
    dpi : int, default 200
        Resolution for saved figures.
    """
    results_directory = Path(results_directory).expanduser().resolve()
    pattern = re.compile(r"(?P<thresh>[0-9.]+)_thresh_(?P<win>\d+)s_window")

    # --------------------------- 1. Load all metrics --------------------------
    rows = []
    for sub in results_directory.iterdir():
        if not sub.is_dir():
            continue
        m = pattern.fullmatch(sub.name)
        if m is None:
            continue

        # Parse hyper-params from folder name
        thresh = float(m.group("thresh"))
        window = int(m.group("win"))

        metrics_file = sub / "metrics.json"
        if not metrics_file.exists():
            print(f"⚠️  No metrics.json in {sub}")
            continue

        with metrics_file.open() as f:
            data = json.load(f)

        rows.append({"thresh": thresh, "window": window, **data})

    if not rows:
        raise RuntimeError(f"No matching result folders found under {results_directory}")

    df = pd.DataFrame(rows)

    # --------------------------- 2. Plot each metric --------------------------
    for metric in metrics:
        if metric not in df.columns:
            print(f"⚠️  Metric '{metric}' not found in any metrics.json – skipping")
            continue

        # Pivot → thresholds as rows, windows as cols
        grid = df.pivot_table(index="thresh", columns="window", values=metric)
        grid = grid.sort_index()
        grid = grid.reindex(sorted(grid.columns), axis=1)

        # Matplotlib heat-map (no seaborn – per project rules)
        fig, ax = plt.subplots()
        im = ax.imshow(grid.values, vmin=vmin, vmax=vmax)  # uses default colormap (viridis)

        # Pretty ticks / labels
        ax.set_xticks(range(len(grid.columns)))
        ax.set_xticklabels(grid.columns)
        ax.set_xlabel("Window size (s)")

        ax.set_yticks(range(len(grid.index)))
        ax.set_yticklabels(grid.index)
        ax.set_ylabel("Apnea prob. threshold")

        ax.set_title(metric.upper())
        fig.colorbar(im, ax=ax)

        plt.tight_layout()

        if save_figures:
            out = results_directory / f"{metric}_grid.png"
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            print(f"Saved → {out}")

        plt.show()


def plot_grid_search_results_temp_cooked(
        results_directory: str | Path,
        metrics: tuple[str, ...] = ("f1",),
        save_figures: bool = False,
        dpi: int = 200,
        vmin: float | None = None,
        vmax: float | None = None,
):
    """TEMPORARY FUNCTION FOR A VERY SPECIFIC PURPOSE"""

    results_directory = Path(results_directory).expanduser().resolve()
    pattern = re.compile(r"(?P<thresh>[0-9.]+)_thresh_(?P<win>\d+)s_window")

    # --------------------------- 1. Load all metrics --------------------------
    rows = []
    for sub in results_directory.iterdir():
        if not sub.is_dir():
            continue
        m = pattern.fullmatch(sub.name)
        if m is None:
            continue

        # Parse hyper-params from folder name
        thresh = float(m.group("thresh"))
        window = int(m.group("win"))

        metrics_file = sub / "lightning_logs/version_0/metrics.json"
        if not metrics_file.exists():
            print(f"⚠️  No metrics.json in {sub}")
            continue

        with metrics_file.open() as f:
            data = json.load(f)

        rows.append({"thresh": thresh, "window": window, **data})

    if not rows:
        raise RuntimeError(f"No matching result folders found under {results_directory}")

    df = pd.DataFrame(rows)

    # --------------------------- 2. Plot each metric --------------------------
    for metric in metrics:
        if metric not in df.columns:
            print(f"⚠️  Metric '{metric}' not found in any metrics.json – skipping")
            continue

        # Pivot → thresholds as rows, windows as cols
        grid = df.pivot_table(index="thresh", columns="window", values=metric)
        grid = grid.sort_index()
        grid = grid.reindex(sorted(grid.columns), axis=1)

        # Matplotlib heat-map (no seaborn – per project rules)
        fig, ax = plt.subplots()
        im = ax.imshow(grid.values, vmin=vmin, vmax=vmax)

        # Pretty ticks / labels
        ax.set_xticks(range(len(grid.columns)))
        ax.set_xticklabels(grid.columns)
        ax.set_xlabel("Window size (s)")

        ax.set_yticks(range(len(grid.index)))
        ax.set_yticklabels(grid.index)
        ax.set_ylabel("Apnea prob. threshold")

        ax.set_title(metric.upper())
        fig.colorbar(im, ax=ax)

        plt.tight_layout()

        if save_figures:
            out = results_directory / f"{metric}_grid.png"
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            print(f"Saved → {out}")

        plt.show()


if __name__ == "__main__":
    plot_grid_search_results(config.path.apnea_model_directory,
                             metrics=("mcc",),
                             save_figures=True,
                             vmin=0.0,
                             vmax=0.30)