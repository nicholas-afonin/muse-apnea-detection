import matplotlib.pyplot as plt
import config
import os
from sklearn.metrics import confusion_matrix
import numpy as np



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