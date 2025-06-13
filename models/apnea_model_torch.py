import os, glob, random, json, pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef,
    f1_score, precision_score, recall_score
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

import config                                     # unchanged
from diagnostics.evaluation_helper import *

torch.set_float32_matmul_precision('medium')


""" -------------------------------------- HELPER FUNCS -------------------------------------- """


def combine_files_into_df(csv_path_list):
    """Take a list of paths to csv files and convert them all into one big dataframe"""
    dfs = [pd.read_csv(f) for f in csv_path_list]
    combined = pd.concat(dfs, ignore_index=True)

    return combined


""" -------------------------------------- DATA LOADING -------------------------------------- """


def load_data(processed_features_directory: str, train_val_test_ratio: (float, float, float)):
    all_csv_file_paths = glob.glob(processed_features_directory + "*.csv")

    # Break down into data proportions
    train_ratio, val_ratio, test_ratio = train_val_test_ratio

    # Shuffle the list (for randomness, but reproducible with a seed)
    random.seed(21)
    random.shuffle(all_csv_file_paths)

    # Split into train and test
    split_index = int(len(all_csv_file_paths) * train_ratio)
    train_files = all_csv_file_paths[:split_index]
    test_files = all_csv_file_paths[split_index:]

    train_df = combine_files_into_df(train_files)
    test_df = combine_files_into_df(test_files)

    # Remove all rows with NaNs
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    # Scale all data to be from 0 to 1
    scaler = MinMaxScaler()
    scaler.fit(train_df)
    train_df = pd.DataFrame(scaler.transform(train_df), columns=train_df.columns)
    test_df = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)

    # Split into X and Y train and test
    X_train = train_df.drop(columns=['ts', 'arousal_event'])
    X_test = test_df.drop(columns=['ts', 'arousal_event'])

    y_train = X_train.pop('apnea_event')
    y_test = X_test.pop('apnea_event')

    # Create stratified train/validation split
    if val_ratio != 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=val_ratio/train_ratio,
            stratify=y_train,
            random_state=43
        )
    else:
        # IF VALIDATION DATA IS SET UP AS EQUAL TO TEST, BE CAREFUL
        X_val = X_test
        y_val = y_test

    return X_train, X_val, X_test, y_train, y_val, y_test

""" -------------------------------------- MODEL ARCHITECTURE --------------------------------------"""

class ApneaClassifier(pl.LightningModule):
    def __init__(self, input_dim: int = 168, lr: float = 1e-5,
                 pos_weight: float = 1.0):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128,  80), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear( 80,  80), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear( 80,   1)
        )
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    def forward(self, x):               # (B, 168)
        return self.net(x).squeeze(1)   # (B,)

    def training_step(self, batch, _):
        x, y = batch
        loss = self.loss_fn(self(x), y.float())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        loss = self.loss_fn(self(x), y.float())
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        decay = 0.01
        return torch.optim.Adam(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=decay)


""" -------------------------------------- TRAINING --------------------------------------"""


def _to_loader(X: pd.DataFrame, y: pd.Series, batch: int, shuffle=False, persistent_workers=False, pin_memory=False):
    ds = TensorDataset(torch.tensor(X.values, dtype=torch.float32),
                       torch.tensor(y.values, dtype=torch.long))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=4, persistent_workers=persistent_workers,
                      pin_memory=pin_memory)


def train_model(X_train, y_train, X_val, y_val, save_as: str):
    # class-imbalance weighting → BCEWithLogits(pos_weight)
    weights = class_weight.compute_class_weight('balanced',
                                                classes=np.unique(y_train),
                                                y=y_train)
    pos_w = torch.tensor(weights[1], dtype=torch.float32)

    # Initialize model and shit.
    model = ApneaClassifier(pos_weight=pos_w)

    root_dir = os.path.join(config.path.apnea_model_directory, save_as)
    model_specific_dir = os.path.join(config.path.apnea_model_directory, save_as, "lightning_logs", f"version_{VERSION}")
    logger = CSVLogger(root_dir, name="lightning_logs", flush_logs_every_n_steps=1)
    ckpt = ModelCheckpoint(dirpath=model_specific_dir,
                           filename="best",
                           save_top_k=1,
                           monitor="val_loss",
                           mode="min")
    early = EarlyStopping(monitor="val_loss", patience=10, mode="min")

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="auto", devices="auto",
        logger=logger,
        log_every_n_steps=10,
        callbacks=[ckpt, early],
        default_root_dir=root_dir,
        enable_progress_bar=True,
        enable_model_summary=True,
        check_val_every_n_epoch=1
    )

    trainer.fit(
        model,
        _to_loader(X_train, y_train, 256, shuffle=True, persistent_workers=True),
        _to_loader(X_val,   y_val,   256, shuffle=False, persistent_workers=True, pin_memory=True)
    )


""" -------------------------------------- EVALUATION --------------------------------------"""

@torch.no_grad()
def evaluate_model(X, y, load_as: str,
                   relevant_threshold_value, relevant_window_size, version):

    # Load the model
    root_dir = os.path.join(config.path.apnea_model_directory, load_as, "lightning_logs", f"version_{version}")


    ckpt_path = os.path.join(root_dir, "best.ckpt")
    model = ApneaClassifier.load_from_checkpoint(ckpt_path).eval().to("cpu")

    # Obtain logits and predicted probabilities
    logits = model(torch.tensor(X.values, dtype=torch.float32)).numpy()
    probs  = 1 / (1 + np.exp(-logits))
    preds  = (probs >= 0.50).astype(int)
    actual = y.values.astype(int)

    title = f"{relevant_window_size} Window, {relevant_threshold_value} Threshold"
    plot_confusion_matrix(actual, preds, title, root_dir)

    metrics = {
        "roc_auc":  float(roc_auc_score(actual, probs)),
        "mcc":      float(matthews_corrcoef(actual, preds)),
        "f1":       float(f1_score(actual, preds)),
        "precision":float(precision_score(actual, preds)),
        "recall":   float(recall_score(actual, preds))
    }
    json.dump(metrics, open(os.path.join(root_dir, "metrics.json"), "w"), indent=2)


""" -------------------------------------- INTERFACE --------------------------------------"""


def main(threshold, window_size, train_val_test_ratio: (float, float, float), model_name, version, evaluate_only=False, ):
    # Load data in appropriate proportions (randomize and split by files)
    full_data_path = os.path.join(config.path.EEG_ACC_features_labelled,
                                  f'EEG_ACC_features_labelled_ApneaThreshold{threshold}_ArousalThreshold{threshold}_window{window_size}/')
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(full_data_path, train_val_test_ratio)

    # Train the model
    if not evaluate_only:
        train_model(X_train, y_train, X_val, y_val, save_as=model_name)

    # Evaluate the model
    evaluate_model(X_test, y_test, load_as=model_name, relevant_threshold_value=threshold,
                   relevant_window_size=window_size, version=version)


if __name__ == "__main__":
    VERSION = 0  #  NOTE WHOLE VERSION THING IS BROKEN STILL - NEED TO FIGURE OUT WHAT TO DO ABOUT THAT

    threshold = 0.50
    window = 15
    train_val_test_ratio = (0.70, 0.15, 0.15)
    model_name = f"{threshold}_thresh_{window}s_window"

    main(threshold, window, train_val_test_ratio, model_name, VERSION, evaluate_only=False)