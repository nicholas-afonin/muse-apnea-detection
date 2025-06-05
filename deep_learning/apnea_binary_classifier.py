import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.metrics import Precision
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, f1_score
from sklearn.metrics import classification_report, roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
import config
import random
import glob
import os
import pickle
import json


""" -------------------------------------- HELPER FUNCS -------------------------------------- """


def combine_files_into_df(csv_path_list):
    """Take a list of paths to csv files and convert them all into one big dataframe"""
    dfs = [pd.read_csv(f) for f in csv_path_list]
    combined = pd.concat(dfs, ignore_index=True)

    return combined


""" -------------------------------------- CORE FUNCS -------------------------------------- """


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

    # Optional - Oversample the minority classes in the training data to give more to train on
    # smote = SMOTE()
    # X_train, y_train = smote.fit_resample(X_train, y_train)

    # Optional - Debugging
    # print("Unique labels in y_train:", y_train.unique())
    # print("Any NaNs?", y_train.isna().sum())
    # print(X_train.shape)
    # print(X_train.describe())
    # assert not np.isnan(X_train.values).any(), "NaNs in X_train!"
    # assert not np.isinf(X_train.values).any(), "Infs in X_train!"

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train, y_train, X_val, y_val, save_as: str):
    # Create model architecture
    basic_model = Sequential()  # initialize a basic sequential model
    basic_model.add(Dense(units=128, activation='relu', input_shape=(168,), kernel_regularizer=l2(0.01)))  # input layer
    basic_model.add(Dropout(0.4))  # to prevent overfitting
    basic_model.add(Dense(80, activation='relu', kernel_regularizer=l2(0.01)))
    basic_model.add(Dropout(0.4))
    basic_model.add(Dense(80, activation='relu', kernel_regularizer=l2(0.01)))
    basic_model.add(Dropout(0.4))
    basic_model.add(Dense(1, activation='sigmoid'))

    adam = keras.optimizers.Adam(learning_rate=0.00001)
    basic_model.compile(loss='binary_crossentropy', optimizer=adam)

    # Add early stopping to prevent wasting time without validation improvements
    early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

    # Tell the loss function to weigh the minority cases much more heavily
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = {i: weights[i] for i in range(len(weights))}

    # Train model
    history = basic_model.fit(
        X_train, y_train,
        epochs=100,
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=[early_stop],
        class_weight=class_weights
    )

    # Save the model to disk
    directory = os.path.join(config.path.apnea_model_directory, save_as)  # "apnea_model/save_as/"
    os.makedirs(directory, exist_ok=True)
    model_save_path = os.path.join(directory, "model.keras")  # "apnea_model/save_as/model.keras"
    basic_model.save(model_save_path)

    # Save the history to disk
    with open(os.path.join(directory, "history.pkl"), "wb") as f:  # "apnea-model/save_as/history.pkl"
        pickle.dump(history.history, f)


def evaluate_model(X_test, y_test, X_val, y_val, load_as: str, relevant_threshold_value, relevant_window_size):
    # Load model and training history
    model = load_model(os.path.join(config.path.apnea_model_directory, load_as, "model.keras"))  # "apnea_model/load_as/model.keras"
    with open(os.path.join(config.path.apnea_model_directory, load_as, "history.pkl"), "rb") as f:  # "apnea_model/load_as/history.pkl"
        history = pickle.load(f)

    # Basic evaluation
    loss_and_metrics = model.evaluate(X_test, y_test)
    print(loss_and_metrics)
    print('Loss = ', loss_and_metrics)

    # Plot training & validation loss
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # SAVE TRAINING LOSS PLOT
    save_dir = os.path.join(config.path.apnea_model_directory, load_as)
    os.makedirs(save_dir, exist_ok=True)
    loss_plot_path = os.path.join(save_dir, f"training_and_validation_loss.png")
    plt.savefig(loss_plot_path)
    plt.close()

    # Get predictions, performance metrics, and confusion matrix FOR VALIDATION
    val_predicted_probabilities = model.predict(X_val)
    val_predicted_probabilities = tf.squeeze(val_predicted_probabilities)
    val_actual = np.array(y_val)

    # Identify optimal threshold to use for considering a model output an apnea prediction (>0.5 confidence means
    # it's an apnea for example)
    best_f1, best_thresh = 0, 0
    for t in np.linspace(0.1, 0.9, 50):
        pred = tf.cast(val_predicted_probabilities > t, tf.int32)
        score = f1_score(val_actual, pred)
        if score > best_f1:
            best_f1, best_thresh = score, t

    print("Best Thresh:", best_thresh)
    print("Best F1 Score:", best_f1)
    best_thresh = 0.50

    # Get predictions, performance metrics, and confusion matrix for testing
    predicted_probabilities = model.predict(X_val)
    predicted_probabilities = tf.squeeze(predicted_probabilities)
    actual = np.array(y_val)

    predicted = np.array([1 if x >= best_thresh else 0 for x in predicted_probabilities])  # arbitrary threshold

    print(classification_report(actual, predicted))  # includes precision, recall, F1
    print("AUC-ROC:", roc_auc_score(actual, predicted_probabilities))
    print("MCC:", matthews_corrcoef(actual, predicted))

    # Compute raw confusion matrix and normalized one (row-wise)
    conf_mat_raw = confusion_matrix(actual, predicted)
    conf_mat_norm = confusion_matrix(actual, predicted, normalize='true')  # normalize by row

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
           title=f"{relevant_window_size} Window, {relevant_threshold_value} Threshold")

    # Rotate the tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate each cell with count and percentage
    fmt = '.1f'
    thresh = conf_mat_norm.max() / 2.

    for i in range(n_classes):
        for j in range(n_classes):
            count = conf_mat_raw[i, j]
            percent = conf_mat_norm[i, j] * 100
            ax.text(j, i, f"{count}\n({percent:.1f}%)",
                    ha="center", va="center",
                    color="white" if conf_mat_norm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.ioff()

    # SAVE PLOT instead of displaying
    save_dir = os.path.join(config.path.apnea_model_directory, load_as)
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, f"confusion_matrix.png")
    fig.savefig(plot_path)
    plt.close(fig)

    # Save calculated metrics
    metrics = {
        "best_threshold": float(best_thresh),
        "best_f1": float(best_f1),
        "roc_auc": float(roc_auc_score(actual, predicted_probabilities)),
        "mcc": float(matthews_corrcoef(actual, predicted)),
        "classification_report": classification_report(actual, predicted, output_dict=True)
    }

    metrics_path = os.path.join(save_dir, f"metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


def main(threshold, window_size, train_val_test_ratio: (float, float, float), model_name, evaluate_only=False):
    # Load data in appropriate proportions (randomize and split by files)
    full_data_path = os.path.join(config.path.EEG_ACC_features_labelled,
                                  f'EEG_ACC_features_labelled_ApneaThreshold{threshold}_ArousalThreshold{threshold}_window{window_size}/')
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(full_data_path, train_val_test_ratio)

    # Train the model
    if not evaluate_only:
        train_model(X_train, y_train, X_val, y_val, save_as=model_name)

    # Evaluate the model
    evaluate_model(X_test, y_test, X_val, y_val, load_as=model_name, relevant_threshold_value=threshold,
                   relevant_window_size=window_size)


if __name__ == "__main__":
    for thresh in [0.01, 0.25, 0.5, 0.75, 0.95]:
        for window in [1, 5, 10, 15, 20, 25, 30]:
            main(thresh, window, (0.70, 0, 0.15), evaluate_only=False, model_name=f"{thresh}_thresh_{window}s_window")

    # window = 30
    # thresh = 0.01
    # main(thresh, window, (0.70, 0, 0.15), evaluate_only=False, model_name=f"{thresh}_thresh_{window}s_window")