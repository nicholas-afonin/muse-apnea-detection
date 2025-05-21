import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
import sklearn
import pandas as pd
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.metrics import Precision
# from tensorflow.python.keras.models import Sequential, load_model
# from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import classification_report, roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
import config
import random
import glob


""" -------------------------------------- HELPER FUNCS -------------------------------------- """


def combine_files_into_df(csv_path_list):
    """Take a list of paths to csv files and convert them all into one big dataframe"""
    dfs = [pd.read_csv(f) for f in csv_path_list]
    combined = pd.concat(dfs, ignore_index=True)

    return combined


""" -------------------------------------- CORE FUNCS -------------------------------------- """


def load_data(processed_features_directory, train_test_ratio):
    all_csv_file_paths = glob.glob(processed_features_directory + "*.csv")

    # Shuffle the list (for randomness, but reproducible with a seed)
    random.seed(21)
    random.shuffle(all_csv_file_paths)

    # Split into train and test
    split_index = int(len(all_csv_file_paths) * train_test_ratio)
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
    X_train = train_df.drop(columns=['ts', 'name'])
    X_test = test_df.drop(columns=['ts', 'name'])

    y_train = X_train.pop('apnea_event')
    y_test = X_test.pop('apnea_event')

    # You already have X_train, y_train
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.15,
        stratify=y_train,
        random_state=42
    )

    # Oversample the minority classes in the training data to give more to train on
    smote = SMOTE()
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Debugging
    # print("Unique labels in y_train:", y_train.unique())
    # print("Any NaNs?", y_train.isna().sum())
    # print(X_train.shape)
    # print(X_train.describe())
    # assert not np.isnan(X_train.values).any(), "NaNs in X_train!"
    # assert not np.isinf(X_train.values).any(), "Infs in X_train!"

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train, y_train, X_val, y_val):
    # Create model architecture
    basic_model = Sequential()  # initialize a basic sequential model
    basic_model.add(Dense(units=128, activation='relu', input_shape=(167,), kernel_regularizer=l2(0.01)))  # input layer
    basic_model.add(Dropout(0.4))  # to prevent overfitting
    basic_model.add(Dense(64, activation='tanh', kernel_regularizer=l2(0.01)))
    basic_model.add(Dropout(0.4))
    basic_model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
    basic_model.add(Dropout(0.4))
    basic_model.add(Dense(1, activation='sigmoid'))

    adam = keras.optimizers.Adam(learning_rate=0.00005)
    basic_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy", keras.metrics.Precision()])

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
        epochs=300,
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=[early_stop],
        class_weight=class_weights
    )

    return basic_model, history


def evaluate_model(model, history, X_test, y_test):
    # Basic evaluation
    loss_and_metrics = model.evaluate(X_test, y_test)
    print(loss_and_metrics)
    print('Loss = ', loss_and_metrics[0])
    print('Accuracy = ', loss_and_metrics[1])

    # Plot training & validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Get predictions, performance metrics, and confusion matrix
    predicted_probabilities = model.predict(X_test)
    predicted_probabilities = tf.squeeze(predicted_probabilities)
    predicted = np.array([1 if x >= 0.2 else 0 for x in predicted_probabilities])  # arbitrary threshold
    actual = np.array(y_test)

    print(classification_report(actual, predicted))  # includes precision, recall, F1
    print("AUC-ROC:", roc_auc_score(actual, predicted_probabilities))
    print("MCC:", matthews_corrcoef(actual, predicted))

    conf_mat = confusion_matrix(actual, predicted)
    labels = ["No Apnea", "Apnea"]

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=labels)
    disp.plot()

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(config.path.EEG_ACC_features_labelled, 0.75)
    model, history = train_model(X_train, y_train, X_val, y_val)
    evaluate_model(model, history, X_test, y_test)

