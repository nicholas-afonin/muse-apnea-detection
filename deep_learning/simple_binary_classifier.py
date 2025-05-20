import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
import pandas as pd
import keras
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
import tensorflow as tf
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
import config
import glob



# Load and clean the data
all_csv_file_paths = glob.glob(config.path.EEG_ACC_features_labelled + "*.csv")

print(all_csv_file_paths)



# Split dataset
# train_prop, val_prop, test_prop = train_val_test_prop
# seed = 2
# self.training, self.testing = train_test_split(self.files,
#                                                train_size=train_prop,
#                                                random_state=seed)
# self.validation, self.testing = train_test_split(self.testing,
#                                                  train_size=val_prop / (val_prop + test_prop),
#                                                  random_state=seed)
# self.files = self.training

# torch.save({"training": self.training, "validation": self.validation, "testing": self.testing}, MODELS + "/new_data_split.pth")


# X = df.drop(columns='label')
# y = df['label']
#
# df = pd.read_csv(config.path.EEG_ACC_features_labelled)
#
#
# df.head()



def train_model(X_train, X_test, y_train, y_test, model_path):
    pass


def evaluate_model():
    pass


if __name__ == "__main__":
    train_model()
    evaluate_model()