import numpy as np
import matplotlib as plt
import tensorflow as tf
import tensorboard as tb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras


def read_dataset(filename, lines):
    df = pd.read_csv(filename, nrows=lines)
    x = df[df.columns[0]].values
    y = df[df.columns[1]]

    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encode(y)
    X = []
    for i in x:
        i = [ord(c) for c in i]
        X.append(i)
    return [np.array(X), np.array(Y)]


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    out = np.zeros((n_labels, n_unique_labels))
    out[np.arange(n_labels), labels] = 1
    return out


def pad_data(data):
    max_len = len(max(data, key=len))
    data = keras.preprocessing.sequence.pad_sequences(data,
                                                      value=0,
                                                      padding='post',
                                                      maxlen=max_len)
    return data


train_data = pad_data(read_dataset("data/training.csv", 10000)[0])

