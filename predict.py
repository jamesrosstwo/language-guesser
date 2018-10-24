import matplotlib as plt
import tensorflow as tf
import tensorboard as tb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras


def read_dataset(filename):
    df = pd.read_csv(filename)
    x = df[df.columns[0]].values
    y = df[df.columns[1]]

    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encode(y)
    for i in x:
        i = [ord(c) for c in i]
    encoder.fit(x)
    x = encoder.transform(x)
    X = one_hot_encode(x)
    return [X, Y]


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    out = np.zeros((n_labels, n_unique_labels))
    out[np.arange(n_labels), labels] = 1
    return out

print(read_dataset("testing.csv"))