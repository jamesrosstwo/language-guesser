import matplotlib as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def read_dataset():
    df = pd.read_csv("C:\\Users\\james\\Desktop\\language-predictor\\words.csv")
    X = df[df.columns[0]].values
    y = df[df.columns[1]]

    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encode(y)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    out = np.zeros((n_labels, n_unique_labels))
    out[np.arange(n_labels), labels] = 1
    return out

read_dataset()
