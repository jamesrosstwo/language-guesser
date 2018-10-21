import matplotlib as plt
import tensorflow as tf
import numpy as np
import pandas as pd


def read_dataset():
    df = pd.read_csv("C:\\Users\\james\\Desktop\\language-predictor\\words.csv")
    X = df[df.columns[0]]
    y = df[df.columns[1]]
    # print(X)
read_dataset()
