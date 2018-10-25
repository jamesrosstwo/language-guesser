import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

pad_length = 0


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
    global pad_length
    pad_length = len(max(data, key=len))
    word_data = keras.preprocessing.sequence.pad_sequences(data,
                                                           value=0,
                                                           padding='post',
                                                           maxlen=pad_length)
    return word_data


data_length = 10000
train_dataset = read_dataset("data/training.csv", data_length)
train_data = pad_data(train_dataset[0])
train_labels = train_dataset[1]

test_dataset = read_dataset("data/testing.csv", data_length)
test_data = pad_data(train_dataset[0])
test_labels = train_dataset[1]

model = keras.Sequential()
model.add(keras.layers.Embedding(pad_length, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = train_data[:data_length // 2]
partial_x_train = train_data[data_length // 2:]

y_val = keras.utils.to_categorical(train_labels[:data_length // 2])
partial_y_train = keras.utils.to_categorical(train_labels[data_length // 2:])


history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

results = model.evaluate(test_data, test_labels)

print(results)
