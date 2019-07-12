import tensorflow as tf
import numpy as np
import pandas as pd


def read_dataset(filename, lines):
    train_test_ratio = 0.8
    df = pd.read_csv(filename, nrows=lines)
    x_matrix = df[df.columns[0]].values
    x_vals = encode_words(x_matrix)
    train_x = x_vals[:int(lines * train_test_ratio)]
    test_x = x_vals[int(lines * train_test_ratio):]

    y_matrix = df[df.columns[1]].values
    y_vals = encode_langs(y_matrix)
    train_y = y_vals[:int(lines * train_test_ratio)]
    test_y = y_vals[int(lines * train_test_ratio):]

    print("Finished reading", filename)
    return np.asarray(train_x), np.asarray(train_y), np.asarray(test_x), np.asarray(test_y)


def decode_word(one_hot_word):
    word = ""
    for one_hot_char in one_hot_word:
        for i in range(len(one_hot_char)):
            if one_hot_char[i] == 1:
                word += chr(map_to_char[i])
                break
    return word


def encode_words(words):
    letter_count = 0
    global longest_word_len
    global char_to_map
    global map_to_char
    out = []
    for word in words:
        current_word = []
        for letter in word:
            if len(word) > longest_word_len:
                longest_word_len = len(word)
            if ord(letter) not in char_to_map:
                char_to_map[ord(letter)] = letter_count
                map_to_char[letter_count] = ord(letter)
                current_word.append(letter_count)
                letter_count += 1
            else:
                current_word.append(char_to_map[ord(letter)])
        out.append(current_word)
    for word_idx in range(len(out)):
        word = out[word_idx]
        one_hot_word = [[0] * letter_count] * longest_word_len
        for i in range(len(word)):
            one_hot_word[i][word[i]] = 1
        out[word_idx] = one_hot_word
    return out


def encode_langs(arr):
    count = 0
    labels = {}
    out = []
    for item in arr:
        if item not in labels:
            labels[item] = count
            languages.append(item)
            count += 1
        out.append(labels[item])
    l = len(labels)
    for i in range(len(out)):
        a = [0] * l
        a[out[i]] = 1
        out[i] = a
    return out


def encode_user_word(word):
    global num_letters
    global longest_word_len
    out = []
    for i in range(longest_word_len):
        out.append([0] * num_letters)
    for idx, c in enumerate(word):
        c_code = ord(c)
        if c_code not in char_to_map:
            print("Character", c, "not recognized")
            continue  # skip unrecognized chars
        m = char_to_map[c_code]
        out[idx][m] = 1
    return out


def create_layers(i_shape):
    layers = [
        tf.keras.layers.Flatten(input_shape=i_shape),
        tf.keras.layers.Dense(700, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(languages), activation=tf.nn.softmax)
    ]
    return layers


def create_model(x, y):
    global longest_word_len
    global num_letters
    layers = create_layers((longest_word_len, num_letters))
    m = tf.keras.models.Sequential()
    for layer in layers:
        m.add(layer)
    m.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    m.fit(x, y, epochs=6)
    return m


def guess_words(model):
    i = input()
    while i != "quit":
        one_hot_word = encode_user_word(i)
        print(one_hot_word)
        print(decode_word(one_hot_word))
        prediction = model.predict([[one_hot_word]]).argmax()
        print("The model predicts", languages[prediction])

        i = input()


if __name__ == "__main__":
    data_length = 10000
    longest_word_len = 0
    languages = []
    map_to_char = {}
    char_to_map = {}

    x_train, y_train, x_test, y_test = read_dataset("data/training.csv", data_length)
    num_letters = len(map_to_char)

    print(num_letters, longest_word_len)
    print(x_train.shape)
    print(x_test.shape)
    model = create_model(x_train, y_train)
    model.evaluate(x_test, y_test)

    guess_words(model)
