import random

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
    c = 0
    for word in words:
        current_word = []
        c += 1
        try:
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
        except TypeError:
            print("Unreadable word at line " + str(c))
        out.append(current_word)
    for word_idx in range(len(out)):
        word = out[word_idx]
        one_hot_word = []
        for i in range(longest_word_len):
            one_hot_word.append([0] * letter_count)
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
        # tf.keras.layers.Embedding(num_letters, 50),
        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100)),
        tf.keras.layers.Dense(700, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(200, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
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
        loss="categorical_crossentropy",
        metrics=['accuracy'])
    m.fit(x, y, epochs=7)
    return m


def guess_words_from_data(model, x, y):
    num = 5
    w = x[:num]
    l = y[:num]
    for i in range(num):
        p = model.predict(np.asarray([w[i]]))
        d = decode_word(w[i])
        a = l[i]
        print("The model predicts:", d, "-", languages[p.argmax()] + ". The actual language is", languages[a.argmax()])


def guess_words_from_input(model):
    i = input("Enter a word \n")
    while i != "quit":
        one_hot_word = np.asarray([encode_user_word(i)])
        prediction = model.predict(one_hot_word)
        print(languages)
        print(prediction)
        print("The model predicts: ", i, "-", languages[prediction.argmax()])
        i = input()


def game(model, x, y):
    player_score = 0
    bot_score = 0
    idx = random.randrange(0, len(x))
    one_hot_word = x[idx]
    decoded_word = decode_word(one_hot_word)
    correct = languages[y[idx].argmax()]
    i = input(decoded_word)
    while i is not "quit":
        bot_guess = languages[model.predict(np.asarray([one_hot_word])).argmax()]
        print("You guessed", i, "while the bot guessed", bot_guess)
        print("The actual answer is", correct)
        if bot_guess == correct:
            bot_score += 1
        if i == correct:
            player_score += 1
        print("Your score:", player_score, " bot score:", bot_score)
        idx = random.randrange(0, len(x))
        one_hot_word = x[idx]
        decoded_word = decode_word(one_hot_word)
        correct = languages[y[idx].argmax()]
        i = input(decoded_word)


if __name__ == "__main__":
    data_length = 30000
    longest_word_len = 0
    languages = []
    map_to_char = {}
    char_to_map = {}

    # requested_dataset = input("Enter the dataset name to use \n")
    requested_dataset = "EasternEurope"
    p = "data/" + requested_dataset + ".csv"
    x_train, y_train, x_test, y_test = read_dataset(p, data_length)
    num_letters = len(map_to_char)

    print(languages)

    print(num_letters, "letters - longest word is", longest_word_len, "letters long")
    model = create_model(x_train, y_train)
    model.evaluate(x_test, y_test)

    # guess_words_from_input(model)
    game(model, x_train, y_train)
