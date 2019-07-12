import os
import random


def get_path_from_request(req):
    global sources
    lang_name = req
    if req.isdigit():
        lang_name = sources[int(req)]
    return "data/sources/" + lang_name + ".txt", lang_name


def write_lang(lang_path, lang_name, out_file):
    with open(lang_path) as source:
        global num_words
        for n in range(num_words):
            line = source.readline()[:-1]
            if line == "":
                continue
            line += ", " + lang_name
            out_file.write(line + "\n")


def scramble_langs(o_path):
    with open(o_path, mode="r+") as o_file:
        lines = o_file.readlines()
        o_file.seek(0)
        random.shuffle(lines)
        o_file.writelines(lines)


def write_langs(langs, o_path):
    try:
        with open(o_path, mode='x') as out_file:
            for l in langs:
                lang_path, lang_name = get_path_from_request(l)
                write_lang(lang_path, lang_name, out_file)
        scramble_langs(o_path)
    except FileExistsError:
        print("Cannot write to this file, file already exists.")


if __name__ == "__main__":
    sources = [x[:-4] for x in os.listdir("data/sources")]
    for i, s in enumerate(sources):
        print(i, ":", s)
    requested_langs = input("Enter the languages you would like to use, separated by a space. \n\n").split(" ")
    num_words = int(input("Enter the number of words per language (1-10000)"))
    out_path = "data/" + input("Enter the output file name") + ".csv"
    write_langs(requested_langs, out_path)
