# IMPORTANT for cyhunspell (ubuntu-wsl):
# gotta get the greek dictionaries and put them in the hunspell dics
# in this location ~/.local/lib/python3.10/site-packages/hunspell/dictionaries
# and from terminal do "nautilus ." to open it in GUI and copy-paste dics
# required for cyhunspell (ubuntu-wsl)
# apt install -y  autoconf libtool  gettext autopoint
# pip install https://github.com/MSeal/cython_hunspell/archive/refs/tags/2.0.3.tar.gz

from sentistrength import PySentiStr
import pandas as pd
import csv
from itertools import zip_longest
from difflib import SequenceMatcher
from hunspell import Hunspell
import os

mode = "bin"


def clearfiles(mode):
    data = pd.read_csv("dataset/dirtyreviews.csv")

    data = data.drop("topic", axis=1)
    data = data.drop("title", axis=1)

    data = data.dropna()
    data = data.drop_duplicates(subset=["comment"], keep="first")
    temp = []
    temp = data["stars"].values.tolist()
    name = "reviewstars" + mode
    if mode == "bin":
        for i in range(0, len(data["stars"])):

            if int(temp[i]) <= 3:
                temp[i] = 0
            else:
                temp[i] = 1

    else:
        for i in range(0, len(data["stars"])):

            if int(temp[i]) <= 2:
                temp[i] = -1
            elif int(temp[i]) == 3:
                temp[i] = 0
            else:
                temp[i] = 1
    data["stars"] = temp

    cols = data.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    data = data[cols]

    data.to_csv(
        "dataset/" + name + ".csv",
        header=["reviews", "sentiment"],
        index=False,
        encoding="utf-8",
    )


def splitfiles(mode):
    if mode == "bin":
        data = pd.read_csv("dataset/reviewstarsbin.csv")

        data["sentiment"].to_csv("starsbin.csv", header=["sentiment"], index=False)

        data["reviews"].to_csv(
            "reviews.csv", header=["reviews"], index=False, encoding="utf-8"
        )
    else:
        data = pd.read_csv("dataset/reviewstarsnonbin.csv")

        data["sentiment"].to_csv("starsnonbin.csv", header=["sentiment"], index=False)

        data["reviews"].to_csv(
            "reviews.csv", header=["reviews"], index=False, encoding="utf-8"
        )


def clean_accent(text):

    t = text

    # el
    t = t.replace("Ά", "Α")
    t = t.replace("Έ", "Ε")
    t = t.replace("Ί", "Ι")
    t = t.replace("Ή", "Η")
    t = t.replace("Ύ", "Υ")
    t = t.replace("Ό", "Ο")
    t = t.replace("Ώ", "Ω")
    t = t.replace("ά", "α")
    t = t.replace("έ", "ε")
    t = t.replace("ί", "ι")
    t = t.replace("ή", "η")
    t = t.replace("ύ", "υ")
    t = t.replace("ό", "ο")
    t = t.replace("ώ", "ω")
    t = t.replace("ς", "σ")
    t = t.replace("♡", "")
    t = t.replace("☆", "")
    t = t.replace("*", "")

    return t


def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros


# Hunspell check
h = Hunspell("el_GR")
# if not a new .csv is downloaded and in folder
# clear it and fix it
if not (os.path.isfile("dataset/reviewstars" + mode + ".csv")):
    clearfiles(mode)
    print("Cleared")
# run split to have both reviews and stars .csv
splitfiles(mode)
if mode == "nonbin":
    # File with reviews
    file_name = "reviews.csv"
    stars_name = "starsnonbin.csv"
else:
    file_name = "reviews.csv"
    stars_name = "starsbin.csv"

with open(file_name, newline="\n", encoding="utf-8") as f:
    df = csv.reader(f)
    df = list(df)
    df = list(filter(None, df))  # list of reviews with no duplicates
with open(stars_name, newline="\n") as g:
    stt = []
    for row in csv.reader(g, delimiter=";"):

        stt.append(row[0])  # stars array


# pharm lexicon
with open("finallexformysenti/EmotionLookupTable.txt", "r", encoding="utf-8") as file:
    terms_list = file.read().splitlines()

word = []  # 2 arrays for word and score
score = []

for t in terms_list:
    t = t.split("	")
    word.append(t[0])
    score.append(t[1])

for i in range(0, len(score)):
    score[i] = int(score[i])  # make int from string

for i in range(0, len(word)):
    word[i] = clean_accent(word[i].lower())  # clean accent of word


# emoticontable same as pharm
with open("finallexformysenti/EmoticonLookupTable.txt", "r") as file:
    emotic_list = file.read().splitlines()
emot = []
scorem = []
for te in emotic_list:
    te = te.split("	")
    emot.append(te[0])
    scorem.append(te[1])
for i in range(0, len(scorem)):
    scorem[i] = int(scorem[i])


# boosterwords same as before
with open("finallexformysenti/BoosterWordList.txt", "r", encoding="utf-8") as file:
    terms_listbo = file.read().splitlines()

boost = []
scorebo = []

for tb in terms_listbo:
    tb = tb.split("	")
    boost.append(tb[0])
    scorebo.append(tb[1])
for i in range(0, len(scorebo)):
    scorebo[i] = int(scorebo[i])
for i in range(0, len(boost)):
    boost[i] = clean_accent(boost[i].lower())

# negwords
with open("finallexformysenti/NegatingWordList.txt", "r", encoding="utf-8") as file:
    terms_listneg = file.read().splitlines()
neg = []
for tn in terms_listneg:
    tn = tn.split("	")
    neg.append(tn[0])
for i in range(0, len(neg)):
    neg[i] = clean_accent(neg[i].lower())


# Constants declarations
suffix_prune_el = 3  # prune in words
string_min_score = 0.76  # matching score
kek = 0  # number of words that were checked
lel = 0  # sum of words


scorerev = [0]  # score per review
mins = [-1]  # min score per review
maxs = [1]  # max score per review
i = 0  # an i
stikshh = [
    ".",
    " ",
    "-",
    "_",
    "+",
    "w",
    "°",
    "?",
    ";",
    "!",
    ":",
    "(",
    ")",
]  # unwanted chars
stiksh = [
    ".",
    " ",
    "-",
    "_",
    "+",
    "w",
    "°",
    "?",
    ";",
    "!",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]  # unwanted chars that may repeat
summinmax = [0]
with open(
    "dataset/finalgreekmysenti" + mode + ".csv", "w", newline="", encoding="utf8"
) as f:  # results csv
    writer = csv.writer(f, delimiter=",")
    writer.writerow(("review", "mysentiment", "min", "max", "sentiment"))  # row titles
    for review in df:  # every review

        review = [
            x.replace("\n", "") for x in review
        ]  # bgazw to /n pou ebale to opencsv

        flag = False  # kathe review arxikopoiw false. An ginei true meta h epomenh leksh pou brisketai den metrietai

        rvwords = review[0].split(" ")  # kathe leksh pou exei to review

        rvwords = list(rvwords)  # list

        for words in rvwords:
            sr = 0  # sr start every word
            lel = lel + 1  # count words
            words = clean_accent(words)  # clean accent of word

            # emoticon first before any stiksh split so not to lose
            if words in emot:
                kek = kek + 1  # word find counter
                sr = scorem[emot.index(words)]
                scorerev[i] = scorerev[i] + sr  # if found adds score to review score
            else:

                # punctuation if no emoticon found

                a = [""]  # starts a dummy array to see if there is a !
                if "!" in words:
                    a = words.split(
                        "!"
                    )  # word is spliting from !. After this algorithm
                    # cant find ! and word remains the same without !
                    # so I can add word's score with ! boost

                for p in range(0, len(words)):
                    if (
                        words[p : p + 1] in stikshh
                    ):  # replacing every weird char with '' so word can be clear
                        words = words.replace(words[p : p + 1], "")
                        words = words.replace(".", "")

                # threepeat letters checker and hunspell sugestion after removing them.
                # Tested and gives good suggestions. Check also that word is not a punctuation or number
                k = [""]
                for p in range(3, len(words)):

                    if words[p - 1 : p] == words[p - 2 : p - 1] == words[
                        p - 3 : p - 2
                    ] and (words[p - 1 : p] not in stiksh):
                        words = "".join(sorted(set(words), key=words.index))
                        # print(words)

                        k = h.suggest(words)
                        if k != ():
                            words = k[0]
                        break

                # Negative word check. If found flag=True and next word emotion skipped
                if words in neg:
                    kek = kek + 1
                    flag = True

                # main list check and scoring
                # get words that start with the first letter of word that we check
                # saves A LOT of time
                for wrd in [m for m in word if m.lower().startswith(words[:1])]:
                    match = words.find(
                        wrd[: max(3, len(wrd) - suffix_prune_el)]
                    )  # match word with pruning
                    scorera = SequenceMatcher(
                        None, words, wrd
                    ).ratio()  # ratio of final matching
                    if match == 0 and scorera > string_min_score:  # match and ratio>
                        kek = kek + 1  # word counter
                        if flag == True:
                            flag = False  # if flag=True do it false and stop
                        else:
                            sr = score[word.index(wrd)]  # found score of word
                            if a[0] != "":  # If ! found
                                if sr == -1:  # score of word from -1->2
                                    sr = 2
                                else:
                                    sr = sr + 1  # other score of word +1

                            scorerev[i] = scorerev[i] + sr  # sum score of review
                # if words in boost add in score
                if words in boost:
                    kek = kek + 1  # word counter
                    sr = scorebo[boost.index(words)]
                    scorerev[i] = scorerev[i] + sr
            # check for max review score	until this word in every case se is the added score
            # from word
            if sr > maxs[i]:
                maxs[i] = sr
            # check for min review score	until this word
            if sr < mins[i]:
                mins[i] = sr

        # add min and max to produce the final score and label
        # -1 if neg, 0 if neutr, 1 if positive
        summinmax[i] = maxs[i] + mins[i]
        if summinmax[i] <= 0:
            summinmax[i] = 0

        # elif -1 <summinmax[i]<1:

        # 	summinmax[i]=0
        else:
            summinmax[i] = 1

        i = i + 1
        summinmax.append(0)
        scorerev.append(0)
        mins.append(-1)
        maxs.append(1)

    print(
        "Words found in lexicon: ", kek, " Total words: ", lel
    )  # words found,total words
    print("\n Ratio: ", kek / lel)  # ratio found

    t = [df, summinmax, mins, maxs, stt]  # exported data
    export_data = zip_longest(*t)  # zip and write
    writer.writerows(export_data)

# Prediction accuracy
dataset = "dataset/finalgreekmysenti" + mode + ".csv"

df = pd.read_csv(dataset)

res = []
sent = []

sent = df["sentiment"]
res = df["mysentiment"]

cnt = 0
for i in range(1, len(res) - 1):
    if int(res[i]) == int(sent[i]):
        cnt = cnt + 1
print("Correct Predicted: ", cnt, " = ", cnt / len(summinmax) * 100, "%")
