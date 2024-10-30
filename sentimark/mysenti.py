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

# from hunspell import Hunspell
import argparse
import os


def clearfiles():
    data = pd.read_csv(dataset_path, encoding="utf-8")

    data = data.drop("topic", axis=1)
    data = data.drop("title", axis=1)

    data = data.dropna()
    data = data.drop_duplicates(subset=["comment"], keep="first")
    temp = []
    temp = data["stars"].values.tolist()

    file_name = file_name.replace("dirty", "")
    dataset_path = f"../neuralnet/preprocessed_datasets/{file_name}_{mode}.csv"

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
        dataset_path,
        header=["reviews", "sentiment"],
        index=False,
        encoding="utf-8",
    )

    return file_name, dataset_path


def splitfiles():
    data = pd.read_csv(dataset_path, encoding="utf-8")

    stars_path = f"{dir_path}/{file_name}_stars_{mode}.csv"
    reviews_path = f"{dir_path}/{file_name}_reviews_{mode}.csv"

    data["sentiment"].to_csv(stars_path, header=["sentiment"], index=False)

    data["reviews"].to_csv(
        reviews_path,
        header=["reviews"],
        index=False,
        encoding="utf-8",
    )

    return stars_path, reviews_path


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


# Define command-line arguments
parser = argparse.ArgumentParser(
    description="Run the sentimark algorithm to detect sentiment in greek text."
)
parser.add_argument(
    "--mode",
    type=str,
    required=True,
    choices=["bin", "nonbin"],
    help="Mode of the analysis: nonbin or bin",
)
parser.add_argument(
    "--file_name", type=str, required=True, help="Name of file to preprocess"
)
args = parser.parse_args()

# Access the mode arguments
mode = args.mode
file_name = args.file_name
dataset_path = f"../neuralnet/preprocessed_datasets/{file_name}_{mode}.csv"

# Hunspell check
# h = Hunspell("el_GR")
# if not a new .csv is downloaded and in folder
# clear it and fix it
if "dirty" in file_name:
    file_name, dataset_path = clearfiles()
    print("Cleared")

# Create directories
dir_path = f"{file_name}_{mode}"
os.makedirs(dir_path, exist_ok=True)

# run split to have both reviews and stars .csv
stars_path, reviews_path = splitfiles()

with open(reviews_path, newline="\n", encoding="utf-8") as f:
    df = csv.reader(f)
    df = list(df)
    df = list(filter(None, df))  # list of reviews with no duplicates

with open(stars_path, newline="\n") as g:
    stt = []
    for row in csv.reader(g, delimiter=";"):

        stt.append(row[0])  # stars array

# pharm lexicon
with open(
    "../finallexformysenti/EmotionLookupTable.txt", "r", encoding="utf-8"
) as file:
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
with open(
    "../finallexformysenti/EmoticonLookupTable.txt", "r", encoding="utf-8"
) as file:
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
with open("../finallexformysenti/BoosterWordList.txt", "r", encoding="utf-8") as file:
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
with open("../finallexformysenti/NegatingWordList.txt", "r", encoding="utf-8") as file:
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
checkedWords = 0  # number of words that were checked
totalWords = 0  # sum of words


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
mysenti_predictions_path = f"{dir_path}/mysenti_{file_name}_{mode}.csv"
with open(
    mysenti_predictions_path, "w", newline="", encoding="utf8"
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
            totalWords = totalWords + 1  # count words
            words = clean_accent(words)  # clean accent of word

            # emoticon first before any stiksh split so not to lose
            if words in emot:
                checkedWords = checkedWords + 1  # word find counter
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

                        # k = h.suggest(words)
                        # if k != ():
                        #     words = k[0]
                        # break

                # Negative word check. If found flag=True and next word emotion skipped
                if words in neg:
                    checkedWords = checkedWords + 1
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
                        checkedWords = checkedWords + 1  # word counter
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
                    checkedWords = checkedWords + 1  # word counter
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
        "Words found in lexicon: ", checkedWords, " Total words: ", totalWords
    )  # words found,total words
    ratio = checkedWords / totalWords
    print("\n Ratio: ", ratio)  # ratio found

    t = [df, summinmax, mins, maxs, stt]  # exported data
    export_data = zip_longest(*t)  # zip and write
    writer.writerows(export_data)

# Prediction accuracy
df = pd.read_csv(mysenti_predictions_path)

res = []
sent = []

sent = df["sentiment"]
res = df["mysentiment"]

cnt = 0
for i in range(1, len(res) - 1):
    if int(res[i]) == int(sent[i]):
        cnt = cnt + 1

accuracy = cnt / len(summinmax) * 100
print("Correct Predicted: ", cnt, " = ", accuracy, "%")

# CSV file to save results
results_csv_path = "results.csv"

# Initialize results DataFrame if the file doesn't exist
if not os.path.exists(results_csv_path):
    results_df = pd.DataFrame(
        columns=[
            "Dataset",
            "Accuracy",
            "Percentage of words found in lexicon",
            "Total Words",
            "Words Found",
            "Correct Predicted",
            "Total Predicted",
        ]
    )
    results_df.to_csv(results_csv_path, index=False)

# Save model results to a CSV
results = {
    "Model Name": f"{file_name}_{mode}",
    "Accuracy": accuracy / 100,
    "Percentage of words found in lexicon": ratio,
    "Total Words": totalWords,
    "Words Found": checkedWords,
    "Correct Predicted": cnt,
    "Total Predicted": len(summinmax),
}

results_df = pd.DataFrame([results])
results_df.to_csv(
    results_csv_path, mode="a", header=False, index=False
)  # Append results
