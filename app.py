"""
UIC CS 411 - Project 03: Naive Bayes Text Classifier from Scratch
"""

import os
import csv
import math
import re


# stuff the model learns during training
vocab = set()
word_count = {"positive": {}, "negative": {}}
total_words = {"positive": 0, "negative": 0}
sentence_count = {"positive": 0, "negative": 0}
prior = {"positive": 0.0, "negative": 0.0}
is_trained = False

# punctuation to remove
PUNCTUATION = ".,!?;:\"()[]{}<>/\\|-_*&^%$#@~`+=“”…"


def get_path(file_name):
    if os.path.exists(file_name):
        return file_name

    folder = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(folder, file_name)


def read_csv(file_name):
    data = []

    f = open(get_path(file_name), "r", encoding="utf-8-sig", newline="")
    reader = csv.DictReader(f)

    for row in reader:
        text = (row.get("text") or "").strip()
        label = (row.get("label") or "").strip().lower()

        if text != "" and (label == "positive" or label == "negative"):
            data.append((text, label))

    f.close()
    return data


def tokenize(text):
    text = text.lower()
    text = text.replace("’", "'")

    for p in PUNCTUATION:
        text = text.replace(p, " ")

    words = []
    for w in text.split():
        w = w.strip("'")
        if w != "":
            words.append(w)

    return words


def train_model(train_file="data/train.csv"):
    """
    Train the Naive Bayes classifier on the training dataset.
    Builds the vocabulary, word counts per class, and class priors.
    """
    global vocab, word_count, total_words, sentence_count, prior, is_trained

    # reset everything so retraining starts fresh
    vocab = set()
    word_count = {"positive": {}, "negative": {}}
    total_words = {"positive": 0, "negative": 0}
    sentence_count = {"positive": 0, "negative": 0}

    data = read_csv(train_file)

    for text, label in data:
        sentence_count[label] += 1

        for word in tokenize(text):
            vocab.add(word)

            if word in word_count[label]:
                word_count[label][word] += 1
            else:
                word_count[label][word] = 1

            total_words[label] += 1

    total_sentences = len(data)

    for label in ["positive", "negative"]:
        if total_sentences > 0:
            prior[label] = sentence_count[label] / total_sentences
        else:
            prior[label] = 0.0

    is_trained = True


def get_prob(word, label):
    
    count = word_count[label].get(word, 0)
    return (count + 1) / (total_words[label] + len(vocab))


def get_scores(text):
    if not is_trained:
        train_model()

    words = tokenize(text)
    scores = {}

    for label in ["positive", "negative"]:
        if prior[label] > 0:
            score = math.log(prior[label])
        else:
            score = float("-inf")

        for word in words:
           
            if word in vocab:
                score += math.log(get_prob(word, label))

        scores[label] = score

    return scores


def predict(text):
    """
    Predict sentiment class for an input sentence.
    Returns: 'positive' or 'negative'
    Tie-breaking: If scores are tied, select the class with the higher prior P(c),
                  or default to 'positive' if priors are equal.
    """
    scores = get_scores(text)
    pos = scores["positive"]
    neg = scores["negative"]

    if pos > neg:
        return "positive"
    elif neg > pos:
        return "negative"

    if prior["negative"] > prior["positive"]:
        return "negative"
    return "positive"


def evaluate_model(test_file="data/test.csv"):
   
    if not is_trained:
        train_model()

    data = read_csv(test_file)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    wrong = []

    for text, real_label in data:
        guess = predict(text)

        if guess == "positive" and real_label == "positive":
            TP += 1
        elif guess == "negative" and real_label == "negative":
            TN += 1
        elif guess == "positive" and real_label == "negative":
            FP += 1
        else:
            FN += 1

        if guess != real_label:
            wrong.append((text, real_label, guess))

    if len(data) > 0:
        accuracy = (TP + TN) / len(data)
    else:
        accuracy = 0.0

    return {
        "accuracy": accuracy,
        "confusion_matrix": {"TP": TP, "TN": TN, "FP": FP, "FN": FN},
        "wrong": wrong,
    }


def show_model_info():
    print("TRAINING INFO")
    print("Vocabulary size:", len(vocab))

    for label in ["positive", "negative"]:
        print(label + ": sentences =", sentence_count[label],
              ", N_c =", total_words[label],
              ", P(c) =", round(prior[label], 4))


def show_results(result):
    cm = result["confusion_matrix"]

    print("\nEVALUATION")
    print("Accuracy:", round(result["accuracy"], 4))
    print("Confusion Matrix:", "TP =", cm["TP"], " TN =", cm["TN"],
          " FP =", cm["FP"], " FN =", cm["FN"])


def show_steps(text):
    words = tokenize(text)
    unknown = [w for w in words if w not in vocab]
    scores = get_scores(text)

    print("Tokens:", words)
    print("Ignored (not in vocab):", unknown)

    for w in words:
        if w in vocab:
            print("  P(" + w + "|pos) =", round(get_prob(w, "positive"), 6),
                  "  P(" + w + "|neg) =", round(get_prob(w, "negative"), 6))

    print("Score positive:", round(scores["positive"], 4))
    print("Score negative:", round(scores["negative"], 4))
    print("Prediction:", predict(text))


if __name__ == "__main__":
    train_model()
    show_model_info()

    try:
        show_results(evaluate_model())
    except FileNotFoundError:
        print("test.csv not found")

    print("\nEnter a sentence (type quit to exit)")

    while True:
        try:
            sentence = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if sentence.strip().lower() in ["quit", "exit", "q"]:
            break

        if sentence.strip() != "":
            show_steps(sentence)
            print()