from __future__ import division

import string
import re

import pandas as pd
import numpy as np

from collections import Counter


def remove_punctuation(s):
    """see http://stackoverflow.com/questions/265960/best-way-to-\
    strip-punctuation-from-a-string-in-python"""
    table = string.maketrans("", "")
    return s.translate(table, string.punctuation)


def clean(x):
    x = remove_punctuation(x)
    return " ".join(re.split("\W+", x))


def createVocabs(df):
    spams = df[df["type"] == "spam"]["text"].apply(clean)
    hams = df[df["type"] == "ham"]["text"].apply(clean)

    ham_text = [x.lower() for x in (" ".join(hams.tolist())).split()]
    ham_vocab = Counter(ham_text)

    spam_text = [x.lower() for x in (" ".join(spams.tolist())).split()]
    spam_vocab = Counter(spam_text)

    total_vocab = spam_vocab + ham_vocab
    vocab_size = len(total_vocab)

    return ham_vocab, spam_vocab, vocab_size


def createPriors(df):
    prior_ham = np.sum(df["type"] == "ham")/len(df)
    prior_spam = np.sum(df["type"] == "spam")/len(df)
    return prior_ham, prior_spam


def naive(document):
    prob_spam = []
    prob_ham = []
    for word in document:
        tf_ham = ham_vocab[word] + 1
        tf_spam = spam_vocab[word] + 1

        est_ham = tf_ham / (ham_len + vocab_size)
        est_spam = tf_spam / (spam_len + vocab_size)

        prob_spam.append(est_spam)
        prob_ham.append(est_ham)
    if np.prod(prob_ham)*prior_ham > np.prod(prob_spam)*prior_spam:
        return "ham"
    else:
        return "spam"


def run_bayes(x):
    y = clean(x).split()
    return naive(y)

if __name__ == '__main__':
    df = pd.read_csv("SMSSpamCollection.csv", sep="\t", header=None,
                     names=["type", "text"])
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    test = df[~msk]

    prior_ham, prior_spam = createPriors(train)
    ham_vocab, spam_vocab, vocab_size = createVocabs(train)

    spam_len = sum(spam_vocab.values())
    ham_len = sum(ham_vocab.values())

    pred = test["text"].apply(run_bayes)

    pd.crosstab(pred, test["type"])
    acc = sum(pred == test["type"])/len(test)
    print "Accuracy is: %s%%" % round(acc*100, 2)
