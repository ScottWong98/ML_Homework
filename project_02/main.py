import re
import string
from typing import Any

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB


def load_data():
    """ Load 20newsgroups dataset """
    dataset: Any = fetch_20newsgroups(
        subset="all", shuffle=True, remove=("headers", "footers", "quotes")
    )
    return dataset


def clean_data(dataset):
    """ Delete number and punctuation in dataset """
    # Use dataframe to quickly clean data
    df = pd.DataFrame({"data": dataset.data, "target": dataset.target})

    # delete number and punctuation
    alphanumeric = lambda x: re.sub(r"""\w*\d\w*""", " ", x)
    punc_lower = lambda x: re.sub(f"[{re.escape(string.punctuation)}]", " ", x.lower())

    df["data"] = df["data"].map(alphanumeric).map(punc_lower)
    return df


def vectorization(df):
    """ Convert text to vector using tf-idf """
    # delete stop words
    tfidf = TfidfVectorizer(stop_words="english")
    features = tfidf.fit_transform(df["data"])
    labels = df["target"].values
    return features, labels


def train_predict(features, labels):
    model = MultinomialNB()

    accuracy = cross_val_score(model, features, labels, cv=5, scoring="accuracy")
    recall = cross_val_score(model, features, labels, cv=5, scoring="recall_weighted")
    f1 = cross_val_score(model, features, labels, cv=5, scoring="f1_weighted")
    print(f"Accuracy: {accuracy.mean()}")
    print(f"Recall: {recall.mean()}")
    print(f"F1: {f1.mean()}")


def main():
    train_predict(*vectorization(clean_data(load_data())))


if __name__ == "__main__":
    main()
