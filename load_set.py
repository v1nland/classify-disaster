import json
from fastapi.params import Body
import requests

import numpy as np


def LoadSetFromFile(parsed_train_set_file_name):
    train_file = open(
        "../word-vectorization/" + parsed_train_set_file_name + ".parsed_train_set",
        "r",
    )

    y_train = []
    x_train = []

    for line in train_file:
        tweet = json.loads(line)

        y_train.append(tweet["type"])

        tweet_vectors = []

        for word in tweet["data"]:
            tweet_vectors.append(word["vector"])

        x_train.append(np.array(tweet_vectors))

    return (np.array(x_train), np.array(y_train))


def LoadTestSetFromApi(model_name, dim):
    response = requests.get(
        "http://localhost:8000/models/{}/test_set?dim={}".format(model_name, dim)
    )
    body = response.json()

    y_train = [tweet["type"] for tweet in body["response"]]
    x_train = []

    for tweet in body["response"]:
        tweet_vectors = []

        for word in tweet["data"]:
            tweet_vectors.append(word["vector"])

        x_train.append(np.array(tweet_vectors))

    return (np.array(x_train), np.array(y_train))


def ConvertTweetFromApi(model_name, dim, tweet_type, tweet_data):
    response = requests.post(
        "http://localhost:8000/models/{}/tweet?dim={}".format(model_name, dim),
        json={"tweet_type": tweet_type, "tweet_data": tweet_data},
    )
    body = response.json()

    y_train = [body["type"]]
    x_train = []

    tweet_vectors = []

    for word in body["data"]:
        tweet_vectors.append(word["vector"])

    x_train.append(np.array(tweet_vectors))

    return (np.array(x_train), np.array(y_train))
