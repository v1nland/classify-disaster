import json
import requests
from tensorflow import keras

import numpy as np


def LoadSetFromFile(parsed_train_set_file_name, num_classes):
    print("Loading set: " + parsed_train_set_file_name + "\n")
    train_file = open(
        "../word-vectorization/" + parsed_train_set_file_name,
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

    # convert to np array
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255

    # Make sure all 40 images have shape (WORDS_PER_TWEET, TWEET_DIM, 1)
    x_train = np.expand_dims(x_train, -1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)

    return (x_train, y_train)


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
