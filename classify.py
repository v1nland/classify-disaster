# run as: python3 classify.py <model_name> "<tweet_data>"

import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from load_set import ConvertTweetFromApi


def Classify(model, words_per_tweet, tweet_type, tweet_data):
    # Model / data parameters
    num_classes = 2
    # input_shape = (WORDS_PER_TWEET, TWEET_DIM, 1)

    model = load_model("models/" + model + ".h5")

    print("Loading sets...")

    # (x_test, y_test) = LoadTestSetFromApi("earthquake_hurricane", WORDS_PER_TWEET)
    (x_test, y_test) = ConvertTweetFromApi(
        "earthquake_hurricane", words_per_tweet, tweet_type, tweet_data
    )

    print("Test set loaded successfully")

    # Scale images to the [0, 1] range
    x_test = x_test.astype("float32") / 255

    # Make sure all 40 images have shape (WORDS_PER_TWEET, TWEET_DIM, 1)
    x_test = np.expand_dims(x_test, -1)

    print("x_test shape:", x_test.shape)
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_test = keras.utils.to_categorical(y_test, num_classes)

    predicted_value = model.predict(x_test)

    if np.argmax(predicted_value[0]) == 0:
        return "earthquake"
    elif np.argmax(predicted_value[0]) == 1:
        return "hurricane"


def main():
    if len(sys.argv) < 3:
        print('correct usage: python3 classify.py <model_name> "<tweet_data>"')
        return

    MODEL = sys.argv[1]
    TWEET_DATA = sys.argv[2]
    # words per tweet must be equal to the trained model (in this case 10)
    WORDS_PER_TWEET = 10

    result = Classify(MODEL, WORDS_PER_TWEET, 0, TWEET_DATA)

    print("the tweet type is: " + result)


if __name__ == "__main__":
    main()