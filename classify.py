import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from load_set import LoadTestSetFromApi, ConvertTweetFromApi

TWEET_DIM = 200
WORDS_PER_TWEET = 10

# Model / data parameters
num_classes = 2
input_shape = (WORDS_PER_TWEET, TWEET_DIM, 1)

model = load_model("models/earthquake_hurricane.h5")

print("Loading sets...")

# (x_test, y_test) = LoadTestSetFromApi("earthquake_hurricane", WORDS_PER_TWEET)
(x_test, y_test) = ConvertTweetFromApi(
    "earthquake_hurricane", WORDS_PER_TWEET, 0, "the glasses are falling down"
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

print("(0=eq, 1=hu) input is of type: " + str(np.argmax(predicted_value[0])))

# # Evaluate the trained model
# score = model.evaluate(x_test, y_test, verbose=1)

# print("Test loss:", score[0])
# print("Test accuracy:", score[1])
