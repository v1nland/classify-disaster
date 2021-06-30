import json
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.callbacks import EarlyStopping

# reduce core use
# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(112)

WORD_DIM = 200
WORDS_PER_TWEET = 16


# Model / data parameters
# - train dataset lines = batch_size * steps_per_epoch
# - validate dataset lines = batch_size * validate_steps_per_epoch
# - in this case, we are working with 1.600.000 tweets for training and 300.000 for validation
#   so valid values are: batch_size = 20.000, steps_per_epoch = 80, validate_steps_per_epoch = 15
# - if training size is 20.000 and validation size is 4.000
#   valid values are: batch_size = 1.000, steps_per_epoch = 20, validate_steps_per_epoch = 4

input_shape = (WORDS_PER_TWEET, WORD_DIM, 1)  # shape of the input vector
num_classes = 2  # output target 0 eq 1 hurricane

epochs = 15  # number of epochs
batch_size = 10000  # number of tweets in each batch

steps_per_epoch = 160
validate_steps_per_epoch = 32


def train_batch_generator(file_name, epochs, batch_size, num_classes, ending_line):
    for _ in range(epochs):
        batch_number = 0
        lines_counter = 0

        x_train = []
        y_train = []

        dataset = open(
            file_name,
            "r",
        )

        for line in dataset:
            lines_counter = lines_counter + 1

            if lines_counter > ending_line:
                break

            tweet = json.loads(line)

            y_train.append(tweet["type"])
            x_train.append(np.array([word["vector"] for word in tweet["data"]]))

            if lines_counter % batch_size == 0:
                sample = tweet["data"][0]["word"]
                batch_number = batch_number + 1

                print(
                    "\ntraining batch n: "
                    + str(batch_number)
                    + ", batch_size: "
                    + str(len(x_train))
                    + ", sample "
                    + str(len(x_train) * batch_number)
                    + ": "
                    + sample
                )

                # convert to np array
                x = np.array(x_train)
                y = np.array(y_train)

                # Scale images to the [0, 1] range
                x = x.astype("float32") / 255

                # Make sure all 40 images have shape (WORDS_PER_TWEET, TWEET_DIM, 1)
                x = np.expand_dims(x, -1)

                # convert class vectors to binary class matrices
                y = keras.utils.to_categorical(y, num_classes)

                x_train.clear()
                y_train.clear()

                yield (x, y)

        dataset.close()


# note: this prints one more "set" than specified on steps_per_epoch
# but the first one isn't used on the model validation
def validate_batch_generator(file_name, epochs, batch_size, num_classes, starting_line):
    for _ in range(epochs):
        batch_number = 0
        lines_counter = 0

        x_train = []
        y_train = []

        dataset = open(
            file_name,
            "r",
        )

        for line in dataset:
            lines_counter = lines_counter + 1

            if lines_counter < starting_line + 1:
                continue

            tweet = json.loads(line)

            y_train.append(tweet["type"])
            x_train.append(np.array([word["vector"] for word in tweet["data"]]))

            if lines_counter % batch_size == 0:
                sample = tweet["data"][0]["word"]
                batch_number = batch_number + 1

                print(
                    "\nvalidation batch n: "
                    + str(batch_number)
                    + ", batch_size: "
                    + str(len(x_train))
                    + ", sample "
                    + str((len(x_train) * batch_number) + starting_line)
                    + ": "
                    + sample
                )

                # convert to np array
                x = np.array(x_train)
                y = np.array(y_train)

                # Scale images to the [0, 1] range
                x = x.astype("float32") / 255

                # Make sure all 40 images have shape (WORDS_PER_TWEET, TWEET_DIM, 1)
                x = np.expand_dims(x, -1)

                # convert class vectors to binary class matrices
                y = keras.utils.to_categorical(y, num_classes)

                x_train.clear()
                y_train.clear()

                yield (x, y)

            # lines_counter = 0

        dataset.close()


file = "../word-vectorization/vectorizated_shuffled_set/nepal_harvey-irma-maria.vectorizated_shuffled_set"

train_gen = train_batch_generator(
    file, epochs, batch_size, num_classes, batch_size * steps_per_epoch
)
valid_gen = validate_batch_generator(
    file, epochs, batch_size, num_classes, batch_size * steps_per_epoch
)

print("Generators created successfully")

# Build the model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

early_stopping = EarlyStopping(
    monitor="loss", mode="min", patience=5, restore_best_weights=True
)

model.fit(
    x=train_gen,
    # y_train, not specified because using generator
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_gen,
    validation_steps=validate_steps_per_epoch - 1,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    callbacks=[early_stopping],
)

model.save("models/nepal_harvey-irma-maria_05.h5")
