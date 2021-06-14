import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from load_set import LoadSetFromFile

WORD_DIM = 200
WORDS_PER_TWEET = 16

# Model / data parameters
num_batches = 100  # file batches to train model
batch_size = 20000  # num of tweet in each batch
training_samples = num_batches * batch_size
num_classes = 2  # output target 0 eq 1 hurricane
steps_per_epoch = np.ceil(50)  # steps equals to the data quantity of each batch
# validation_steps = np.ceil(28000 / batch_size)
input_shape = (WORDS_PER_TWEET, WORD_DIM, 1)
epochs = 15


def batch_generator(Train_df, batch_size, num_batches, num_classes):
    idx = 0
    while True:
        yield LoadSetFromFile(
            # "parsed_train_set/earthquake_hurricane_batches/" + str(idx)
            "parsed_train_set/nepal_harvey-irma-maria_batches/" + str(idx),
            num_classes,
        )

        if idx < num_batches:
            idx += 1
        else:
            break


print("Loading set...")

# (x_train, y_train) = LoadSetFromFile(
#     "parsed_train_set/nepal_harvey-irma-maria.parsed_train_set", num_classes
# )

# Scale images to the [0, 1] range
# x_train = x_train.astype("float32") / 255
# x_test = x_test.astype("float32") / 255


# Make sure all 40 images have shape (WORDS_PER_TWEET, WORD_DIM, 1)
# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)

# print("x_train shape:", x_train.shape)
# print(x_train.shape[0], "train samples")

# convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)

my_training_batch_generator = batch_generator("", 0, num_batches, num_classes)
my_validation_batch_generator = batch_generator("", 0, num_batches, num_classes)

print("Train set loaded successfully")

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

# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
model.fit_generator(
    my_training_batch_generator,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    verbose=1,
)

model.save("models/nepal_harvey-irma-maria.h5")
