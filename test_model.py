# import numpy as np
from tensorflow.keras.models import load_model
from load_set import LoadSetFromFile

# WORD_DIM = 200
# WORDS_PER_TWEET = 16


def TestModel(model):
    model = load_model("models/" + model + ".h5")  # make sure of model shape

    validation_batches = range(20)

    total_loss = 0
    total_accuracy = 0
    for i in validation_batches:
        (x_test, y_test) = LoadSetFromFile(
            "vectorizated_shuffled_set/nepal_harvey-irma-maria_validation_batches/"
            + str(i),
            2,
        )

        score = model.evaluate(x_test, y_test, verbose=1)
        total_loss = total_loss + score[0]
        total_accuracy = total_accuracy + score[1]

    print("Test loss:", total_loss / len(validation_batches))
    print("Test accuracy:", total_accuracy / len(validation_batches))


TestModel("nepal_harvey-irma-maria_03")
