import tensorflow.keras as keras
from preprocessing import gen_train_seq, SEQUENCE_LENGTH

# Constants
OUTPUT_UNITS = 40
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
NUM_UNITS = [256]
EPOCHS = 80
BATCH_SIZE = 256
SAVE_MODEL_PATH = "model.h5"

def build_model(output_units, num_units, loss, learning_rate):
    """Builds and compiles the model.

    :param output_units (int): Number of output units.
    :param num_units (list of int): Number of units in hidden layers.
    :param loss (str): Type of loss function to use.
    :param learning_rate (float): Learning rate to apply.

    :return model (keras.Model): Compiled model.
    """

    # Create the model architecture
    input = keras.layers.Input(shape=(None, output_units))
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)

    # Compile the model
    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])
    model.summary()

    return model

def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):
    """Trains and saves the TensorFlow model.

    :param output_units (int): Number of output units.
    :param num_units (list of int): Number of units in hidden layers.
    :param loss (str): Type of loss function to use.
    :param learning_rate (float): Learning rate to apply.
    """

    # Generate the training sequences
    inputs, targets = gen_train_seq(SEQUENCE_LENGTH)

    # Build the network
    model = build_model(output_units, num_units, loss, learning_rate)

    # Train the model
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Save the model
    model.save(SAVE_MODEL_PATH)

if __name__ == "__main__":
    train()
