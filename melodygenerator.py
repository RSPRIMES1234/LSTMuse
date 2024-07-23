import json
import numpy as np
import tensorflow.keras as keras
import music21 as m21
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH

class MelodyGenerator:
    """A class that wraps the LSTM model and offers utilities to generate melodies."""

    def __init__(self, model_path="model.h5"):
        """Constructor that initializes the TensorFlow model."""
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH

    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        """Generates a melody using the LSTM model.

        :param seed (str): Melody seed with the notation used to encode the dataset.
        :param num_steps (int): Number of steps to be generated.
        :param max_sequence_length (int): Max number of steps in seed to be considered for generation.
        :param temperature (float): Float in interval [0, 1]. Numbers closer to 0 make the model more deterministic.
                                     A number closer to 1 makes the generation more unpredictable.
        :return melody (list of str): List with symbols representing a melody.
        """
        # Split the seed into a list of symbols
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # Map seed symbols to integers
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):
            # Limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]

            # One-hot encode the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            onehot_seed = onehot_seed[np.newaxis, ...]

            # Make a prediction
            probabilities = self.model.predict(onehot_seed)[0]

            # Sample an output symbol using temperature
            output_int = self._sample_with_temperature(probabilities, temperature)

            # Update seed
            seed.append(output_int)

            # Map the integer back to a symbol
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            # Check whether we're at the end of a melody
            if output_symbol == "/":
                break

            # Update melody
            melody.append(output_symbol)

        return melody

    def _sample_with_temperature(self, probabilities, temperature):
        """Samples an index from a probability array reapplying softmax using temperature.

        :param probabilities (nd.array): Array containing probabilities for each of the possible outputs.
        :param temperature (float): Float in interval [0, 1]. Numbers closer to 0 make the model more deterministic.
                                    A number closer to 1 makes the generation more unpredictable.
        :return index (int): Selected output symbol.
        """
        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities)

        return index

    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="melody.mid"):
        """Converts a melody into a MIDI file.

        :param melody (list of str): List of symbols representing the melody.
        :param step_duration (float): Duration of each time step in quarter length.
        :param format (str): Format to save the file in (default is "midi").
        :param file_name (str): Name of the MIDI file to be saved.
        """
        # Create a music21 stream
        stream = m21.stream.Stream()

        start_symbol = None
        step_counter = 1

        # Parse all the symbols in the melody and create note/rest objects
        for i, symbol in enumerate(melody):
            if symbol != "_" or i + 1 == len(melody):
                if start_symbol is not None:
                    quarter_length_duration = step_duration * step_counter

                    # Handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                    # Handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)
                    step_counter = 1

                start_symbol = symbol
            else:
                step_counter += 1

        # Write the music21 stream to a MIDI file
        stream.write(format, file_name)


if __name__ == "__main__":
    mg = MelodyGenerator()
    
    # Example seeds
    seeds = [
        "67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _",
        "67 _ _ _ _ _ 65 _ 64 _ 62 _ 60 _ _ _",
        "60 _ 60 _ 67 _ 67 _ 69 _ 69 _ 67 _ _",
        "67 _ 67 _ 69 _ 69 _ 70 _ 72 _ 72 _ 70 _ 69 _ 67 _ _",
        "62 _ 62 _ 62 _ 66 _ 66 _ 64 _ 64 _",
        "64 _ 60 _ 60 _ 62 _ 67 _ 66 _ 64 _"
    ]
    
    # Generate a melody using one of the seeds
    melody = mg.generate_melody(seeds[5], 500, SEQUENCE_LENGTH, 0.3)
    print(melody)
    
    # Save the generated melody to a MIDI file
    mg.save_melody(melody)
