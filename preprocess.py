import os
import json
import music21 as m21
import numpy as np
import tensorflow.keras as keras

# Constants
KERN_DATASET_PATH = "deutschl/erk"
SAVE_DIR = "dataset"
SINGLE_FILE_DATASET = "file_dataset"
MAPPING_PATH = "mapping.json"
SEQUENCE_LENGTH = 64

# Durations are expressed in quarter length
ACCEPTABLE_DURATIONS = [
    0.25,  # 16th note
    0.5,   # 8th note
    0.75,
    1.0,   # quarter note
    1.5,
    2,     # half note
    3,
    4      # whole note
]

def load_songs_in_kern(dataset_path):
    """Loads all kern pieces in the dataset using music21.

    :param dataset_path (str): Path to the dataset.
    :return songs (list of m21.stream.Stream): List containing all pieces.
    """
    songs = []

    # Iterate through all files in the dataset and load them with music21
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".krn"):  # Consider only kern files
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs

def has_acceptable_durations(song, acceptable_durations):
    """Checks if a song has only acceptable durations.

    :param song (m21.stream.Stream): The song to check.
    :param acceptable_durations (list): List of acceptable durations in quarter length.
    :return (bool): True if all durations are acceptable, False otherwise.
    """
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True

def transpose(song):
    """Transposes song to C major/A minor.

    :param song (m21.stream.Stream): The song to transpose.
    :return transposed_song (m21.stream.Stream): The transposed song.
    """
    # Get the key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

    # Estimate key using music21 if not already defined
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    # Calculate the interval for transposition
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # Transpose song by the calculated interval
    transposed_song = song.transpose(interval)
    return transposed_song

def encode_song(song, time_step=0.25):
    """Converts a score into a time-series-like music representation.

    :param song (m21.stream.Stream): The song to encode.
    :param time_step (float): Duration of each time step in quarter length.
    :return encoded_song (str): Encoded song as a string.
    """
    encoded_song = []

    for event in song.flat.notesAndRests:
        # Handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
        # Handle rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        # Convert the note/rest into time series notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # Convert encoded song to string
    encoded_song = " ".join(map(str, encoded_song))
    return encoded_song

def preprocess(dataset_path):
    """Main preprocessing function to load, filter, transpose, encode, and save songs.

    :param dataset_path (str): Path to the dataset.
    """
    # Load folk songs
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")

    for i, song in enumerate(songs):
        # Filter out songs with non-acceptable durations
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue

        # Transpose songs to Cmaj/Amin
        song = transpose(song)

        # Encode songs
        encoded_song = encode_song(song)

        # Save encoded songs to text files
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)

        if i % 10 == 0:
            print(f"Song {i} out of {len(songs)} processed")

def load(file_path):
    """Loads a song from a file.

    :param file_path (str): Path to the file.
    :return song (str): The loaded song.
    """
    with open(file_path, "r") as fp:
        song = fp.read()
    return song

def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    """Generates a file collating all the encoded songs and adding new piece delimiters.

    :param dataset_path (str): Path to the folder containing the encoded songs.
    :param file_dataset_path (str): Path to the file for saving songs in a single file.
    :param sequence_length (int): Number of time steps to be considered for training.
    :return songs (str): String containing all songs in the dataset with delimiters.
    """
    new_song_delimiter = "/ " * sequence_length
    songs = ""

    # Load encoded songs and add delimiters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs += song + " " + new_song_delimiter

    # Remove the last space from the string
    songs = songs.rstrip()

    # Save the string containing all the dataset
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)

    return songs

def create_mapping(songs, mapping_path):
    """Creates a JSON file that maps the symbols in the song dataset onto integers.

    :param songs (str): String with all songs.
    :param mapping_path (str): Path to save the mapping.
    """
    mappings = {}

    # Identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))

    # Create mappings
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # Save vocabulary to a JSON file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)

def convert_songs_to_int(songs):
    """Converts song symbols to integers using the mappings.

    :param songs (str): String with all songs.
    :return int_songs (list): List of integer-encoded songs.
    """
    int_songs = []

    # Load mappings
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)

    # Transform songs string to list
    songs = songs.split()

    # Map songs to integers
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs

def generate_training_sequences(sequence_length):
    """Creates input and output data samples for training. Each sample is a sequence.

    :param sequence_length (int): Length of each sequence.
    :return inputs (ndarray): Training inputs.
    :return targets (ndarray): Training targets.
    """
    # Load songs and map them to integers
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)

    inputs = []
    targets = []

    # Generate the training sequences
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])

    # One-hot encode the sequences
    vocabulary_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    print(f"There are {len(inputs)} sequences.")

    return inputs, targets

def main():
    """Main function to preprocess the dataset and create training sequences."""
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    # Uncomment the following line to generate training sequences
    # inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

if __name__ == "__main__":
    main()
