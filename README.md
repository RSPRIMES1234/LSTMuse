# LSTMuse: An AI-Powered Melody Generator
![LSTMUSE](https://github.com/user-attachments/assets/45abc6d1-09a3-4b80-930a-8e2255d6af32)


LSTMuse is a deep learning project that generates unique melodies in various musical styles. It leverages Long Short-Term Memory (LSTM) neural networks, a type of recurrent neural network well-suited for sequential data like music. The project is designed to be adaptable, allowing for training on different datasets to explore various musical genres and influences.

## Key Features

- **Preprocessing:** Transforms music data into a suitable format for the model. Includes transposing to a common key, encoding melodies into numerical representations, and preparing training sequences.
- **LSTM Model:** Employs an LSTM network to learn patterns and structures in musical data. The model predicts the probability distribution of the next note based on the preceding notes, capturing musical relationships and dependencies.
- **Melody Generation:** Generates original melodies by sampling from the model's predicted probabilities. A temperature parameter controls the degree of randomness, allowing for both structured and more experimental compositions.
- **MIDI Output:** Saves the generated melodies as MIDI files, a standard format for musical data that can be easily played, shared, and further manipulated.

## Technical Details

- **Programming Language:** Python
- **Libraries:** TensorFlow/Keras (deep learning), music21 (music processing)
- **Model Architecture:** Customizable LSTM model with options for adjusting layers, units, and hyperparameters.
- **Dataset:** Can be trained on various datasets of symbolic music representations.

## Getting Started

### Clone the Repository:

```bash
git clone https://your-repository-url.git
```

### Install Dependencies:

```bash
pip install -r requirements.txt
```

### Prepare Your Dataset

Ensure your musical data is in a compatible format (e.g., MIDI or MusicXML) and preprocess it according to the `preprocess.py` script's requirements.

### Train the Model (Optional)

To train on a new dataset, modify the configuration settings in `train.py` and run the script.

### Generate Melodies

Use the `melodygenerator.py` script to generate melodies from a seed or starting point. Customize parameters like temperature to influence the style and creativity of the generated output.

## Potential Applications

- **Creative Inspiration:** Provides musicians and composers with novel melodic ideas and starting points.
- **Game Development:** Generates dynamic and adaptive background music for video games.
- **Music Education:** Aids in learning music theory and composition by exploring AI-generated musical structures.
- **AI and Music Research:** Contributes to research on computational creativity and music generation.

## Contributing

Contributions to LSTMuse are welcome! Feel free to open issues, submit pull requests, or share your generated melodies.
