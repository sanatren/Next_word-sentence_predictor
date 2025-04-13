# Next Word & Sentence Predictor

Built using LSTM and RNN techniques, this project predicts the next word or generates entire sentences based on user input. It leverages TensorFlow and Keras for model building and Streamlit for an interactive web interface.

> ### 🔗 Live Demo: [whatisnext.streamlit.app](https://whatisnext.streamlit.app/)

---

## Demo Preview

Try it live → [whatisnext.streamlit.app](https://whatisnext.streamlit.app/)

---

## Features

- 🔮 Predicts the next word given a text input.
- ✍️ Generates entire sentences in the style of Shakespeare.
- 🧠 Built with LSTM and RNN architecture using TensorFlow / Keras.
- 🖥️ Simple and interactive web interface using Streamlit.
- 🔄 Easy to retrain or fine-tune with your own dataset.

---

## Tech Stack

- Python
- TensorFlow / Keras
- LSTM & RNN
- Streamlit
- Pickle (Tokenizer storage)
- Dataset: Shakespeare's Text

---
Project Structure
.
├── app.py                    # Streamlit Web App
├── practical.ipynb           # Model Training Notebook
├── shakespeare.txt           # Dataset
├── shakespeare_lstm_model.keras # Trained Model
├── tokenizer.pickle          # Fitted Tokenizer
├── requirements.txt          # Python dependencies
└── need.txt                  # Additional notes


## Setup Locally
1. Clone the Repository
bashgit clone https://github.com/sanatren/Next_word-sentence_predictor.git
cd Next_word-sentence_predictor
2. Create and Activate Virtual Environment (Recommended)
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
bashpip install -r requirements.txt
4. Run the Streamlit App
bashstreamlit run app.py
Features

Next Word Prediction: Enter text and get predictions for the most likely next word
Text Generation: Generate Shakespeare-like text based on a seed phrase
Interactive Web Interface: Easy-to-use Streamlit application

## Model Details
The project uses an LSTM (Long Short-Term Memory) neural network trained on Shakespeare's works. The model learns patterns in Shakespeare's language and writing style to make predictions about what word would naturally follow a given sequence of text.
Training Your Own Model
If you want to train the model yourself, explore the practical.ipynb notebook which contains the complete training pipeline from data preprocessing to model evaluation.
Requirements

Python 3.8+
TensorFlow 2.x
Streamlit
Additional dependencies in requirements.txt
