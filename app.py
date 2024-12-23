import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.backend import clear_session
import pickle

# Load model and tokenizer using Streamlit session state
if "model" not in st.session_state:
    st.session_state.model = load_model("shakespeare_lstm_model.keras")

if "tokenizer" not in st.session_state:
    with open("tokenizer.pickle", "rb") as handle:
        st.session_state.tokenizer = pickle.load(handle)

model = st.session_state.model
tokenizer = st.session_state.tokenizer
max_sequence_len = model.input_shape[1] + 1

# Function to predict the next word
def predict_next_word_safe(model, tokenizer, text, max_sequence_len):
    clear_session()
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding="pre")
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Function to predict the next sequence of words
def predict_next_words_safe(model, tokenizer, text, max_sequence_len, num_words=6, temperature=0.6):
    clear_session()
    generated_words = []
    current_text = text

    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([current_text])[0]
        if len(token_list) >= max_sequence_len:
            token_list = token_list[-(max_sequence_len-1):]
        padded_sequence = pad_sequences([token_list], maxlen=max_sequence_len-1, padding="pre")
        predictions = model.predict(padded_sequence, verbose=0)[0]
        predictions = np.clip(np.log(predictions) / temperature, 1e-8, 1.0)
        exp_predictions = np.exp(predictions - np.max(predictions))
        probabilities = exp_predictions / np.sum(exp_predictions)
        predicted_index = np.random.choice(len(probabilities), p=probabilities)
        output_word = None
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        if output_word:
            generated_words.append(output_word)
            current_text += " " + output_word
        else:
            break

    return current_text

# Streamlit Interface
st.title("Next Word and Sentence Predictor")

st.sidebar.header("Choose Prediction Type")
prediction_type = st.sidebar.selectbox("Select the task:", ["Next Word", "Next Sentence"])

if prediction_type == "Next Word":
    st.header("Predict the Next Word")
    input_text = st.text_input("Enter your text:", "shall we open up")
    if st.button("Predict Next Word"):
        next_word = predict_next_word_safe(model, tokenizer, input_text, max_sequence_len)
        st.write(f"Next Word Prediction: **{next_word}**")

elif prediction_type == "Next Sentence":
    st.header("Predict the Next Sentence")
    input_text = st.text_input("Enter your text:", "in the name of the lord")
    num_words = st.slider("Number of words to predict:", 1, 10, 6)
    temperature = st.slider("Set Temperature (higher = more creative):", 0.1, 1.0, 0.6)
    if st.button("Predict Sentence"):
        next_sentence = predict_next_words_safe(model, tokenizer, input_text, max_sequence_len, num_words, temperature)
        st.write(f"Predicted Sentence: **{next_sentence}**")
