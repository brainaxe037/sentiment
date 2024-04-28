from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from joblib import load

# Load and preprocess the data (if not already done)
text = 'i am jealous'

tokenizer = load("tokenizer.joblib")
input_text = tokenizer.texts_to_sequences([text])
max_sequence_length = 200  # Use the same max sequence length as during training
input_text_padded = pad_sequences(input_text, maxlen=max_sequence_length)

# Load the trained model
model = load_model('bi_lstm.h5')

# Predict using the model
predictions = model.predict(input_text_padded)

# Determine the sentiment label based on the prediction
predicted_class = np.argmax(predictions)
sentiment_labels = ['negative', 'neutral', 'positive']  # Replace with your labels
predicted_label = sentiment_labels[predicted_class]
print(predicted_label)


