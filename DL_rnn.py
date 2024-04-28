import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from text_processing import ProcessingText
from sklearn.metrics import confusion_matrix, classification_report

sample_data = pd.read_csv('dataset.csv')
# text_processor = TextProcessor()
# sample_data['Sentence'] = text_processor.fit_transform(sample_data['Sentence'])

X_train, X_test, Y_train, Y_test = train_test_split(sample_data['Sentence'], sample_data['Sentiment'], test_size=0.15)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sample_data['Sentence'])
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)
voc_size = len(tokenizer.word_index)

max_sequence_length = 200
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_sequence_length)
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_sequence_length)

label_encoder = LabelEncoder()
Y_train_encoded = label_encoder.fit_transform(Y_train)
Y_test_encoded = label_encoder.transform(Y_test)

Y_train_encoded = to_categorical(Y_train_encoded, num_classes=3)
Y_test_encoded = to_categorical(Y_test_encoded, num_classes=3)

model = Sequential()
model.add(Dense(128, input_dim=max_sequence_length, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(X_train_padded, Y_train_encoded, validation_data=(X_test_padded, Y_test_encoded),
                    epochs=5, batch_size=32)

# assigning the losses and accuracies
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Plot training & validation loss values
plt.figure(figsize=(10, 5))
plt.subplot(121)  # Subplot for loss
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training & validation accuracy values
plt.subplot(122)  # Subplot for accuracy
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# display the plots
plt.show()

# Evaluate the model
predictions = np.argmax(model.predict(X_test_padded))
y_true = np.argmax(Y_test_encoded)

# Calculate the confusion matrix
print(confusion_matrix(y_true, predictions))

# Calculate and print the classification report
print(classification_report(y_true, predictions))

# model.save('rnn_model.h5')