import pandas as pd
from joblib import dump
from text_processing import ProcessingText
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
input_dataset = pd.read_csv('dataset.csv')

# Create a pipeline with text processing, Count Vectorization, and MNB Classifier
code_chain = Pipeline([
    ('text_processor', ProcessingText()),
    ('count_vectorized', CountVectorizer()),
    ('multinomial_nb', MultinomialNB())
])

# Split your data
X_train, X_test, Y_train, Y_test = train_test_split(input_dataset['Sentence'],
                                                    input_dataset['Sentiment'], test_size=0.22, random_state=101)

# Fit the pipeline on the training data
code_chain.fit(X_train, Y_train)

# Make predictions on the test data
predicted_values = code_chain.predict(X_test)


# Evaluate the MNB model
print('Confusion Matrix:\n', confusion_matrix(Y_test, predicted_values))
print('\n')
print('Classification Report:\n', classification_report(Y_test, predicted_values))
print('\n')
print('Accuracy:', accuracy_score(Y_test, predicted_values))

# Save the trained model
dump(code_chain, filename="mnb_model.joblib")
