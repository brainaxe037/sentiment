import pandas as pd
from joblib import dump
from text_processing import ProcessingText
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
sample_data = pd.read_csv('dataset.csv')

# Create a pipeline with text processing, Count Vectorization, and Logistic Regression
code_chain = Pipeline([
    ('text_processor', ProcessingText()),
    ('count_vectorized', CountVectorizer()),
    ('logistic_regression', LogisticRegression(max_iter=1000))
])

# Split your data
X_train, X_test, Y_train, Y_test = train_test_split(sample_data['Sentence'],
                                                    sample_data['Sentiment'], test_size=0.15, random_state=13)

# Fit the pipeline on the training data
code_chain.fit(X_train, Y_train)

# Make predictions on the test data
predictions = code_chain.predict(X_test)

# Evaluate the Logistic Regression model
print('Confusion Matrix:\n', confusion_matrix(Y_test, predictions))
print('\n')
print('Classification Report:\n', classification_report(Y_test, predictions))
print('\n')
print('Accuracy:', accuracy_score(Y_test, predictions))

# Save the trained model
dump(code_chain, filename="logr_model.joblib")
