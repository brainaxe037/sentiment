from sklearn.base import BaseEstimator, TransformerMixin
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer  # Import the PorterStemmer


class ProcessingText(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stemmer = PorterStemmer()  # Initialize the stemmer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        processed_texts = [self.text_stemming(self.text_process(text)) for text in X]
        return processed_texts

    def text_process(self, mess):
        no_punctuation = [char for char in mess if char not in string.punctuation]
        no_punctuation = ''.join(no_punctuation)
        return ' '.join([word for word in no_punctuation.split() if word.lower() not in stopwords.words('english')])

    def text_stemming(self, text):
        stemmed = ' '.join([self.stemmer.stem(token) for token in text.split()])  # Use the stemmer
        return stemmed
