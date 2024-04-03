import json
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def preprocess_text(text):
    """
    Preprocesses text using various techniques such as
    normalization, tokenization, stop word removal,
    stemming, etc.

    @param text:
        The text to be pre-processed

    @return: preprocessed_text
        The pre-processed string
    """

    # Normalization - (Lowercase the text)
    text = text.lower()

    # Normalization - (Remove non-alphanumeric characters and extra whitespaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization - (splitting text into individual words):
    tokens = word_tokenize(text)

    # Stopword Removal - (commonly occurring words that carry little meaning such as "is", "the")
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Perform stemming (reduce words to the root form such as loving, loved -> "love")
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


def perform_preprocessing(file_path: str, column: str):
    """
    Performs text preprocessing on any given column.

    @param file_path:
        The path of the JSON file to be processed

    @param column:
        The column to be preprocessed

    @return: None
    """
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
        df = pd.DataFrame(data)

    # Data Preprocessing (Clean dataset)
    df[column] = df[column].apply(preprocess_text)  # => Overwrites 'text' column

    # Save pre-processed data to a new JSON file
    df.to_json(file_path.replace('.json', '_preprocessed.json'), orient='records', lines=True)
