import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from sklearn.metrics import classification_report
import sqlalchemy as db
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import precision_recall_fscore_support
import pickle
import time
from sklearn.metrics import f1_score

def load_data(database_filepath):
    """
    This function is to load the saved dataset file through EIL pipeline
    Input:
    database_filepath: the name of dataset we just made for our saved dataset
    Output:
    X: an independent variable, here is the messages
    Y: a dependent variable, here is the categories variables (1 or 0)
    category_names: the labels of Y variable
    """
    df = pd.read_sql_table('table','sqlite:///{}'.format(database_filepath))
    X = df.message.values
    Y = df.iloc[:,4:].values
    category_names = df.iloc[:,4:].columns
    return X, Y, category_names

def tokenize(message):
    """
    This function is to tokenize the message and oupput the tokenized form of messages, which is ready for the machine learning model
    Input: raw text massages 
    Output: the tokenized the text words in a list
    """
    # Identify any urls in text, and replace each one with the word, "urlplaceholder"
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, message)
    for url in detected_urls:
        message = message.replace(url, "urlplaceholder")
    # Split text into tokens.
    tokens = word_tokenize(message)
    # For each token: lemmatize, normalize case, and strip leading and trailing white space.
    lemmatizer = WordNetLemmatizer()
    # Return the tokens in a list!
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    """
    This function is to build a model to Train classifier, it include:
    - Fit and transform the training data with CountVectorizer. Hint: You can include your tokenize function in the tokenizer keyword argument!
    - Fit and transform these word counts with TfidfTransformer.
    - Fit a classifier to these tfidf values.
    Input: None
    Output: a machine learning pipeline
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function is to evaluate the model result and display result
    Input:
    model: the model pipeline we build in the function 'build_model()'
    X_test: independent variable in test dataset
    Y_test: dependent variable in test dataset
    category_names: the label name of dependent variables
    Output: None
    """
    labels = category_names
    # Predict on test data
    Y_pred = model.predict(X_test)
    # Display a classification report and accuracy score based on the model's predictions.
    accuracy = (Y_pred == Y_test).mean()
    print("The accuracy of predicted labels of test dataset is {}".format(accuracy))
    print('The classification report is')
    print(classification_report(Y_test, Y_pred, target_names= labels))
    pass


def save_model(model, model_filepath):
    """
    This function is to save the model in a pickle file
    Input:
    model: the model we build
    model_filepath: the file path we want to save the model
    Output: none
    """
    pickle.dump(model,open(model_filepath,'wb'))
    pass


def main():
    """
    A function that organized all previous functions together
    Input: none
    Output: none
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        # Load data and perform a train test split
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
