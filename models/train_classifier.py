import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
import pandas as pd
import numpy as np
import sys
from sqlalchemy import create_engine
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):
    '''
    load data from database and return X and y

    database_filename: name of database

    returns
    X: Predictors
    y: targets
    categories: names of target categories
    '''
    database_name =  'sqlite:///' + database_filepath
    engine = create_engine(database_name)
    df = pd.read_sql_table('disaster_clean', engine)

    X = df.message.values
    X = X[0:500]
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    y = y.iloc[0:500, :]

    category_names = list(y.columns.values)

    return X,y,category_names

def tokenize(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    words = word_tokenize(text)

    words = [w for w in words if not w in stop_words]
    words = [word.lower() for word in words if word.isalpha()]
    words = [stemmer.stem(word) for word in words]

    return words


def build_model():
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])

    ''' 
    parameters = {
        'clf__estimator__n_jobs': [10,20]
        #clf__max_features = ['auto', 'sqrt'],
        #clf__min_samples_leaf = [1, 2, 4]
        #clf__bootstrap = [True, False]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters,cv=3)
    return cv
    '''

    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    label_df: pandas dataframe containing test targets
    preds_array: numpy array containing predicted targets

    Function to print classification report for each target column
    '''
    y_pred = model.predict(X_test)

    for i, col in enumerate(category_names):
        print(col)
        print(classification_report(y_true=Y_test[col], y_pred=y_pred[:, i]))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
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