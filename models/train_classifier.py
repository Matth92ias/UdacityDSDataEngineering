import nltk
nltk.download('stopwords')
nltk.download('punkt')
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
from sklearn.metrics import hamming_loss, accuracy_score 
from sklearn.metrics import multilabel_confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings(action="ignore", message=r'.*Use subset.*of np.ndarray is not recommended')
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
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
    '''
    array with texts

    tokenizer used for function CountVectorizer
    '''
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    words = word_tokenize(text)

    words = [w for w in words if not w in stop_words]
    words = [word.lower() for word in words if word.isalpha()]
    words = [stemmer.stem(word) for word in words]

    return words


def build_model(y_train):
    '''
    input: y_train (needed to know the amount of classes in training set)

    define complete CV pipeline to define model. Best cv result for fit on complete data
    '''
    model = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf',MultiOutputClassifier(xgb.XGBClassifier(objective='multi:softprob',
                                  eval_metric='mlogloss',
                                  use_label_encoder=False,
                                  num_class= y_train.shape[1])))
    ])
    

    params = { 'clf__estimator__max_depth': [2,3,5],
           'clf__estimator__learning_rate': [0.1, 0.2, 0.3],
           'clf__estimator__subsample': np.arange(0.5, 1.0, 0.1),
           'clf__estimator__colsample_bytree': np.arange(0.4, 1.0, 0.1),
           'clf__estimator__colsample_bylevel': np.arange(0.4, 1.0, 0.1),
           'clf__estimator__n_estimators': [30,50,100],
           #'clf__estimator__n_estimators': [10],
           'clf__estimator__reg_lambda': [0,0.25,0.5,0.75,1]         
         }

                  
    cv = RandomizedSearchCV(estimator=model,
                        param_distributions=params,
                        n_iter=10,
                        cv=3,
                        verbose=2)

    
    return cv

def evaluate_model(model, X_test, y_test, category_names):
    '''
    label_df: pandas dataframe containing test targets
    preds_array: numpy array containing predicted targets

    Function to print classification report for each target column
    '''
    
    print('Best CV parameters are')
    print(model.best_params_)
    
    final_model = model.best_estimator_
    y_pred = final_model.predict(X_test)
    
    print(hamming_loss(y_test.values,y_pred))
    print(accuracy_score(y_test.values,y_pred))

    for i, col in enumerate(category_names):
        print('{} category metrics: '.format(col))
        print(classification_report(y_test.iloc[:,i], y_pred[:,i]))


def save_model(model, model_filepath):
    '''
    model: sklearn object used to fit model (complete pipeline)
    model_filepath: path where model is saved

    save trained model as pickle to disk
    '''
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        # one column with a none binary value - cleaned here
        for col in y_train.columns:
            y_train[col] = y_train[col].map(lambda x: 1 if x > 0 else 0)
        for col in y_test.columns:
            y_test[col] = y_test[col].map(lambda x: 1 if x > 0 else 0)

        print('Building model...')
        model = build_model(y_train)
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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