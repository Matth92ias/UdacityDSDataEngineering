import json
import plotly
import numpy as np
import pandas as pd

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('disaster_clean', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals

    # Distribution of different categories
    cat_count = []
    categories = df.iloc[:,4:].columns
    for cat in categories:
        cat_count.append(df[cat].sum())
    cat_count = np.array(cat_count)
    order = np.argsort(cat_count)[::-1]

    cat_count_x = categories[order]
    cat_count_y = cat_count[order]

    # Histogram distribution of number of words in messsages
    words_in_message = df.message.str.split().str.len()
    words_x, words_y = np.histogram(
            words_in_message,
            range=(0, words_in_message.quantile(0.99))
        )


    # create visuals
    graphs = [

        {
            'data': [
                Bar(
                    marker= {'color': '#0000b2'},
                    x=cat_count_x,
                    y=cat_count_y
                )
            ],

            'layout': {
                'title': 'Distribution of different Categories',
                'yaxis': {
                    'title': "Message Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    marker= {'color': '#0000b2'},
                    x=words_x,
                    y=words_y
                    )
                 ],
            'layout': {
                'title': 'Distribution of words per message',
                
                'yaxis': {
                    'title': "Word Count"
                },
                'xaxis': {
                    'title': "Words per Message"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()

