# the run.py works as a combination of route.py and _init_.py
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask # import flask library
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__) # create a variable called app

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
df = pd.read_sql_table('table', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    Related_counts_per_genre = df.groupby('genre').sum()['related']
    Request_counts_per_genre = df.groupby('genre').sum()['request']
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals

    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                ),
            ],    
            'layout': {
                'title': 'Distribution of Message Genres',
                'font':{
                    'family': 'Raleway, sans-serif'
                },

                'showlegend': 'false',
                'yaxis': {
                    'title': "Frequency of Genre Type",
                    'gridwidth': 2,
                    'zeroline': 'false'
                },
                'xaxis': {
                    'title': "Genre Type"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=Related_counts_per_genre
                ),
            ],    
            'layout': {
                'title': 'Distribution of Related Message per Genres',
                'font':{
                    'family': 'Raleway, sans-serif'
                },

                'showlegend': 'false',
                'yaxis': {
                    'title': "Frequency of Related Message",
                    'gridwidth': 2,
                    'zeroline': 'false'
                },
                'xaxis': {
                    'title': "Genre Type"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=Request_counts_per_genre
                ),
            ],    
            'layout': {
                'title': 'Distribution of Request Message per Genres',
                'font':{
                    'family': 'Raleway, sans-serif'
                },

                'showlegend': 'false',
                'yaxis': {
                    'title': "Frequency of Request Message",
                    'gridwidth': 2,
                    'zeroline': 'false'
                },
                'xaxis': {
                    'title': "Genre Type"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    # ids is a list containing the distinct id for each plot
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    # graphJason contains a dictionary of graphs and transform it with a plotly format that Json can decode
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render specific web page with plotly graphs
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