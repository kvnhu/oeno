import difflib
import numpy as np
import os
import pandas as pd
import re
import string

from flask import Flask
from flask import render_template
from flask import request
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

app = Flask(__name__)

DESCRIPTION = 'description'
TITLE = 'title'
EMBEDDING = 'embedding'

WINE_DATA_FILE_PATH = "./data/winemag-data-130k-v2.csv"

original_wine = None
wine_term = None
wines = None
model = None

@app.route("/")
def index():
    """ Index page for the web app. """
    return render_template('index.ejs')

@app.route("/search-wine", methods=['POST'])
def recommend_wine():
    """ Saves the wine already liked by the user. """
    global original_wine
    global wines
    global model

    print('fitting model and obtaining original enjoyed wine...')
    fit_model_and_compute_description_embeddings()
    original_wine = str(request.form['wine'])

    def match_wine_title():
        title_match_scores = wines[TITLE].apply(lambda x: difflib.SequenceMatcher(None, original_wine.lower(), x.lower()).ratio())
        return title_match_scores.argmax()

    print('matching wines...')
    matched_idx = match_wine_title()
    description_match_scores = wines[EMBEDDING].apply(
        lambda x: np.dot(
            np.array(x),
            np.array(wines[EMBEDDING].iloc[matched_idx])
        )
    )

    print('generating output...')
    html = ""
    for wine in wines[TITLE][description_match_scores.nlargest(11).index]:
        html += wine + '<br>'
    return html

@app.route("/search-term", methods=['POST'])
def get_wine_term():
    """ Finds similar terms. """
    global wine_term
    global model

    print('fitting model and obtaining wine term...')
    fit_model_and_compute_description_embeddings(model_only=True)
    wine_term = request.form['term'].lower()

    print ('generating output...')
    html = u"Closest terms to %s:<br>" % wine_term
    for term, score in model.wv.most_similar(wine_term, topn=1000):
        try:
            html += '> ' + term.decode('utf-8') + '<br>'
        except:
            print (term)
    return html

@app.route("/wine-math", methods=['POST'])
def wine_math():
    """ Computes wine flavor arithmetic. """
    global wine_term
    global model

    print('fitting model and obtaining wine terms...')
    fit_model_and_compute_description_embeddings(model_only=True)
    add_terms = request.form['adds'].lower()
    subtract_terms = request.form['subtracts'].lower()

    add_term_list = ','.join(add_terms.split(', ')).split(',')
    sub_term_list = ','.join(subtract_terms.split(', ')).split(',')

    print ('generating output...')
    html = u"%s <br> - %s <br> = <br>" % (' + '.join(add_term_list), ' - '.join(sub_term_list))
    strong_html = "Strong matches:<br>"
    weak_html = "Weak matches:<br>"
    for term, score in model.wv.most_similar(positive=add_term_list, negative=sub_term_list, topn=100):
        no_match = True
        for candidate in (add_term_list + sub_term_list):
            if difflib.SequenceMatcher(None, term.decode('utf-8'), candidate.decode('utf-8')).ratio() > 0.5:
                print term
                no_match = False
        if no_match:
            if score > 0.5:
                strong_html += '> ' + term.decode('utf-8') + '<br>'
            elif score > 0.3:
                weak_html += '> ' + term.decode('utf-8') + '<br>'

    return html + '<br>' + strong_html + '<br>' + weak_html

def prepare_dataset(fp):
    """ Cleans up the dataset for analysis. """
    wines = pd.read_csv(fp).drop_duplicates(subset=DESCRIPTION)
    wines.index = range(len(wines))
    return wines

def tokenize_descriptions(descriptions):
    """ Tokenizes review descriptions to be digested by Word2Vec. """
    return [
        [
            word.lower().translate(None, string.punctuation)
            for word in re.split(' |-', sentence)
        ]
        for sentence in descriptions
    ]

def train_language_model(sentences, min_count=10, size=500, window=10, sample=1e-3):
    """ Uses Word2Vec to train a word embedding, given the reviews as a corpus.
    
    Parameters
    ----
    sentences : list of lists of strings
    min_count : minimum frequency of a term to be trained on/fitted
    size : dimensionality of the Word2Vec vector space
    window : context window when training (narrower focuses on word meaning; wider focuses on topical meaning)
    sample : sampling rate for high frequency words
    """
    model = Word2Vec(sentences, min_count=min_count, size=size, window=window, sample=sample)
    return model

def fit_model_and_compute_description_embeddings(fp='wine.model', model_only=False):
    """ Fits the model, and then uses the model to compute the normalized sum of word vectors for each description."""
    global wines
    global model

    if not model:
        if model_only and os.path.exists(fp):
            model = Word2Vec.load(fp)
        else:
            wines = prepare_dataset(WINE_DATA_FILE_PATH)
            sentences = tokenize_descriptions(list(wines[DESCRIPTION].values))
            if os.path.exists(fp):
                model = Word2Vec.load(fp)
            else:
                model = train_language_model(sentences)
                model.save(fp)
        if not model_only:
            normed = []
            for sentence in sentences:
                summed = np.sum([model.wv[word] for word in np.unique(sentence)
                                 if (word in model.wv)
                                 # and (vocab[word] < 10000)
                                 ], axis=0)
                normed.append(summed / np.sum(summed ** 2.0) ** 0.5)
            wines[EMBEDDING] = normed


if __name__ == '__main__':
    app.run()
