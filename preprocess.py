#! /usr/bin/python3
import nltk
import pdb
import pickle
import pandas as pd
import numpy as np
import json

stemmer = nltk.stem.porter.PorterStemmer()
stop_words = set(nltk.corpus.stopwords.words('english'))

def is_alphanumeric(character):
    to_ord = ord(character)
    is_alpha = (to_ord >= ord('A') and to_ord <= ord('Z')) or (to_ord >= ord('a') and to_ord <= ord('z'))
    is_numeric = to_ord >= ord('0') and to_ord <= ord('9')
    return is_alpha or is_numeric

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def reduce_stem(stem):
    # Remove unwanted characters such as punctuations
    reduced = []
    for character in stem:
        if not is_alphanumeric(character):
            continue
        reduced.append(character)
    return ''.join(reduced)

def tokenize(text):
    text = text.decode('utf-8').lower()
    # Replace periods with spaces. This fixes cases
    # where there's no space after a period. Punctuation
    # will be dropped later in processing, anyway.
    text.replace('.', ' ')
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    # Remove punctuations and stop words
    stems_reduced = []
    for ix, stem in enumerate(stems):
        if stem in stop_words:
            continue
        reduced_stem = reduce_stem(stem)
        if len(reduced_stem) > 0:
            stems_reduced.append(reduced_stem)
    return stems_reduced

articles = []
article_id = 0
def cache_for_analysis(url, title, stems, feed_id):
    global article_id
    article_id = article_id + 1
    articles.append((article_id, url, title, stems, feed_id))

def dump_articles():
    pickle.dump(articles, open("articles.pickle", "wb"))

def article_to_json(row, all_terms):
    stems = row[3]
    vec = [("1" if term in stems else "0") for term in all_terms]
    vec_string = "".join(vec)
    return {
        "article_id": row[0],
        "url": row[1],
        "title": row[2],
        "feed_id": row[4],
        "vec": vec_string
    }

def analyze_articles():
    seen_stems = set()
    # do some kind of clustering with tf/idf!
    # build up a data frame with schema:
    # terms (string) | a1_terms (bool) | a2_terms
    for row in articles:
        stems = row[3]
        for stem in stems:
            seen_stems.add(stem)
    all_terms = list(seen_stems)
    for row in articles:
        json_article = article_to_json(row, all_terms)
        print(json.dumps(json_article))

if __name__ == "__main__":
    articles = pickle.load(open("articles.pickle", "rb"))
    analyze_articles()

