#! /usr/bin/python3
from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np
import pdb
import pickle

from scipy.spatial.distance import cdist

# articles is a list of (title, text) tuples.
def group(articles):
    def decode_bytes(bytes):
        if bytes.__class__ == ''.__class__:
            return bytes
        return bytes.decode('utf-8')
    def decode_article(article):
        title = ''
        text = ''
        if article[0] != None:
            title = decode_bytes(article[0])
        if article[1] != None:
            text = decode_bytes(article[1])
        return (title, text)
    articles = [decode_article(a) for a in articles if a != None]
    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    
    use_hashing = False
    n_components = None
    verbose = False
    n_features = 10000
    minibatch = True
    use_idf = True
    
    #dataset = fetch_20newsgroups(subset='all', categories=categories,
    #                             shuffle=True, random_state=42)
    # dataset should be a list of document strings.
    # Include the title by prepending it to the document
    dataset = [article[0] + ' ' + article[1] for article in articles]
    
    print("%d documents" % len(dataset))
    print()
    
    #labels = dataset.target
    labels = ["label1", "label2", "label3", "label4"]
    true_k = np.unique(labels).shape[0]
    
    print("Extracting features from the training dataset using a sparse vectorizer")
    t0 = time()
    if use_hashing:
        if use_idf:
            # Perform an IDF normalization on the output of HashingVectorizer
            hasher = HashingVectorizer(n_features=n_features,
                                       stop_words='english', non_negative=True,
                                       norm=None, binary=False)
            vectorizer = make_pipeline(hasher, TfidfTransformer())
        else:
            vectorizer = HashingVectorizer(n_features=n_features,
                                           stop_words='english',
                                           non_negative=False, norm='l2',
                                           binary=False)
    else:
        vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features,
                                     min_df=2, stop_words='english',
                                     use_idf=use_idf)
    X = vectorizer.fit_transform(dataset)
    
    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)
    print()
    
    if n_components:
        print("Performing dimensionality reduction using LSA")
        t0 = time()
        # Vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        svd = TruncatedSVD(n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
    
        X = lsa.fit_transform(X)
    
        print("done in %fs" % (time() - t0))
    
        explained_variance = svd.explained_variance_ratio_.sum()
        print("Explained variance of the SVD step: {}%".format(
            int(explained_variance * 100)))
    
        print()
    
    
    ###############################################################################
    # Do the actual clustering
    
    if minibatch:
        km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                             init_size=1000, batch_size=1000, verbose=verbose)
    else:
        km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                    verbose=verbose)
    
    print("Clustering sparse data with %s" % km)
    t0 = time()
    km.fit(X)
    print("done in %0.3fs" % (time() - t0))
    print()
    
    # Don't try to score the result because this is unsupervised
    #print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    #print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    #print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    #print("Adjusted Rand-Index: %.3f"
    #      % metrics.adjusted_rand_score(labels, km.labels_))
    #print("Silhouette Coefficient: %0.3f"
    #      % metrics.silhouette_score(X, km.labels_, sample_size=1000))
    
    print()
    
    
    if not use_hashing:
        print("Top terms per cluster:")
    
        if n_components:
            original_space_centroids = svd.inverse_transform(km.cluster_centers_)
            order_centroids = original_space_centroids.argsort()[:, ::-1]
        else:
            order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        for i in range(true_k):
            print("Cluster %d:" % i, end='')
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind], end='')
            print()
        # Next step: Use elbow method to determine optimal number of clusters.
        # http://stackoverflow.com/questions/6645895/calculating-the-percentage-of-variance-measure-for-k-means
        # D_k = [cdist(X, cent, 'euclidean') for cent in order_centroids]
        # cIdx = [np.argmin(D,axis=1) for D in D_k]
        # dist = [np.min(D,axis=1) for D in D_k]
        # avgWithinSS = [sum(d)/X.shape[0] for d in dist]

if __name__ == "__main__":
    articles = pickle.load(open("articles.pickle", "rb"))
    group(articles)

