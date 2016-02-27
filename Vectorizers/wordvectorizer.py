# -*- coding: utf-8 -*-
__author__ = 'hadyelsahar'


import numpy as np
import gensim

from sklearn.base import TransformerMixin
from nltk.tokenize import TreebankWordTokenizer
from os import path

_W2V_BINARY_PATH = path.dirname(path.abspath(__file__)) + "/word2vec/GoogleNews-vectors-negative300.bin.gz"

class WordVectorizer(TransformerMixin):

    def __init__(self, ner=True, pos=True, dependency=False, embeddings="word2vec", tokenizer=None, w2v_path=_W2V_BINARY_PATH):
        """

        :param ner: Boolean indicating adding named entity recognition features in the feature vector or not
        :param pos: Boolean indicating adding part of speech tagging in the features in the feature vector or not
        :param dependency: Boolean to add dependency features or not
        :param embeddings: name of which word vectors to use, default word2vec
        :return:
        """
        if tokenizer is None:
            self.tokenize = TreebankWordTokenizer().tokenize

        if embeddings == "word2vec":
            print "loading word2vec model ..."
            self.model = gensim.models.Word2Vec.load_word2vec_format(w2v_path, binary=True)
            print "finished loading word2vec !!"

        self.ner = ner
        self.pos = pos
        self.dependency = dependency


    def transform(self, sentences, **transform_params):
        """
        :param X: iterator of sentences
        :param transform_params:
        :return: csr matrix each row contains a word (tokenized using standard tokenizer)
         in sequence and columns indicating feature vector.
        """
        feature_vector_size = 0
        if self.pos:
            feature_vector_size += 1
        if self.ner:
            feature_vector_size += 1
        if self.model:
            feature_vector_size += self.model.vector_size

        # large matrix containing words per row and features per column
        X = np.zeros((0, feature_vector_size), np.float32)
        word_list = []

        for s in sentences:

            tokens = self.tokenize(s)
            word_list += tokens

            # features for words per sentence # empty now hstack later
            words_features = np.zeros((len(tokens), 0))

            if self.model:
                wordvec = np.zeros((len(tokens), self.model.vector_size), np.float32)
                for i, w in enumerate(tokens):
                    wordvec[i] = self.word2vec(w)
                words_features = np.hstack([words_features, wordvec])

            if self.pos:
                # todo
                pass
                # posvec = np.zeros((len(tokens), 1), np.float32)
                # words_features = np.hstack([words_features, posvec])


            if self.ner:
                # todo
                pass
                # nervec = np.zeros((len(tokens), 1), np.float32)
                # words_features = np.hstack([words_features, nervec])

            if self.dependency:
                # todo
                # depvec = np.zeros((len(tokens), 1), np.float32)
                # words_features = np.hstack([words_features, depvec)
                pass

            # matrix nxm
            # n : number of words in in a sentence
            # m : number of features per word
            X = np.vstack([X, words_features])

        return X, word_list

    def fit(self, X, y=None, **fit_params):
        return self


    def word2vec(self, phrase):
        """
        using loaded word2vec model given a vector return it's equivalent word2vec representation
         - if word is not existing, replace by zero vector
         - if word contain one or many words inside tokenize and use average representations of all vectors
            (used to generate word vectors for segments of multiple words as well)
         - to do :
        :param phrase: a word or a phrase of multiple words
        :return: raw numpy vector of a word dtype = float32
        """
        def lookup(word):
            if word in self.model:
                return self.model[word]
            else:
                return np.zeros(self.model.vector_size, np.float32)

        words = self.tokenize(phrase)
        return np.average([lookup(w) for w in words], axis = 0)


