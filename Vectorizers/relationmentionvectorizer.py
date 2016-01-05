__author__ = 'hadyelsahar'

from sklearn.base import TransformerMixin
import wordvectorizer
import numpy as np
from wordvectorizer import WordVectorizer


# inputs :
# segments            : array of strings of max length m (padding smaller sizes sequences with zeros)
# segments labels     : array of strings
# ent1,ent2 : position of entity1 and entity2 in segments    0 <= ent1, ent2 < m
# outputs :
# vector representation of each segment   : mxn martix  m = len(segments+padding), n = len(Wvec + position_vec + features)

class RelationMentionVectorizer(TransformerMixin):

    def __init__(self, position_vector=True, Wposition_size = 10, ner=False, pos=False, dependency=False):

        self.position_vector = position_vector

        # Wposition vectors will be filled when calling fit function
        self.Wposition = None
        self.Wposition_size = Wposition_size
        self.wordvectorizer = WordVectorizer(ner=ner, pos=pos, dependency=dependency)

        # sizes of the output sequence matrix m is number of words in the sequence
        # n is the size of the vector representation of each word in the sequence
        self.m = None
        self.n = self.wordvectorizer.model.vector_size + 2*self.Wposition_size

    def transform(self, X, **transform_params):
        """
        :param X: array(dict) [{segments:[],segment_labels:[],ent1:int, ent2:int}]
        :param transform_params:
        :return:
        """

        X_out = np.zeros([0, self.m, self.n], np.float32)

        for i in X:

            # padding with zeros
            x1 = [self.wordvectorizer.word2vec(w) for w in i["segments"]]
            x1 = np.array(x1, dtype=np.float32)

            padsize = self.m - x1.shape[0]

            if padsize > 0:
                temp = np.zeros((padsize, self.wordvectorizer.model.vector_size))

                x1 = np.vstack([x1, temp])

            # position with respect to ent1
            x2 = self.Wposlookup(i["ent1"])     # dimension m x _
            # position with respect to ent2
            x3 = self.Wposlookup(i["ent2"])     # dimension m x _

            x = np.hstack([x1, x2, x3])         # merging different parts of vector representation of words

            X_out = np.append(X_out, [x], axis=0)

        return X_out



    def fit(self, X, y=None, **fit_params):
        """
        :param X: array(dict) [{segments:[], segment_labels:[], ent1:int, ent2:int}]
        :param y:
        :param fit_params:
        :return:
        """

        l = max([len(i["segments"]) for i in X])
        self.m = l

        # original index = -l+1,....,0,...l-1
        # array index    = 0,.......,(l-1),...(2xl)-1
        self.Wposition = np.random.rand((2*l)-1, self.Wposition_size)



        return self

    def Wposlookup(self, p):
        """

        :param p: position of entity
        :return: array of dimension self.m x self.Wposition_size

        example : if ent1 = 2 self.m = 10   i.e. : (w0, w1, w2(e1), w3, w4, w5, w6, w7, w8, w9)
                  return: Wposition[-2:8]   === add (l-1) to get indices between (0,2l-1) ===>  Wposition[7:17]
        """
        start = -p + self.m - 1
        end = start + self.m

        return self.Wposition[start:end]










