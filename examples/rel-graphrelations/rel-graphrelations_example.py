__author__ = 'hadyelsahar'


from relationpreprocessor import *
sys.path.append('../../')
from Vectorizers.relationmentionvectorizer import *
from CNN import *

"""
Preprocessing:
Convert into Inputs into the relationmentionvectorizer:
X: array(dict) [{sentence_id:, id:, segments:[], segment_labels:[], ent1:int, ent2:int}]
y: array(string) : labels of realations
"""

p = RelationPreprocessor()

vectorizer = RelationMentionVectorizer()
vectorizer.fit(p.X)

X = vectorizer.transform(p.X)

# todo use scikit learn cv vectorizer as RelationMentionVectorizer implements scikitlearn interface

X_train = X[0:1100]
X_test = X[1100:]

y_train = p.y[0:1100]
y_test = p.y[1100:]





