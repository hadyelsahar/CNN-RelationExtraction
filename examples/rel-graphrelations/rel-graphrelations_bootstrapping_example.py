__author__ = 'hadyelsahar'


from relationpreprocessor import *
sys.path.append('../../')
from Vectorizers.relationmentionvectorizer import *
from sklearn.metrics import classification_report
from CNN import *
import pickle as pk
import os
import numpy as np

###########################################
# Preprocessing:
# Convert into Inputs into the relationmentionvectorizer:
# X: array(dict) [{sentence_id:, id:, segments:[], segment_labels:[], ent1:int, ent2:int}]
# y: array(string) : labels of realations
#######################################

if not os.path.exists("/mnt/dataset.p"):
    print "preprocessed data file doesn't exist.. running extraciton process"
    p = RelationPreprocessor()
    p_bootstrap = RelationPreprocessor(inputdir='./data/bootstrap')

    vectorizer = RelationMentionVectorizer()

    # debug with small size bootstrapping data
    p_bootstrap.X = p_bootstrap.X[0:5000]
    p_bootstrap.y = p_bootstrap.y[0:5000]

    print "fitting dataset.."
    vectorizer.fit(np.concatenate([p.X, p_bootstrap.X], 0))
    print "done fitting dataset"
    print "max width of the datasets is %s" % vectorizer.m

    print "vectorization of manual annotated data.."
    X = vectorizer.transform(p.X)

    print "vectorization of bootstrapped data.."
    X_bootstrap = vectorizer.transform(p_bootstrap.X)
    y_bootstrap = p_bootstrap.y
    y = p.y
    print "saving models ..."
    pk.dump((X, y, X_bootstrap, y_bootstrap), file=open("/mnt/dataset.p", 'w'))
    print "models saved"
else:
    print "preprocessed file exists.. loading.."
    X, y, X_bootstrap, y_bootstrap = pk.load(open("/mnt/dataset.p", 'r'))

# todo use scikit learn cv vectorizer as RelationMentionVectorizer implements scikitlearn interface

y = np.array(y)

x_train = np.concatenate([X[0:1700], X_bootstrap], 0)
x_test = X[1700:]

y_train = np.concatenate([y[0:1700], y_bootstrap], 0)
y_test = y[1700:]

print "size of dataset is : %s \n" \
      "bootstrapping size  : %s \n" \
      "training size: %s \n " \
      "testing size: %s " \
      % (len(y), len(y_bootstrap), len(y_train), len(y_test))

print "now training..."

####################
# Training the CNN #
####################
max_w = X.shape[1]
print max_w
# max_w = 27

x_train = np.reshape(x_train, [-1, max_w, 320, 1])
x_test = np.reshape(x_test, [-1, max_w, 320, 1])


y_classes = np.unique(np.concatenate([y, y_bootstrap], 0))

cnn = CNN(input_shape=[max_w, 320, 1], classes=y_classes, conv_shape=[4, 320], epochs=2500)
cnn.fit(x_train, y_train, x_test, y_test)

print "done training"
print "testing"

# Testing :
###########
y_pred = cnn.predict(x_test)

print classification_report(y_test, y_pred)




