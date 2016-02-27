__author__ = 'hadyelsahar'

import time
from relationpreprocessor import *
sys.path.append('../../')
from Vectorizers.relationmentionvectorizer import *
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from CNN import *
import cPickle as pk
import os
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='relation extraction using bootstrapped training set and CNN')
parser.add_argument('-bs', '--bootstrap_size', help='size of the bootstrapping 0 for no bootstrapping', required=False)
parser.add_argument('-smp', '--saved_model_path', help='saved model folder name', required=True)
args = parser.parse_args()


###########################################
# Preprocessing:
# Convert into Inputs into the relationmentionvectorizer:
# X: array(dict) [{sentence_id:, id:, segments:[], segment_labels:[], ent1:int, ent2:int}]
# y: array(string) : labels of realations
#######################################

fout = open('experiment-results-%s.txt' % time.strftime("%d-%m-%Y-%I:%M:%S"), 'w')

saved_model_path = args.saved_model_path if args.saved_model_path.endswith("/") else args.saved_model_path+"/"

if not os.path.exists(saved_model_path+"dataset.p"):
    print "preprocessed data file doesn't exist.. running extraciton process"

    p = RelationPreprocessor()
    vectorizer = RelationMentionVectorizer(threads=30)

    if int(args.bootstrap_size) > 0:

        p_bootstrap = RelationPreprocessor(inputdir='./data/bootstrap')

        print "reducing bootstrapping sizes for experiments time"
        selector = np.where((p_bootstrap.y  != "NoEdge") & (p_bootstrap.y  != "c_p") & (p_bootstrap.y  != "conj") & (p_bootstrap.y  != "coref") & (p_bootstrap.y  != "poss"))[0]
        selector = selector[:10000]

        p_bootstrap.y = p_bootstrap.y[selector]
        p_bootstrap.X = p_bootstrap.X[selector]

        print "fitting dataset.."
        vectorizer.fit(np.concatenate([p.X, p_bootstrap.X], 0))
        print "done fitting dataset"

    else:
        print "fitting dataset.."
        vectorizer.fit(p.X)
        print "done fitting dataset"

    print "max width of the datasets is %s" % vectorizer.m

    print "vectorization of manual annotated data.."
    X = vectorizer.transform(p.X)
    y = p.y

    if int(args.bootstrap_size) > 0:
        print "vectorization of bootstrapped data.."
        X_bootstrap = vectorizer.transform(p_bootstrap.X)
        y_bootstrap = p_bootstrap.y
    else:
        X_bootstrap, y_bootstrap = [], []

    print "saving manual annotated data..."
    np.save(open(saved_model_path+"dataset_X.p", 'w'), X)
    np.save(open(saved_model_path+"dataset_y.p", 'w'), y)

    if int(args.bootstrap_size) > 0:
        print "saving bootstrapped data.."
        np.save(open(saved_model_path+"dataset_X_boot.p", 'w'), X_bootstrap)
        np.save(open(saved_model_path+"dataset_y_boot.p", 'w'), y_bootstrap)

    print "models saved"

else:
    print "preprocessed file exists.. loading.."
    X = np.load(open(saved_model_path+"dataset_X.p", 'r'))
    y = np.load(open(saved_model_path+"dataset_y.p", 'r'))

    if int(args.bootstrap_size) > 0:
        X_bootstrap = np.load(open(saved_model_path+"dataset_X_boot.p", 'r'))
        y_bootstrap = np.load(open(saved_model_path+"dataset_y_boot.p", 'r'))

    else:
        X_bootstrap, y_bootstrap = [], []


y = np.array(y)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


X_bootstrap_limited = X_bootstrap[0:args.bootstrap_size]
y_bootstrap_limited = y_bootstrap[0:args.bootstrap_size]

# addition of bootstrapping data
x_train = np.concatenate([x_train, X_bootstrap_limited], 0)
y_train = np.concatenate([y_train, y_bootstrap_limited], 0)


otherselector_indices = np.where((y_train != "NoEdge") & (y_train != "c_p") & (y_train != "conj") & (y_train != "coref") & (y_train != "poss"))[0]
NoEdge_indices = np.where((y_train  == "NoEdge"))[0][0:len(otherselector_indices)/5]  # training with only half total size

selector = np.append(otherselector_indices,NoEdge_indices)
x_train = x_train[selector]
y_train = y_train[selector]





print "size of dataset is : %s \n" \
      "bootstrapping size  : %s \n" \
      "limited to  : %s ]n" \
      "training size: %s \n " \
      "testing size: %s " \
      % (len(y), len(y_bootstrap), args.bootstrap_size, len(y_train), len(y_test))

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

cnn = CNN(input_shape=[max_w, 320, 1], classes=y_classes, conv_shape=[4, 55], epochs=20000)
cnn.fit(x_train, y_train, x_test, y_test)

print "done training"
print "testing"

# Testing :
###########
y_pred = np.array([])

for c, t in enumerate(Batcher.chunks(x_test, 100)):
    y_pred = np.append(y_pred, cnn.predict(t))

classification_rep = classification_report(y_test, y_pred)
print classification_rep

fout.write(classification_rep+"\n")
fout.flush()

fout.close()




