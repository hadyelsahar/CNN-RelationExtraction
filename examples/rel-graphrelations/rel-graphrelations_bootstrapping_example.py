__author__ = 'hadyelsahar'

import time
from relationpreprocessor import *
sys.path.append('../../')
from Vectorizers.relationmentionvectorizer import *
from sklearn.metrics import classification_report
from CNN import *
import pickle as pk
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

    if int(args.bootstrap_size) > 0 :
        print "vectorization of bootstrapped data.."
        X_bootstrap = vectorizer.transform(p_bootstrap.X)
        y_bootstrap = p_bootstrap.y
    else:
        X_bootstrap, y_bootstrap = [], []

    print "saving manual annotated data..."
    pk.dump((X, y), file=open(saved_model_path+"dataset.p", 'w'))

    if int(args.bootstrap_size) > 0:
        print "saving bootstrapped data.."
        pk.dump((X_bootstrap, y_bootstrap), file=open(saved_model_path+"dataset_bootstrapped.p", 'w'))

    print "models saved"

else:
    print "preprocessed file exists.. loading.."
    X, y = pk.load(open(saved_model_path+"dataset.p", 'r'))
    if int(args.bootstrap_size) > 0:
        X_bootstrap, y_bootstrap = pk.load(open(saved_model_path+"dataset_bootstrapped.p", 'r'))
    else:
        X_bootstrap, y_bootstrap = [], []


y = np.array(y)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


X_bootstrap_limited = X_bootstrap[0:args.bootstrap_size]
y_bootstrap_limited = y_bootstrap[0:args.bootstrap_size]

# addition of bootstrapping data
x_train = np.concatenate([x_train, X_bootstrap_limited], 0)
y_train = np.concatenate([y_train, y_bootstrap_limited], 0)


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




