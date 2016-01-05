__author__ = 'hadyelsahar'


from relationpreprocessor import *
sys.path.append('../../')
from Vectorizers.relationmentionvectorizer import *
from CNN import *
import pickle as pk
import os
###########################################
# Preprocessing:
# Convert into Inputs into the relationmentionvectorizer:
# X: array(dict) [{sentence_id:, id:, segments:[], segment_labels:[], ent1:int, ent2:int}]
# y: array(string) : labels of realations

#######################################
if not os.path.exists("./dataset.p"):
    p = RelationPreprocessor()
    vectorizer = RelationMentionVectorizer()
    vectorizer.fit(p.X)
    X = vectorizer.transform(p.X)
    y = p.y
    pk.dump((X, y), file=open("./dataset.p", 'w'))
else:
    X, y = pk.load(open("./dataset.p", 'r'))

# todo use scikit learn cv vectorizer as RelationMentionVectorizer implements scikitlearn interface


y = np.array(y)

x_train = X[0:1100]
x_test = X[1100:]

y_train = y[0:1100]
y_test = y[1100:]


####################
# Training the CNN #
####################
x_train = np.reshape(x_train, [-1, 20, 320, 1])
cnn = CNN(input_shape=[20, 320, 1], classes=np.unique(y), conv_shape=[5, 25])
cnn.fit(x_train, y_train)


# Testing :
###########
x_test = np.reshape(x_test, [-1, 20, 320, 1])
y_pred = cnn.predict(x_test)

# y_true = [list(i).index(1) for i in y_true]

# print classification_report(y_true, y_pred)




