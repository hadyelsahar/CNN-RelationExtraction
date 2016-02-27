__author__ = 'hadyelsahar'


from relationpreprocessor import *
sys.path.append('../../')
from Vectorizers.relationmentionvectorizer import *
from sklearn.metrics import classification_report
from CNN import *
import pickle as pk
import os
from IPython.core.debugger import Tracer
from sklearn.cross_validation import train_test_split


###########################################
# Preprocessing:
# Convert into Inputs into the relationmentionvectorizer:
# X: array(dict) [{sentence_id:, id:, segments:[], segment_labels:[], ent1:int, ent2:int}]
# y: array(string) : labels of realations
#######################################

if not os.path.exists("./saved-models/dataset.p"):
    print "preprocessed data file doesn't exist.. running extraciton process"
    p = RelationPreprocessor()
    vectorizer = RelationMentionVectorizer(threads=7)
    vectorizer.fit(p.X)
    print "vectorizing data.."
    X = vectorizer.transform(p.X)
    y = p.y
    print "done vectorizing.."
    pk.dump((X, y), file=open("./saved-models/dataset.p", 'w'))
else:
    print "preprocessed file exists.. loading.."
    X, y = pk.load(open("./saved-models/dataset.p", 'r'))

# todo use scikit learn cv vectorizer as RelationMentionVectorizer implements scikitlearn interface

y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


print "size of dataset is : %s" % len(y)
print "now training..."

####################
# Training the CNN #
####################
max_w = X.shape[1]
print max_w
# max_w = 27

x_train = np.reshape(x_train, [-1, max_w, 320, 1])
x_test = np.reshape(x_test, [-1, max_w, 320, 1])

cnn = CNN(input_shape=[max_w, 320, 1], classes=np.unique(y), conv_shape=[4, 320], epochs=2500)
cnn.fit(x_train, y_train, x_test, y_test)

print "done training"
print "testing"

# Testing :
###########
y_pred = cnn.predict(x_test)

print classification_report(y_test, y_pred)




