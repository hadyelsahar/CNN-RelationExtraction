__author__ = 'hadyelsahar'


from relationpreprocessor import *
sys.path.append('../../')
from Vectorizers.relationmentionvectorizer import *
from sklearn.metrics import classification_report
from CNN import *
import cPickle as pk
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
    vectorizer = RelationMentionVectorizer(threads=32)
    vectorizer.fit(p.X)
    print "vectorizing data.."
    X = vectorizer.transform(p.X)
    y = p.y
    print "done vectorizing.."
    np.save(open("./saved-models/dataset_X.p", 'w'), X)
    np.save(open("./saved-models/dataset_y.p", 'w'), y)

else:
    print "preprocessed file exists.. loading.."
    X = np.load(open("./saved-models/dataset_X.p", 'w'))
    y = np.load(open("./saved-models/dataset_y.p", 'w'))


y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

otherselector_indices = np.where((y_train != "NoEdge") & (y_train != "c_p") & (y_train != "conj") & (y_train != "coref") & (y_train != "poss"))[0]
NoEdge_indices = np.where((y_train  == "NoEdge"))[0][0:len(otherselector_indices)/2]  # training with only half total size

selector = np.append(otherselector_indices,NoEdge_indices)
x_train = x_train[selector]
y_train = y_train[selector]


print "size of dataset is : %s" % len(y)
print "now training..."

####################
# Training the CNN #
####################
max_w = X.shape[1]
print max_w

x_train = np.reshape(x_train, [-1, max_w, 320, 1])
x_test = np.reshape(x_test, [-1, max_w, 320, 1])

cnn = CNN(input_shape=[max_w, 320, 1], classes=np.unique(y), conv_shape=[4, 55], epochs=2500)
cnn.fit(x_train, y_train, x_test, y_test)

print "done training"
print "testing"

# Testing :
###########
y_pred = cnn.predict(x_test)

print classification_report(y_test, y_pred)




