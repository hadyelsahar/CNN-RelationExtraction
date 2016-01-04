__author__ = 'hadyelsahar'

import sys

from sklearn.metrics import classification_report

import input_data

sys.path.append('../../')
from CNN import *


# Reading mnist Data :
######################
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Training :
############
x_train = np.reshape(mnist.train.images, [-1, 28, 28, 1])

# converting mnist correct labels 1 hot vectors into data into ids of correct labels
y_train = mnist.train.labels
y_train = [np.where(i == 1)[0][0] for i in y_train]

classes = np.unique(y_train)
cnn = CNN(input_shape=[28, 28, 1], classes=classes, conv_shape=[5, 5])

cnn.fit(x_train, y_train)

# Testing :
###########
x_test = np.reshape(mnist.test.images, [-1, 28, 28, 1])
y_pred = cnn.predict(x_test)

y_true = mnist.test.labels
y_true = [list(i).index(1) for i in y_true]

print classification_report(y_true, y_pred)


