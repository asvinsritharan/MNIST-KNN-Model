import numpy as np

import gzip

from matplotlib import pyplot as plot

from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import classification_report as classification 

import pickle

# open file containing training images
train = gzip.open("train-images-idx3-ubyte.gz", "r")
# read the first 16 bytes of information as they are not images
train.read(16)
# form a matrix which consists of 60000 images of size 28 * 28
buffer = train.read(28 * 28 * 60000)
data = np.frombuffer(buffer, dtype = np.uint8).astype(np.float32)
new_data = data.reshape(60000, 28, 28, 1)

training_data = new_data[0]
training_data = np.reshape(training_data, (784,1))
for i in range (1,60000):
    temp = new_data[i]
    temp = np.reshape(temp, (784, 1))
    training_data = np.append(training_data, temp,1)
# transpose training data so we can use it in model
training_data = training_data.transpose()
# open file containing training labels
train_labels = gzip.open("train-labels-idx1-ubyte.gz", "r")
# read the first 8 bytes of information as they are not labels
train_labels.read(8)
# create an empty array to add labels to
train_labs = np.empty((0,1),int)
# read each byte and add to label list
for i in range (0,60000):
    buffer = train_labels.read(1)
    a = np.frombuffer(buffer, dtype=np.uint8).astype(np.int64)
    train_labs = np.append(train_labs, a)
########################################################################
########################## TEST DATA ###################################
########################################################################
# open file containing training images
test = gzip.open("t10k-images-idx3-ubyte.gz", "r")
# read the first 16 bytes of information as they are not images
test.read(16)
# form a matrix which consists of 60000 images of size 28 * 28
buffer = test.read(28 * 28 * 10000)
data = np.frombuffer(buffer, dtype = np.uint8).astype(np.float32)
new_test = data.reshape(10000, 28, 28, 1)
# reshape each image to be a column vector where each 28 elements of 
# the 784 long vector is one pixel width of an image
testing_data = new_test[0]
testing_data = np.reshape(testing_data, (784,1))
for i in range (1,10000):
    temp = new_test[i]
    temp = np.reshape(temp, (784, 1))
    testing_data = np.append(testing_data, temp,1)
# open file containing training labels
test_labels = gzip.open("t10k-labels-idx1-ubyte.gz", "r")
# read the first 8 bytes of information as they are not labels
test_labels.read(8)
# create an empty array to add labels to
test_labs = np.empty((0,1),int)
# read each byte and add to label list
for i in range (0,10000):
    buffer = test_labels.read(1)
    a = np.frombuffer(buffer, dtype=np.uint8).astype(np.int64)
    test_labs = np.append(test_labs, a)
# transpose testing data so we can use it with model
testing_data = testing_data.transpose()

# create k nearest neighbours model for the training data
model = knn(n_neighbors=8)
model.fit(training_data, train_labs)
# save model
pickle.dump(model, open('knn_model', 'wb'))
# load saved model for testing purposes
model = pickle.load(open('knn_model', 'rb'))
# get predictions using testing data
preds = model.predict(testing_data)
# get the classification test results for the model
print(classification(test_labs, preds))


