import numpy as np

import gzip

from matplotlib import pyplot as plot
import matplotlib.cm as cm
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import classification_report as classification 

import pickle
########################################################################
########################## TRAINING DATA ###############################
########################################################################
# open file containing training images
train = gzip.open("train-images-idx3-ubyte.gz", "r")
# read the first 16 bytes of information as they are not images
train.read(16)
# form a matrix which consists of 60000 images of size 28 * 28
buffer = train.read(28 * 28 * 60000)
data = np.frombuffer(buffer, dtype = np.uint8).astype(np.float32)
new_data = data.reshape(60000, 28, 28, 1)
a = new_data.shape
# get one image from training image gz file and merge it from 28x28 matrix to
# a 784x1 matrix
training_data = new_data[0]
training_data = np.reshape(training_data, (784,1))
# do the same for the next 59999 images
for i in range (1,60000):
    # get i'th image
    temp = new_data[i]
    # reshape from 28x28 matrix to 784x1 matrix
    temp = np.reshape(temp, (784, 1))
    # add it to training data matrix
    training_data = np.append(training_data, temp,1)
    # print i every 1000 steps
    if (i%1000 == 0):
        print(i)
# transpose training data so we can use it in model
training_data = training_data.transpose()
# dump training data
pickle.dump(training_data, open('training_data', 'wb'))
# open file containing training labels
train_labels = gzip.open("train-labels-idx1-ubyte.gz", "r")
# read the first 8 bytes of information as they are not labels
train_labels.read(8)
# create an empty array to add labels to
train_labs = np.empty((0,1),int)
# read each byte and add to label list
for i in range (0,60000):
    # get i'th label
    buffer = train_labels.read(1)
    a = np.frombuffer(buffer, dtype=np.uint8).astype(np.int64)
    # add label to training label list
    train_labs = np.append(train_labs, a)
    # print i every 1000 steps
    if (i%1000 == 0):
        print(i)    
# dump training labels
pickle.dump(train_labs, open('training_labels', 'wb'))
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
    # get i'th image
    temp = new_test[i]
    # reshape from 28x28 matrix to 784x1 matrix
    temp = np.reshape(temp, (784, 1))
    # add the testing data matrix
    testing_data = np.append(testing_data, temp,1)
    # print i every 1000 steps
    if (i%1000 == 0):
        print(i)    

# open file containing training labels
test_labels = gzip.open("t10k-labels-idx1-ubyte.gz", "r")
# read the first 8 bytes of information as they are not labels
test_labels.read(8)
# create an empty array to add labels to
test_labs = np.empty((0,1),int)
# read each byte and add to label list
for i in range (0,10000):
    # get i'th label
    buffer = test_labels.read(1)
    a = np.frombuffer(buffer, dtype=np.uint8).astype(np.int64)
    # add label to training label list    
    test_labs = np.append(test_labs, a)
    # print i every 1000 steps
    if (i%1000 == 0):
        print(i)
# dump testing labels
pickle.dump(test_labs, open('test_labs', 'wb'))
# transpose testing data so we can use it with model
testing_data = testing_data.transpose()
# dump testing data
pickle.dump(testing_data, open('testing_data', 'wb'))


