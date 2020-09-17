import numpy as np
import gzip

import cv2

import joblib

from matplotlib import pyplot as plot
import matplotlib.cm as cm

from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import classification_report as classification 

import pickle

# load training data
training_data = pickle.load(open('training_data', 'rb'))
# load training labels
training_labels = pickle.load(open('training_labels', 'rb'))
# load testing labels
testing_labels = pickle.load(open('test_labs', 'rb'))
# load testing data
testing_data = pickle.load(open('testing_data', 'rb'))
# create knn model with 10 neighbours, 1 for each number 0-9
model = knn(n_neighbors=10)
# fit model with training data and labels
model.fit(training_data, training_labels)

# save knn model
joblib.dump(model, "knn_model.sav")  

# get predictions for testing data
preds = model.predict(testing_data)

# get the classification test results for the model
print(classification(testing_labels, preds))