# MNIST-KNN-Model
Create a model for detecting handwritten numbers using MNIST dataset

You must download the ubyte files from http://yann.lecun.com/exdb/mnist/ in order to use this model.

A K Nearest Neighbours model is used for classification of the images.

It will take a while to create the model, which then will be saved to a file called knn_model in the working directory.

The saved file can then be opening and used to predict labels for new images of numbers that are to be classified without going through the process of setting up the MNIST dataset into a numpy matrix which is readable by the model. This is done so I don't have to go through the hassle of processing the data again while using an i5 2500k in 2020.

Use the MNIST_file_to_matrices.py to load the MNIST data into testing and training data matrices with labels.

Then run MNIST_model_creator.py to create the KNN model for the dataset

Finally, run the MNIST_number_predictor to get a paint window to write a number, which the script will predict the label of.

You only have to run MNIST_file_to_matrices.py and MNIST_model_creator.py once since they save the model and dataset files. You can run MNIST_number_predictor as many times as you want without having to wait too long for processing.

The time to run each code varies depending on your system. MNIST_file_to_matrices takes about 3 hours on an Intel Core i5 2500k but only about 30-40 minutes on an AMD Ryzen 5 3600.
