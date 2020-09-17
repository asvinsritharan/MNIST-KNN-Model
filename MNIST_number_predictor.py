import numpy as np
import gzip

import cv2


from matplotlib import pyplot as plot
import matplotlib.cm as cm
import joblib
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import classification_report as classification 

import pickle

# allow painting when true
paint = False

# allow recording of mouse position
mode = True


# mouse callback function
def draw(mouse_action,x,y,flags,param):
    # set paint, mode, and current x and y positions as global variables
    # for access outside of function
    global paint, mode, curr_x, curr_y
    # if the left button on the mouse is clicked
    if mouse_action == cv2.EVENT_LBUTTONDOWN:
        # set paint equal to true
        paint = True
        # set the current x and y values as the x and y values of the
        # mouse when the left button is pressed
        curr_x, curr_y = x, y
    # else if mouse has been moved
    elif mouse_action == cv2.EVENT_MOUSEMOVE:
        # if paint is true and mode is true
        if ((paint==True) and (mode==True)):
            # draw a line from the x and y position of when the mouse
            # was first clicked to the point it is at now as a white
            # line with line thickness as 20 px
            cv2.line(image,(curr_x,curr_y),(x,y),(255,255,255),20)
            # set the current x and y values to be that of the current
            # mouse position
            curr_x = x
            curr_y = y
    # if left button of mouse has been lifted
    elif mouse_action == cv2.EVENT_LBUTTONUP:
        # set the paint variable to false
        paint = False
        # if mode is true
        if mode==True:
            # draw a line from the x and y position of when the mouse
            # was first clicked to the point it is at now as a white
            # line with line thickness as 20 px            
            cv2.line(image,(curr_x,curr_y),(x,y),(255,255,255),20)
            # set the current x and y values to be that of the current
            # mouse position            
            curr_x = x
            curr_y = y
    # return the x and y position of the mouse
    return x,y

# load blank image with black background
image = cv2.imread("blank.png")
# name window 
cv2.namedWindow("Draw a number")
# get mouse callback information using draw function
cv2.setMouseCallback('Draw a number',draw)
# print to console instructions to save image
print("Press 'q' to quit and save image as your number")
# loop variable is true
cont = True
# loop while loop variable is true
while cont == True:
    # show the paint window with black background
    cv2.imshow('Draw a number',image)
    # get key press
    key = cv2.waitKey(1) & 0xFF
    # if the q key is pressed
    if (key == ord('q')):
        # save painted image as number.jpg
        cv2.imwrite("number.jpg",image)
        # set loop var to false
        cont = False
# destroy paint window
cv2.destroyAllWindows()
# open the pained number
source = cv2.imread('number.jpg', cv2.IMREAD_GRAYSCALE)
# set width to 28 pixels
w = int(source.shape[1]/10)
# set height to 28 pixels
h = int(source.shape[0]/10)
# create dimension tuple of width and height
dim = (w, h)
# resize picture to fit 28x28 dimension
output = cv2.resize(source, dim)
# write the picture as number_resized.jpg
cv2.imwrite('number_resized.jpg', output)
# read the resized number image
source = cv2.imread('number_resized.jpg', cv2.IMREAD_GRAYSCALE)
# merge 28x28 image represented as a matrix into a 784 x 1 matrix
num_to_predict = np.reshape(source, (784,1))
# transpose matrix
num_to_predict = num_to_predict.transpose()
# open model
model = joblib.load("knn_model.sav")
# create prediction for number
preds = model.predict(num_to_predict)
# print the number
print(preds[0])
