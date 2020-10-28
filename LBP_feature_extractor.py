# -*- coding: utf-8 -*-
"""
Created on Thu May 28 01:03:09 2020

@author: ataka
"""

import numpy as np
from skimage.color import rgb2gray
from matplotlib.pyplot import imread
from skimage import data
from skimage.feature import hog
import os
from os import listdir
from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from keras.utils import to_categorical
import matplotlib.pyplot as plt
METHOD = 'uniform'
num_class = 4
directory = 'dataset'
radius = 3
n_points = 8 * radius
feature_vector = []
classes = []
labels = listdir(directory)
for i, label in enumerate(labels):
    datas_path = directory+'/'+label
    for file_name in os.listdir(datas_path):
        if file_name.endswith(".tif"):
            img = imread(datas_path+'/'+file_name)
            img = rgb2gray(img)
            lbp = local_binary_pattern(img, n_points, radius, METHOD)
            n_bins = int(lbp.max() + 1)
            x = itemfreq(lbp.ravel())
            # Normalize the histogram
            hist = x[:, 1]/sum(x[:, 1])
            hist=hist.reshape(1,hist.shape[0])
            #fd = np.array(fd).astype('float64') 
            feature_vector.append(hist)
            classes.append(i)
          
feature_vector = np.array(feature_vector).astype('float32') 
feature_vector=feature_vector.reshape(feature_vector.shape[0],feature_vector.shape[2]*feature_vector.shape[1])
classes = np.array(classes).astype('float32')
 ## CATEGORİCAL classes = to_categorical(classes, num_class)
np.save('extracted_features/lbp_features.npy', feature_vector)
np.save('extracted_features/lbp_classes.npy', classes)


X = np.load('extracted_features/lbp_features.npy')
Y = np.load('extracted_features/lbp_classes.npy')


testsize = 0.2      ##test size parameter
"""
SVM
"""
#Splitting data to train and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=testsize, random_state=42)
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]

""" Scaler normalizasyon , mean = 0 and variance = 1 """
"""
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(X)
# Apply transform to both the training set and the test set.
"""



steps = [('scalar', StandardScaler()),
         ('SVM', SVC())]
pipeline = Pipeline(steps)
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}
cv = GridSearchCV(pipeline,param_grid=parameters,cv=5)
cv.fit(X_train,Y_train)

y_pred = cv.predict(X_test)

print("Accuracy: {}".format(cv.score(X_test, Y_test)))
print("Tuned Model Parameters: {}".format(cv.best_params_))


#Y_test = to_categorical(Y_test, num_class)

# confusion matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns
# Predict the values from the validation dataset
Y_pred = cv.predict(X_test)
# Convert validation observations to one hot vectors
#Y_true = np.argmax(Y_test,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_test, Y_pred) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix of Validation")
plt.show()



