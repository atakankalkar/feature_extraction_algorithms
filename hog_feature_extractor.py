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
from keras.utils import to_categorical

num_class = 4
directory = 'dataset'
feature_vector = []
classes = []
labels = listdir(directory)
for i, label in enumerate(labels):
    datas_path = directory+'/'+label
    for file_name in os.listdir(datas_path):
        if file_name.endswith(".tif"):
            img = imread(datas_path+'/'+file_name)
            img = rgb2gray(img)
            fd, hog_image = hog(img, orientations=9, pixels_per_cell=(64,64),cells_per_block=(1, 1), visualize=True, multichannel=False,feature_vector = True)
            feature_vector.append(fd)
            classes.append(i)
          
feature_vector = np.array(feature_vector).astype('float32') 
classes = np.array(classes).astype('float32')
 ## CATEGORİCAL classes = to_categorical(classes, num_class)
np.save('extracted_features/hog_features.npy', feature_vector)
np.save('extracted_features/hog_classes.npy', classes)



X = np.load('extracted_features/hog_features.npy')
Y = np.load('extracted_features/hog_classes.npy')
from os import listdir
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
testsize = 0.2      ##test size parameter
"""
SVM
"""
#Splitting data to train and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=testsize, random_state=42)
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]


""" PCA (DOESN'T WORK PROPERLY) """
#pca = PCA(0.99)
""" PCA apply """
#pca.fit(X_train)
""" Pca test and train transform """
#X_train = pca.transform(X_train)
#X_test = pca.transform(X_test)

""


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




import matplotlib.pyplot as plt
""" HOG APPLIED IMAGE VISUALIZATION


from skimage.feature import hog
from skimage import data, exposure

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(img, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()

"""
Y_test = to_categorical(Y_test, num_class)


# confusion matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns


# Predict the values from the validation dataset
Y_pred = cv.predict(X_test)

# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_test,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix of Validation")
plt.show()





