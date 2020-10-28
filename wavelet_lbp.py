# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 21:25:09 2020

@author: ataka
"""
import numpy as np
import matplotlib.pyplot as plt
import pywt
import pywt.data
import os
from skimage.color import rgb2gray
from os import listdir
import datetime
from matplotlib.pyplot import imread
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq
a = datetime.datetime.now()
num_class = 4
directory = 'dataset'
METHOD = 'uniform'
radius = 3
n_points = 8 * radius
feature_vector = []
classes = []
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
labels = listdir(directory)
for i, label in enumerate(labels):
    datas_path = directory+'/'+label
    for file_name in os.listdir(datas_path):
        if file_name.endswith(".tif"):
            a = datetime.datetime.now()
            img = imread(datas_path+'/'+file_name)
            img = rgb2gray(img)
            img = 255 * img
            img = img.astype(np.uint8)
            #img = cv2.imread(datas_path+'/'+file_name)  
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #img = cv2.medianBlur(img,5)
            coeffs2 = pywt.dwt2(img, 'rbio3.1')
            LL, (LH, HL, HH) = coeffs2
            LL = LL.astype(np.uint8) #features were extracted from Approximation coefficient
            lbp = local_binary_pattern(LL, n_points, radius, METHOD)
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
b = datetime.datetime.now()
c = b-a
c.microseconds
np.save('extracted_features/rbio3.1_wave_lbp_features.npy', feature_vector)
np.save('extracted_features/rbio3.1_wave_lbp_features_2.npy', classes)
X = np.load('extracted_features/rbio3.1_wave_lbp_features.npy')
Y = np.load('extracted_features/rbio3.1_wave_lbp_features_2.npy')
testsize = 0.2      ##test size parameter
"""
SVM
"""
#Splitting data to train and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=testsize, random_state=42)
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]


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


Y_test = to_categorical(Y_test, num_class)

# confusion matrix
import datetime
from sklearn.metrics import confusion_matrix
import seaborn as sns
# Predict the values from the validation dataset
#a = datetime.datetime.now()
Y_pred = cv.predict(X_test)
#b = datetime.datetime.now()
#c = b-a
#c.microseconds
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_test,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("WAVE-GLCM - approximation Confusion Matrix")
plt.show()

target_names = ['class 0', 'class 1', 'class 2', 'class 3']
print(classification_report(Y_true, Y_pred, target_names=target_names))


"""

### Bir resmin wavelistteki tüm parametrelerle coefficientlarının görselleştirilmesi
img=plt.imread('c1r1e4n1_0h_aug799.tif')
img = rgb2gray(img)
img = 255 * img
img = img.astype(np.uint8)
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
lizt = []
lizt = pywt.wavelist(kind = 'discrete')
for r,k in enumerate(lizt):
    
    coeffs2 = pywt.dwt2(img, lizt[r])
    
    LL, (LH, HL, HH) = coeffs2
    
    fig = plt.figure(figsize=(12, 3))
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.xlabel(k)
    
    fig.tight_layout()
    plt.show()
    

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(img, cmap=plt.cm.gray)
ax1.set_title('Input image')
"""








