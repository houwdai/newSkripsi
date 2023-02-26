#import lib
from multiprocessing import Value
from this import d
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import os
mypath = './dataset/'
file_name = []
tag = []
full_path = []
for path, subdirs, files in os.walk(mypath):
    for name in files:
        full_path.append(os.path.join(path, name)) 
        tag.append(path.split('/')[-1])        
        file_name.append(name)

import pandas as pd
df_datasetpre = pd.DataFrame({"path":full_path,'file_name':file_name,"tag":tag})
#print(df_datasetpre)

#load library untuk train test split
from sklearn.model_selection import train_test_split
X= df_datasetpre['path']
y= df_datasetpre['tag']

# split dataset awal menjadi data train dan data sisa
#X_train, X_rem, y_train, y_rem = train_test_split(X,y, train_size=0.8)

# #pisahkan dataset menjadi data val dan data test
#X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.9)

df_tra = pd.DataFrame({'path':X_train
              ,'tag':y_train
             ,'set':'train'})

df_test = pd.DataFrame({'path':X_test
              ,'tag':y_test
             ,'set':'test'})

# df_valid = pd.DataFrame({'path':X_valid
#               ,'tag':y_valid
#              ,'set':'validation'})

"""
print('test size', len(df_test))
print('train size', len(df_tra))
print('val size', len(df_valid))
"""
print('test size', len(df_test))
print('train size', len(df_tra))

"""# Ubah data train menjadi csv"""

import cv2

train_image = []
train_label = []
for i in range(len(df_tra)):
    image = cv2.imread (str(df_tra['path'].values[i]), cv2.IMREAD_GRAYSCALE)
    # image = cv2.bitwise_not(image)
    image = cv2.resize(image, (28, 28),  interpolation = cv2.INTER_AREA)
    train_label.append(str(df_tra['tag'].values[i]))
    train_image.append (image)
    

X_data_train = np.array(train_image)
Y_label_train = np.array(train_label)

Y_label = []
for label in Y_label_train:
    if label == 'ba':
        label = 0
        Y_label.append(label)
    if label == 'ca': 
        label = 1
        Y_label.append(label)
    if label == 'da':
        label = 2
        Y_label.append(label)
    if label == 'dha': 
        label = 3
        Y_label.append(label)
    if label == 'ga':
        label = 4
        Y_label.append(label)
    if label == 'ha': 
        label = 5
        Y_label.append(label)
    if label == 'ja':
        label = 6
        Y_label.append(label)
    if label == 'ka': 
        label = 7
        Y_label.append(label)
    if label == 'la':
        label = 8
        Y_label.append(label)
    if label == 'ma': 
        label = 9
        Y_label.append(label)
    if label == 'na':
        label = 10
        Y_label.append(label)
    if label == 'nga': 
        label = 11
        Y_label.append(label)
    if label == 'nya':
        label = 12
        Y_label.append(label)
    if label == 'pa': 
        label = 13
        Y_label.append(label)
    if label == 'ra':
        label = 14
        Y_label.append(label)
    if label == 'sa': 
        label = 15
        Y_label.append(label)
    if label == 'ta':
        label = 16
        Y_label.append(label)
    if label == 'tha': 
        label = 17
        Y_label.append(label)
    if label == 'wa':
        label = 18
        Y_label.append(label)
    if label == 'ya': 
        label = 19
        Y_label.append(label) 
print ("Contoh X_train =", X_data_train)  
print(X_data_train.shape)
#Melihat datanya dulu gays
import matplotlib.pyplot as plt
plt.imshow(np.array(X_data_train[0]))
print("Contoh data images \n", Y_label[0])



"""# data test menjadi .csv"""

test_image = []
test_label = []
for i in range(len(df_test)):
    image = cv2.imread (str(df_test['path'].values[i]), cv2.IMREAD_GRAYSCALE)
    # image = cv2.bitwise_not(image)
    image = cv2.resize(image, (28, 28),  interpolation = cv2.INTER_AREA)
    test_label.append(str(df_test['tag'].values[i]))
    test_image.append (image)

X_data_test = np.array(test_image)
Y_label_test = np.array(test_label)
Y_test = []
for label in Y_label_test:
    if label == 'ba':
        label = 0
        Y_test.append(label)
    if label == 'ca': 
        label = 1
        Y_test.append(label)
    if label == 'da':
        label = 2
        Y_test.append(label)
    if label == 'dha': 
        label = 3
        Y_test.append(label)
    if label == 'ga':
        label = 4
        Y_test.append(label)
    if label == 'ha': 
        label = 5
        Y_test.append(label)
    if label == 'ja':
        label = 6
        Y_test.append(label)
    if label == 'ka': 
        label = 7
        Y_test.append(label)
    if label == 'la':
        label = 8
        Y_test.append(label)
    if label == 'ma': 
        label = 9
        Y_test.append(label)
    if label == 'na':
        label = 10
        Y_test.append(label)
    if label == 'nga': 
        label = 11
        Y_test.append(label)
    if label == 'nya':
        label = 12
        Y_test.append(label)
    if label == 'pa': 
        label = 13
        Y_test.append(label)
    if label == 'ra':
        label = 14
        Y_test.append(label)
    if label == 'sa': 
        label = 15
        Y_test.append(label)
    if label == 'ta':
        label = 16
        Y_test.append(label)
    if label == 'tha': 
        label = 17
        Y_test.append(label)
    if label == 'wa':
        label = 18
        Y_test.append(label)
    if label == 'ya': 
        label = 19
        Y_test.append(label)    
print(Y_test[0])

# print(X_data_test.shape)
# Y_label_test.shape

"""# Data Val menjadi .csv"""
'''
val_image = []
val_label = []
for i in range(len(df_valid)):
    image = cv2.imread (str(df_valid['path'].values[i]), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))
    val_label.append(str(df_valid['tag'].values[i]))
    val_image.append (image)

X_data_val = np.array(val_image)
Y_label_val = np.array(val_label)
# print(Y_label_val)
print("Data Ready To Use")'''
