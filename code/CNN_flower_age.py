#!/usr/bin/env python3

""" Reads in images an performs preprocessing making tem ready for CNN analysis"""

__appname__ = 'RF_CNN.py'
__author__ = '[Elliott Parnell (ejp122@ic.ac.uk)]'
__version__ = '0.0.1'


#from cgi import test
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
import glob 
import os
import pickle
import seaborn as sns
import pyreadr
from sklearn.model_selection import train_test_split

### KERAS IMports ###
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.layers.normalization import batch_normalization
from keras import backend as K

### Sklearn label encoding ###
from sklearn import preprocessing

### Set seed for random selection ###

### Img Input Size ###
SIZE = 100

### Load in dataframe ###
DF = pd.read_pickle("MProjDF.pk1")

DF["flower_start"] = ""
DF["flower_age"]=""


### Lets make a comparitive time stamp for each flower ###
unique_ids = DF["unique_id"].unique()
for iter in range(len(unique_ids)):
    lowest_val = min(((DF[DF["unique_id"] == unique_ids[iter]]))["exp_clock"])
    DF.loc[DF.unique_id == unique_ids[iter], "flower_start"] = lowest_val 


DF["flower_age"] = DF["exp_clock"] - DF["flower_start"]

DF = DF[DF["flower_age"].isin([0,2,4,6,8,24,26,28,30,32])]
DF = DF[DF["plant_id"]>49]

R_import = pyreadr.read_r("~/Documents/MastersProject/code/clean_data.Rda")
R_DF = R_import["DF2"]

DF= DF[DF["unique_id"].isin(R_DF["unique_id"])]

#Images and labels list 
images = []
labels = []

### Load in the labels and NP arrays into a list ###
for iter in range(len(DF)):
    load_str =  "sq_non_norm/" + str(DF.index[iter])[2:-3] + "_sq.npy"
    age = DF.flower_age[iter]
    img = np.load(load_str)
    images.append(img)
    labels.append(age)

images = np.asarray(images)    
labels = np.asarray(labels)


### Training data = 1 , test data = 0 ###
# print(len(DF))
# train_test = [1,0] * int((len(DF)-1)/2)
# train_test.append(1)
# DF["training_data"] = train_test

x_train, x_test, y_train, y_test = train_test_split(images, labels, random_state=123, test_size=0.35, shuffle=True)


### One hot encoding the Y values ###
# from keras.utils import to_categorical
# y_train_one_hot = to_categorical(y_train)
# y_test_one_hot = to_categorical(y_test)


########################
### MAKING OUR MODEL ###
########################

### Rsquared metric ###
def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true - y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


# Make the model 
activation = 'relu'

feature_extractor = Sequential()
feature_extractor.add(Conv2D(32, (3,3), activation = activation, padding = 'same', input_shape = (SIZE, SIZE, 3)))
# feature_extractor.add(batch_normalization.BatchNormalization())
feature_extractor.add(MaxPooling2D(pool_size=(2,2)))

feature_extractor.add(Conv2D(32, (3,3), activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
# feature_extractor.add(batch_normalization.BatchNormalization())
feature_extractor.add(MaxPooling2D(pool_size=(2,2)))

# feature_extractor.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
# feature_extractor.add(batch_normalization.BatchNormalization())

feature_extractor.add(Conv2D(64, (3,3), activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
# feature_extractor.add(batch_normalization.BatchNormalization())
feature_extractor.add(MaxPooling2D(pool_size=(2,2)))

feature_extractor.add(Flatten())


feature_extractor.add(Dense(64, activation=activation))
feature_extractor.add(Dropout(0.5))

feature_extractor.add(Dense(1, activation = 'linear'))








feature_extractor.compile(optimizer='adam',loss = 'mean_squared_error', metrics = [r2_keras])
print(feature_extractor.summary()) 

print("Now lets train the model")

##########################################
#Train the CNN model
# history = cnn_model.fit(x_train, y_train_one_hot, epochs=50, validation_data = (x_test, y_test_one_hot))
history = feature_extractor.fit(x_train, y_train, batch_size = 50, epochs=40, validation_data = (x_test, y_test))

prediction_age = feature_extractor.predict(x_test)

saveDF = {"Actual _flower_age": y_test, "Predicted_flower_age": prediction_age}
saveDF = pd.DataFrame(data=saveDF)

saveDF.to_pickle("~/Documents/MastersProject/code/cnn_age_results.pk1")
#plt.plot(prediction_age, y_test, "ro")

##########################################
###### Visualisation of the training #####
##########################################
#plot the training and validation accuracy and loss at each epoch
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'y', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()


# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# plt.plot(epochs, acc, 'y', label='Training acc')
# plt.plot(epochs, val_acc, 'r', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()






