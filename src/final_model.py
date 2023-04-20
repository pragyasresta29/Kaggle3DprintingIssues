
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Rescaling, RandomFlip, RandomRotation, BatchNormalization, Conv1D, MaxPool1D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,RMSprop
from keras import layers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from keras.regularizers import l2, l1, l1_l2

import tensorflow as tf

import cv2
import os, gc

import numpy as np
import pandas as pd
import warnings
from keras import regularizers

warnings.filterwarnings("ignore")
basePath = "/Users/pragya/PycharmProjects/NLP/Kaggle3DprintingIssues/"
dataPath = "/Users/pragya/PycharmProjects/NLP/Kaggle3DprintingIssues/src/data/"
trainData = 'train_ds_80.npy'
testData = 'test_ds_80.npy'
IMG_SIZE = 80

# Fetching preprocessed train data
train = np.load(dataPath + trainData, allow_pickle=True)

# Sep images and labels.
X = []
y = []

for feature, label in train:
    X.append(feature)
    y.append(label)


X = np.array(X)
y = np.array(y)


# Splitting train data for test and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# clearing cache to free memory
del train
del X
del y
gc.collect()


def conv1D():
    # Training model
    reg = 0.01
    drop=0.2
    model = Sequential()
    model.add(Conv1D(32, 3, padding="same", activation="relu", input_shape=(X_train.shape[1:]), kernel_regularizer=l2(reg)))
    model.add(MaxPool1D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Dropout(drop))
    model.add(Conv1D(64, 3, padding="same", activation="relu", kernel_regularizer=l2(reg)))
    model.add(MaxPool1D())
    model.add(BatchNormalization())

    model.add(Dropout(drop))

    model.add(Conv1D(128, 3, padding="same", activation="relu", kernel_regularizer=l2(reg)))
    model.add(MaxPool1D())
    model.add(BatchNormalization())
    model.add(Dropout(drop))

    model.add(Flatten())
    model.add(Dense(512, activation="relu", kernel_regularizer=l2(reg)))
    model.add(Dropout(drop))

    model.add(Dense(2, activation="sigmoid", kernel_regularizer=l2(reg)))
    model.summary()

    model.compile(optimizer=Adam(lr=0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

    # Evaluating model on unseen test data.
    predictions = model.predict(X_test)
    predictions = np.argmax(predictions, axis=-1)
    print(classification_report(y_test, predictions,
                                target_names=['not_under_extrusion (Class 0)', 'Under Extrusion (Class 1)']))
    return model
model = conv1D()


def conv2d():
    # Training model
    dropout = 0.2
    model = Sequential()
    model.add(
        Conv2D(8, 9, padding="same", activation="relu", input_shape=(X_train.shape[1:]), kernel_regularizer=l2(0.1)))
    model.add(MaxPool2D(pool_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Conv2D(16, 3, padding="same", activation="relu", kernel_regularizer=l2(0.1)))
    model.add(MaxPool2D())
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=l2(0.1)))
    model.add(MaxPool2D())
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(128, activation="relu", kernel_regularizer=l2(0.1)))
    model.add(Dense(2, activation="softmax", kernel_regularizer=l2(0.1)))
    model.summary()

    model.compile(optimizer=Adam(lr=0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=15, batch_size=64, validation_split=0.33, steps_per_epoch=50)
    # Evaluating model on unseen test data.
    predictions = model.predict(X_test)
    predictions = np.argmax(predictions, axis=-1)
    print(classification_report(y_test, predictions,
                                target_names=['not_under_extrusion (Class 0)', 'Under Extrusion (Class 1)']))
    return model

model = conv2d()
checkpoint = tf.train.Checkpoint(model)
checkpoint.write('fm_10')
model.save('fm_10.hdf5')




# Fetching preprocessed train data
test = np.load(dataPath + testData, allow_pickle=True)

# Sep images and labels for test.
X_test = []
y_test = []

for feature, label in test:
    X_test.append(feature)
    y_test.append(label)


X_test = np.array(X_test)
y_test = np.array(y_test)

np.save(dataPath + 'test_160.npy', X_test)
test = np.load(dataPath + 'test_80.npy', allow_pickle=True)

predictions = model.predict(test)
predictions = np.argmax(predictions, axis=-1)

test_df = pd.read_csv(basePath + "data/test.csv")
pred_df = test_df.drop(columns=['print_id', 'printer_id'])
pred_df['has_under_extrusion'] = predictions
# Saving prediction
pred_df.to_csv(dataPath + "/submissions/" + "fm_17_sub.csv", index=False)

