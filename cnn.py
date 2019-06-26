import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import cv2 
import numpy as np

minimap = Sequential()
minimap.add(Conv2D(64, kernel_size= 3, activation='relu', input_shape = (190,230,3)))
minimap.add(Conv2D(4, kernel_size=3, activation='relu'))
minimap.add(Flatten())
minimap.add(Dense(2, activation = 'softmax'))
minimap.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='sparse_categorical_crossentropy',metrics=[tf.keras.metrics.categorical_accuracy])
minimap.load_weights('map.h5')

digits = Sequential()
digits.add(Conv2D(30, kernel_size = 3, activation='relu', input_shape = (30,28,1)))
digits.add(Conv2D(14, kernel_size= 3, activation='relu'))
digits.add(Conv2D(7, kernel_size= 3, activation="relu"))
digits.add(Flatten())
digits.add(Dense(10, activation = 'softmax'))
digits.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy',metrics=[tf.keras.metrics.categorical_accuracy])
digits.load_weights("digits.h5")

def get_speed(img):
    mm = img[380:-30,40:270,:-1]
    sp = img[495:525,804:888,:]
    sp = cv2.cvtColor(sp, cv2.COLOR_BGR2GRAY)
    sp1 = sp[:,0:28]
    sp2 = sp[:,28:56]
    sp3 = sp[:,56:]
    sp1 = np.reshape(sp1,[1,30,28,1])
    sp2 = np.reshape(sp2,[1,30,28,1])
    sp3 = np.reshape(sp3,[1,30,28,1])
    img = mm[40:-40,40:-40,2]
    img = np.resize(img,(22,30))
    mm = np.reshape(mm,[1,190,230,3])
    classes = minimap.predict_classes(mm)
    if classes == [1]:
        x1 = digits.predict_classes(sp1)
        x2 = digits.predict_classes(sp2)
        x3 = digits.predict_classes(sp3)
        if x1 == [5] and x2 == [5]:
            x1 = [0]
            x2 = [0]
        elif x1 == [5]:
            x1 = [0] 
        return img,x1[0]*100+x2[0]*10+x3
    else:
        return img,-1