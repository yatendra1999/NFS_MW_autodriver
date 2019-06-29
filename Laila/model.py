from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os
import preprocessing
import extractlabel
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
dir='/home/ruskin/Downloads/IMAGE_DATASET/'
files=os.listdir(dir)
number_of_files=len(files)
X_data,Y_data=[],[]
cnt=0
for i in range(number_of_files):
    file_name=files[i]
    file_path=dir+file_name
    if file_name.endswith(".jpg"):
        img=cv2.imread(file_path,0)
        img=np.array(img)
        img=preprocessing.process(img)
        img=img.reshape(40,40,1)
        label=extractlabel.extract(file_name)
        if np.array_equal(label,np.array([1,0,0,0,0])):
                cnt+=1
        if cnt<3000:
                X_data.append(img)
                Y_data.append(extractlabel.extract(file_name))
X_data=np.array(X_data)
Y_data=np.array(Y_data)
xTrain, xTest, yTrain, yTest = train_test_split(X_data, Y_data, test_size = 0.2, random_state = 0)

print(xTrain.shape)
print(yTrain.shape)
print(xTest.shape)
print(yTest.shape)

model = Sequential()
model.add(Conv2D(16, (3, 3), padding='same',
                 input_shape=(40,40,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(5, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(xTrain,yTrain,epochs=30,batch_size=500,shuffle=True)
score, acc = model.evaluate(xTest, yTest)
print('Test score:', score)
print('Test accuracy:', acc)
model.save_weights('model_weights.h5')
with open('model_architecture.json', 'w') as f:
        f.write(model.to_json())