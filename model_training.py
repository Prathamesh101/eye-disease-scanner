import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import pickle

Directory=r"E:\dataset"
categories=['Normal','Cataract','Glaucoma','Retina_disease']

IMG_SIZE=200
data=[]
for category in categories:
    folder=os.path.join(Directory,category)
    labels=categories.index(category)
    for img in os.listdir(folder):
        img_path=os.path.join(folder,img)
        img_arr=cv2.imread(img_path)
        new_array = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE)) 
        data.append([new_array,labels])



random.shuffle(data)
len(data)

for sample in data[:10]:
    print(sample[1])


x=[]
y=[]
IMG_SIZE=200
for features,labels in data:
    x.append(features)
    y.append(labels)
    


x= np.array(x)

y=np.array(y)



x[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_out = open("x.pickle","wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()


x=x/255

x.shape

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from tensorflow.keras.callbacks import TensorBoard

model=Sequential()

model.add(Conv2D(64,(3,3),input_shape=x.shape[1:],activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(64,activation='relu'))

model.add(Dense(4,activation="softmax"))


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


model.fit(x,y,epochs=5,validation_split=0.1,batch_size=32)

model.save('E:/dataset/model_final.h5')


