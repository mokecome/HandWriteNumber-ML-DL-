# -*- encoding:utf-8 -*-
import numpy
from tensorflow.keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten

from keras import backend as K
from keras.utils.np_utils import to_categorical
import numpy as np
import random

import PIL
from PIL import Image
import glob
import shutil, os
from time import sleep
import sys
import cv2
#前處理

#讀圖片
def load_img(src):
    resultsX=[]
    resultsY=[]
    myfiles = glob.glob(src + '/*.JPG')  #讀取資料夾全部jpg檔案
    print(src + ' 資料夾：')
    print('開始轉換圖形尺寸！')
    for i, f in enumerate(myfiles):
        img = Image.open(f)
        img_resize = img.resize((32,32), PIL.Image.ANTIALIAS)  #尺寸300x225
        
        
        img_resize=np.array(img_resize)/256 
        

        print(i)
        img_resize=img_resize.flatten()#img_gray=img_gray.reshape(-1,32*32)#784
        resultsX.append(img_resize)
        resultsY.append(src)
        
    resultsX=np.array(resultsX)    
    resultsY=np.array(resultsY)
    resultsY=to_categorical(resultsY,num_classes)
    
    
    return resultsX,resultsY    
        
num_classes=11 #拒絕類 11

dataSet=[]
labels=[]
for i in range(0,10):
    floder=str(i)
    x,y=load_img(floder) #第幾個類
    dataSet.extend(x)
    labels.extend(y)
    

dataSet=np.array(dataSet)
labels=np.array(labels)


print(dataSet.shape)
print(labels.shape)

model = Sequential()
#input_shape=(输入长，输入宽，输入通道数)
model.add(Dense(units=256,input_dim=32*32, activation='relu',use_bias=True))
model.add(Dense(units=128,input_dim=256,activation='relu',use_bias=True))
model.add(Dense(units=128,input_dim=256,activation='relu',use_bias=True))
model.add(Dense(units=num_classes,input_dim=128,use_bias=True, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

model.fit(dataSet,labels, batch_size=50, epochs=10,shuffle=True)#shuffle=True避免以數據規律影響模型
model.save("model")



