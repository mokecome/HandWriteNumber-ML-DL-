# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 19:32:45 2021

@author: Bill
"""
'''
List.extend(iterable) 取出對象的所有元素 加入列表 [0, 1, 2, 3, 4, 0, 1]
List.append(object)           將對象  加入列表 [0, 1, 2, 3, 4, [0, 1]]




#內存不夠  只抽取一部分 每讀XX張處理一次  或   pca降維
        
'''


import cv2
import numpy as np#最裡面往最外面找
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense

import numpy as np
from keras.utils.np_utils import to_categorical

import PIL
from PIL import Image
import glob
import shutil, os
from time import sleep
import sys
from sklearn.model_selection import train_test_split 
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,accuracy_score
import joblib
from skimage import exposure
from skimage import feature
from keras.models import load_model
import imutils

def hog_Feature(img):
    #灰階
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    #作threshold(固定閾值)處理
    (T, thresh) = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

    #使用Canny方法偵側邊緣
    edged = imutils.auto_canny(thresh)

    #尋找輪廓，只取最大的那個
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)

    #取出輪廓的長寬高，用來裁切原圖檔。
    (x, y, w, h) = cv2.boundingRect(c)
    Cutted = gray[y:y + h, x:x + w]

    #將裁切後的圖檔尺寸更改為60×60。
    Cutted = cv2.resize(Cutted, (32, 32))
  
    #取得其HOG資訊及視覺化圖檔
    

    return  feature.hog(Cutted, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(2, 2), transform_sqrt=True, visualize=True)

def load_data(src):
    resultsX=[]
    resultsY=[]
    myfiles = glob.glob(src + '/*.JPG')  #讀取資料夾全部jpg檔案
    print(src + ' 資料夾：')
    print('開始轉換圖形尺寸！')
    for i, f in enumerate(myfiles):
        img = Image.open(f)
        '''
        img_resize = img.resize((32,32), PIL.Image.ANTIALIAS)  #尺寸32x32
        img_new_flatten=img_resize.flatten()  #  X.reshape(-1,32*32)
        resultsX.extend(img_new_flatten)
        '''
        (H, hogImage)=hog_Feature(img)
        print(i)

        resultsX.append(H)
        resultsY.append(src)
        
    resultsX=np.array(resultsX)    
    resultsY=np.array(resultsY)
    resultsY=to_categorical(resultsY,num_classes)
    
    return resultsX,resultsY    
        
num_classes=11#拒絕類11

dataSet=[]
labels=[]
for i in range(0,10):
    floder=str(i)
    x,y=load_data(floder) #第幾個類
    dataSet.extend(x)
    labels.extend(y)
    

dataSet=np.array(dataSet)
labels=np.array(labels)


X_train, X_test, y_train, y_test = train_test_split(
    dataSet,
    labels,
    test_size=0.2,
    shuffle=True,
    random_state=42,
)




clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))

'''
# Read the input image
#無法識別加入訓練集
#判斷低於多少就加入 輸入名字 訓練集  拒絕類或...
pred_imgs = np.array(np.array(imgs),'int16')

for i, img in enumerate(pred_imgs):
    fd = hog(img, orientations=9, pixels_per_cell=(12, 12), cells_per_block=(8, 8), visualize=False)
    nbr = clf.predict(np.array([fd], 'float64'))
    print("{} ---> {}".format(files[i], nbr[0]))
'''
testX=[]

print('開始測試')
mytest = glob.glob('test' + '/*.JPG')
for i, f in enumerate(mytest):
    img = Image.open(f)
    (H, hogImage)=hog_Feature(img)

#使用訓練的模型預測此圖檔
    pred = clf.predict(H.reshape(1, -1))[0]#reshape(-1,1)转换成1列  reshape(1,-1)转化成1行：

    #顯示HOG視覺化圖檔
    hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
    hogImage = hogImage.astype("uint8")
    #cv2.imshow("HOG Image #{}".format(i + 1), hogImage)
    #cv2.waitKey(0)
    #將預測數字顯示在圖片上
    print(pred.title().split("\\")[1])
    cv2.putText(img, pred.title().split("\\")[1], (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow("{}".format(i + 1), img)
  




    
 
 
 