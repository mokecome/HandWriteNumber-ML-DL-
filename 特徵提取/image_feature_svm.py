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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score
import joblib
from skimage import exposure
from skimage import feature
from keras.models import load_model
import imutils

def emptydir(dirname):  #清空資料夾
    if os.path.isdir(dirname):  #資料夾存在就刪除
        shutil.rmtree(dirname)
        sleep(2)  #需延遲,否則會出錯
    os.mkdir(dirname)  #建立資料夾

def Harr(imgs):  #如果是數組運算直接   其他展開
    dstdir = 'cropPlate'
    emptydir(dstdir)
    x=[]
    for img in imgs:
        detector = cv2.CascadeClassifier('haar_carplate.xml')
        signs = detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))  #框出車牌
        #擷取車牌
        if len(signs) > 0 :
            for (x, y, w, h) in signs:          
                image1 = Image.open(img)
                image2 = image1.crop((x, y, x+w, y+h))  #擷取車牌圖形
                image3 = image2.resize((140, 40), Image.ANTIALIAS) #轉換尺寸為140X40
                img_gray = np.array(image3.convert('L'))  #灰階
                _, img_thre = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY) #黑白
                x.append(img_thre)
    return x

def hog_Feature(imgs):
    x=[]
    hI=[]
    for img in imgs:
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

        #將裁切後的圖檔尺寸更改為32×32。
        Cutted = cv2.resize(Cutted, (32, 32))
  
        #取得其HOG資訊及視覺化圖檔
        (H, hogImage)=feature.hog(Cutted, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(2, 2), transform_sqrt=True, visualize=True)
        x.append(H)
        hI.append(hogImage)
    return x,hI

def LBP_describe(image):
    #上面指令是從圖片的HSV色彩模型中，取得其平均值及標準差（有RGB三個channels，因此會各有3組平均值及標準差）作為特徵值
    (means, stds) = cv2.meanStdDev(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    #進行降維處理：將means及stds各3組array使用concatenate指令合成1組，再予以扁平化（變成一維）。
    colorStats = np.concatenate([means, stds]).flatten()
    
    #將圖片轉為灰階
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #P=30
    numPoints = 30
    #r=3
    radius = 3
    #eps指The "close-enough" factor，為一極小值，用以判斷兩數是否相當接近，在此是避免相除時分母為零發生錯誤
    eps = 1e-7
    lbp = feature.local_binary_pattern(gray, numPoints, radius, method="uniform")
    #Numpy的ravel()類似flattern
    (hist, _) = np.histogram(lbp.ravel(), bins=range(0, numPoints + 3), range=(0, numPoints + 2))

    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)

    return np.hstack([colorStats, hist])



def load_data(src):
    resultsX=[]
    resultsY=[]
    myfiles = glob.glob(src + '/*.JPG')  #讀取資料夾全部jpg檔案
    print(src + ' 資料夾：')
    print('開始轉換圖形尺寸！')
    for i, f in enumerate(myfiles):
        img = Image.open(f)
        img_resize = img.resize((32,32), PIL.Image.ANTIALIAS)  #尺寸32x32
        '''
        #深度學習
        img_new_flatten=img_resize.flatten()  #  X.reshape(-1,32*32)  
        '''
        #img_resize=LBP_describe(img_resize)
        print(i)

        resultsX.append(img_resize)
        resultsY.append(src)
        
    resultsX=np.array(resultsX)
    resultsY=np.array(resultsY)
    resultsY=to_categorical(resultsY,num_classes)
    
    return resultsX,resultsY     #列表  np.array
        
num_classes=11#拒絕類11

dataSet=[]
labels=[]
for i in range(0,10):
    floder=str(i)
    x,y=load_data('../'+floder) #第幾個類
    f,hogImage=hog_Feature(x)#f=[hog_Feature(a) for a in x]
    dataSet.extend(f)
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


'''
rf = RandomForestClassifier(n_estimators=args['forest'], random_state=42)
rf.fit(X_train, y_train)
predictions = rf.predict(testData)
print(classification_report(testLabels, predictions))
'''
clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))

'''
# Read the input image
#無法識別加入訓練集
#判斷低於多少就加入 輸入名字 訓練集  拒絕類或...
'''
testX=[]

print('開始測試')
mytest = glob.glob('test' + '/*.JPG')
for i, f in enumerate(mytest):
    img = Image.open(f)
    H, hogImage=hog_Feature(img)

#使用訓練的模型預測此圖檔
    pred = clf.predict(H[0].reshape(1, -1))[0]#reshape(-1,1)转换成1列  reshape(1,-1)转化成1行：

    #顯示HOG視覺化圖檔
    hogImage = exposure.rescale_intensity(hogImage[0], out_range=(0, 255))
    hogImage = hogImage.astype("uint8")
    #cv2.imshow("HOG Image #{}".format(i + 1), hogImage)
    #cv2.waitKey(0)
    #將預測數字顯示在圖片上
    print(pred.title().split("\\")[1])
    cv2.putText(img, pred.title().split("\\")[1], (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow("{}".format(i + 1), img)
  




    
 
 
 