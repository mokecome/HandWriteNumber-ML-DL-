from keras.models import load_model
import numpy as np
from keras.utils import np_utils
import PIL
from PIL import Image
import glob
import shutil, os
from time import sleep
import sys
import cv2

def load_img(src):
    resultsX=[]
    resultsY=[]
    myfiles = glob.glob(src + '/*.JPG')  #讀取資料夾全部jpg檔案
    print(src + ' 資料夾：')
    print('開始轉換圖形尺寸！')
    for i, f in enumerate(myfiles):
        img = Image.open(f)
        img_resize = img.resize((32,32), PIL.Image.ANTIALIAS)  #尺寸300x225
        
        
        img_resize=np.array(img_resize)/256 #DNN 深度學習數值不宜過大(需要小的w 但w初始化一般都基於0-1) X=1.0*X/256
        

        print(i)
        img_resize=img_resize.flatten()#img_gray=img_gray.reshape(-1,32*32)#784
        resultsX.extend(img_resize)
        
        
    resultsX=np.array(resultsX)    
    
    return resultsX

num_classes=11
Y=15

resultsX=load_img('test')
model = load_model('model')  
results=model.predict(resultsX)
for result,y in zip(results,Y):
	print(result)
	print(y)



