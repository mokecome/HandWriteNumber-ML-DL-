def emptydir(dirname):  #清空資料夾
    if os.path.isdir(dirname):  #資料夾存在就刪除
        shutil.rmtree(dirname)
        sleep(2)  #需延遲,否則會出錯
    os.mkdir(dirname)  #建立資料夾

import cv2
from PIL import Image
import glob
import shutil, os
from time import sleep
import numpy as np

print('開始擷取車牌！')
dstdir = 'cropPlate'
emptydir(dstdir)
myfiles = glob.glob("predictPlate\*.JPG")
for imgname in myfiles:
    filename = (imgname.split('\\'))[-1]  #取得檔案名稱
    img = cv2.imread(imgname)  #讀入圖形
    detector = cv2.CascadeClassifier('haar_carplate.xml')
    signs = detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))  #框出車牌
    #擷取車牌
    if len(signs) > 0 :
        for (x, y, w, h) in signs:          
            image1 = Image.open(imgname)
            image2 = image1.crop((x, y, x+w, y+h))  #擷取車牌圖形
            image3 = image2.resize((140, 40), Image.ANTIALIAS) #轉換尺寸為140X40
            img_gray = np.array(image3.convert('L'))  #灰階
            _, img_thre = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY) #黑白
            cv2.imwrite(dstdir + '/'+ filename, img_thre) #存檔
    else:
        print(filename)

print('擷取車牌結束！')