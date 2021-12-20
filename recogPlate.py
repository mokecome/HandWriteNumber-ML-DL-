def emptydir(dirname):  #清空資料夾
    if os.path.isdir(dirname):  #資料夾存在就刪除
        shutil.rmtree(dirname)
        sleep(2)  #需延遲,否則會出錯
    os.mkdir(dirname)  #建立資料夾    
   
def dirResize(src, dst):
    myfiles = glob.glob(src + '/*.JPG')  #讀取資料夾全部jpg檔案
    emptydir(dst)
    print(src + ' 資料夾：')
    print('開始轉換圖形尺寸！')
    for f in myfiles:
        fname = f.split("\\")[-1]
        img = Image.open(f)
        img_new = img.resize((300, 225), PIL.Image.ANTIALIAS)  #尺寸300x225
        img_new.save(dst + '/' + fname)
    print('轉換圖形尺寸完成！\n')    
    
def area(row, col):
    global nn
    if bg[row][col] != 255:
        return
    bg[row][col] = lifearea #記錄生命區的編號
    if col>1: #左方
        if bg[row][col-1]==255:
            nn +=1
            area(row,col-1)
    if col< w-1: #右方
        if bg[row][col+1]==255:
            nn +=1
            area(row,col+1)             
    if row>1: #上方
        if bg[row-1][col]==255:
            nn+=1            
            area(row-1,col)
    if row<h-1: #下方
        if bg[row+1][col]==255:
            nn+=1            
            area(row+1,col)       

import cv2
import PIL
from PIL import Image
import glob
import shutil, os
from time import sleep
import numpy as np
import sys
import re
import pytesseract

#dirResize('predictPlate_sr', 'predictPlate')
dirResize('p', 'predictPlate')
print('開始擷取車牌！')
print('無法擷取車牌的圖片：')
dstdir = 'cropPlate'
myfiles = glob.glob('predictPlate\*.JPG')
emptydir(dstdir)
for imgname in myfiles:
    filename = (imgname.split('\\'))[-1]  #取得檔案名稱
    img = cv2.imread(imgname)  #讀入圖形
    detector = cv2.CascadeClassifier('haar_carplate.xml')
    signs = detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))  #框出車牌
    #割取車牌
    if len(signs) > 0 :
        for (x, y, w, h) in signs:          
            image1 = Image.open(imgname)
            image2 = image1.crop((x, y, x+w, y+h))  #擷取車牌圖形
            image3 = image2.resize((140, 40), Image.ANTIALIAS)  #轉換尺寸為140X40
            img_gray = np.array(image3.convert('L'))  #灰階
            _, img_thre = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)  #黑白
            cv2.imwrite(dstdir + '/'+ filename, img_thre)
    else:
        print(filename)

print('擷取車牌結束！')

myfiles = glob.glob('cropPlate\*.jpg')
for file in myfiles:
    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #灰階
    _,thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV) #轉為黑白
    contours1 = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#尋找輪廓
    contours = contours1[0]   #取得輪廓

    letter_image_regions = [] #文字圖形串列
    for contour in contours:  #依序處理輪廓
        (x, y, w, h) = cv2.boundingRect(contour)  #單一輪廓資料
        letter_image_regions.append((x, y, w, h)) #輪廓資料加入串列
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])  #按X坐標排序
    # print(letter_image_regions)
    
    #先計算可以擷取的字元數
    count=0 #計算共擷取多少個字元
    for box in letter_image_regions:  #依序處理輪廓資料
        x, y, w, h = box        
        # x 必須介於 2~125 且寬度在 5~26、高度在 20~39 才是文字
        if x>=2 and x<=125 and w>=5 and w<=26 and h>=20 and h<40:
            count +=1   
            
    if count<6: #若字元數不足，可能是有兩個字元連在一起，將字元寬度放寬再重新擷取
        wmax=35
    else:
        wmax=26 #正常字元寬度
        
    nChar=0 #計算共擷取多少個字元
    letterlist = [] #儲存擷取的字元 
    for box in letter_image_regions:  #依序處理輪廓資料
        x, y, w, h = box        
        # x 必須介於 2~125 且寬度在 5~wmax、高度在 20~39 才是文字
        if x>=2 and x<=125 and w>=5 and w<=wmax and h>=20 and h<40:
            nChar +=1 
            letterlist.append((x, y, w, h)) #儲存擷取的字元
     
    # print("nChar=",nChar)
    # print("letterlist=",letterlist)        
    
    #去除雜點    
    for i in range(len(thresh)):  #i為高度
        for j in range(len(thresh[i])): #j為寬度  
            if thresh[i][j] == 255:     #顏色為白色
                count = 0 
                for k in range(-2, 3):
                    for l in range(-2, 3):
                        try:
                            if thresh[i + k][j + l] == 255: #若是白點就將count加1
                                count += 1
                        except IndexError:
                            pass
                if count <= 6:  #週圍少於等於6個白點
                    thresh[i][j] = 0  #將白點去除         
   
    #依序擷取字元, 去除第一字元和最後字元的崎鄰地後重組新的車牌    
    real_shape=[]
    for i,box in enumerate(letterlist):  #依序擷取的字元
        x, y, w, h = box        
        # print("box=",box)
        bg=thresh[y:y+h, x:x+w]
        
        # 去除崎鄰地 
        if i==0 or i==nChar: # 只去除第一字元和最後字元的崎鄰地
            lifearea=0 # 生命區塊
            nn=0       # 每個生命區塊的生命數
            life=[]    # 記錄每個生命區塊的生命數串列            
            for row in range(0,h):
                for col in range(0,w):
                  if bg[row][col] == 255:
                      nn = 1  #生命起源
                      lifearea = lifearea + 1  #有生命區塊數
                      area(row,col)  #以生命起源為起點探索每個生命區塊的總生命數
                      life.append(nn)

            maxlife=max(life) #找到最大的生命數
            indexmaxlife=life.index(maxlife) #找到最大的生命數的區塊編號       
               
            for row in range(0,h):
                for col in range(0,w):
                  if bg[row][col] == indexmaxlife+1:
                      bg[row][col]=255
                  else:
                      bg[row][col]=0        

        real_shape.append(bg) #加入字元               
        
        #在圖片週圍加白色空白OCR才能辨識
        image2=thresh.copy()
        newH, newW = image2.shape    
        space = 10  #空白寬度
        bg = np.zeros((newH+space*2, newW+space*2+20, 1), np.uint8)  #建立背景
        bg.fill(0)  #背景黑色
        
        # 將車牌文字加入黑色背景圖片中
        for i,letter in enumerate(real_shape):
            h=letter.shape[0] #原來文字圖形的高、寬
            w=letter.shape[1]
            x=letterlist[i][0] #原來文字圖形的位置
            y=letterlist[i][1]
            for row in range(h):  #將文字圖片加入背景
                for col in range(w):
                    bg[space+y+row][space+x+col+i*3] = letter[row][col] #擷取圖形
                    
        _,bg = cv2.threshold(bg, 127, 255, cv2.THRESH_BINARY_INV) #轉為白色背景、黑色文字                 
        cv2.imwrite('result.jpg', bg)  #存檔          
    
    #OCR辨識車牌
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    result = pytesseract.image_to_string(Image.open(r"result.jpg"))

    # 將 ocr 辨識結果優化
    txt=result.replace("!","1") # 如果是 ! 字元，更改為字元 1
    real_txt=re.findall(r'[A-Z]+|[\d]+',txt) #只取數字和大寫英文字母
    #組合真正的車牌
    txt_Plate="" 
    for char in real_txt:
        txt_Plate += char
    print("ocr 辨識結果：", result)
    basename=os.path.basename(file)
    if basename.split(".")[0]==txt_Plate:
        mess="V"
    else:
        mess="X"                         
    print("優化後：{}   檔名：{}  辨識結果:{}".format(txt_Plate,basename,mess))
    
    cv2.imshow('image', image)     #顯示原始圖形
    cv2.imshow('bg', bg)           #顯示組合的字元
    cv2.moveWindow("image",500,250)#將視窗移到指定位置
    cv2.moveWindow("bg",500,350)   #將視窗移到指定位置     
    key = cv2.waitKey(0)           #按任意鍵結束
    cv2.destroyAllWindows()
    if key == 113 or key==81:  #按q鍵結束
        break