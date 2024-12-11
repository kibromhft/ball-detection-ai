import os
import cv2
from matplotlib import pyplot as plt
from statistics import mean


path='data/obj/'

imglist=os.listdir('pics')

textfile=open('pics.txt','w')
i=0
img_brightness = []
for im in imglist:
    img = cv2.imread('pics/'+im)
    imc = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    imgHSVRGB = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(imgHSVRGB, cv2.COLOR_BGR2GRAY)
    
    # calculate the brightness of the image
    blur = cv2.blur(gray, (5,5))
    brightness = cv2.mean(blur)
    img_brightness.append(brightness[0])
    textfile.write(im+' = '+str(brightness)+'\n')


print(mean(img_brightness))
print(min(img_brightness))
print(max(img_brightness))

av = sum(img_brightness)/len(img_brightness)
print("avg=" +str(av))


    
    
