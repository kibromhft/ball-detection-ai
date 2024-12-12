import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import datetime
import math
import time
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

img = cv2.imread('D:/data/recording4.h264')
vs = cv2.VideoCapture(args["video"])

time.sleep(2.0)
speed = 0

curtime = 0
while True:
    # read the frame
    img = vs.read()
    img = img[1]
    img = imutils.resize(img, width=1000)
    imc = img.copy()

    # convert the image from BGR to RGB and then to HSV color format
    img_copy = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    imgHSVRGB = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
    imgHSVBGR = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(imgHSVRGB, cv2.COLOR_BGR2GRAY)

    # calculate the brightness of the image using mean brightness
    threshold1 = 50
    threshold2 = 100
    threshold3 = 115
    blur = cv2.blur(gray, (5, 5))
    img_mean = cv2.mean(blur)[0]

    dark = 0
    if img_mean <= threshold1:
        dark = 1
    elif (img_mean > threshold1) & (img_mean <= threshold2):\
        dark = 2
    elif (img_mean > threshold2) & (img_mean <= threshold3):
        dark = 3

    img_copy = np.copy(img)
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    filterdImg = img_copy.copy()

    (xCord, yCord) = (0, 0)
    (xDis, yDis) = (0, 0)

    # pass each image according to thier color space.
    if dark == 0:
        lower_hsv = np.array([120, 70, 55])
        upper_hsv = np.array([255, 255, 255])
    elif dark == 1:
        lower_hsv = np.array([110, 40, 15])
        upper_hsv = np.array([255, 255, 255])
    elif dark == 2:
        lower_hsv = np.array([110, 40, 45])
        upper_hsv = np.array([255, 255, 255])
    else:
        lower_hsv = np.array([115, 70, 100])
        upper_hsv = np.array([255, 255, 255])

    # Smooth and remove noise from the image
    mask = cv2.inRange(imgHSVRGB, lower_hsv, upper_hsv)
    img_copy = imgHSVRGB.copy()

    kernel = np.ones((5, 5), np.uint8)
    kernel = np.ones((5, 5), np.uint8)

    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernel)

    maskOpen2 = cv2.morphologyEx(maskClose, cv2.MORPH_OPEN, kernel)
    maskClose2 = cv2.morphologyEx(maskOpen2, cv2.MORPH_CLOSE, kernel)

    erosion = cv2.erode(maskClose2, kernel, iterations=2)
    erosionMain = cv2.dilate(erosion, kernel, iterations=2)

    processImg = filterdImg.copy();
    processImg[erosionMain == 0] = 0
    processImg = cv2.GaussianBlur(processImg, (3, 3), 2)
    processImg = cv2.add(processImg, processImg)
    process_copy = processImg

    # reFilter the image with a wider threshold - removes unwanted region 
    lower_hsv = np.array([10, 0, 0])
    upper_hsv = np.array([255, 255, 255])

    maskX = cv2.inRange(processImg.copy(), lower_hsv, upper_hsv)
    img_copy = imgHSVRGB.copy()

    maskOpenX = cv2.morphologyEx(maskX, cv2.MORPH_OPEN, kernel)
    maskCloseX = cv2.morphologyEx(maskOpenX, cv2.MORPH_CLOSE, kernel)

    maskOpenEx = cv2.morphologyEx(maskCloseX, cv2.MORPH_OPEN, kernel)
    maskCloseEx = cv2.morphologyEx(maskOpenEx, cv2.MORPH_CLOSE, kernel)

    erosionEx = cv2.erode(maskCloseEx, kernel, iterations=2)
    erosionMainEx = cv2.dilate(erosionEx, kernel, iterations=2)

    processImg = filterdImg.copy();
    processImg[erosionMainEx == 0] = 0
    processImg = cv2.GaussianBlur(processImg, (3, 3), 2)
    process_copy = processImg

    # detect object edges
    kernelEdge = np.ones((11, 11), np.uint8)
    edge_detected_image = cv2.Canny(processImg, 0, 1)

    erosion = cv2.dilate(edge_detected_image, kernelEdge, iterations=3)
    erosion = cv2.erode(erosion, kernelEdge, iterations=1)

    erosion = cv2.GaussianBlur(erosion, (3, 3), 2)

    processImg[erosion == 0] = 0

    processImg1 = cv2.cvtColor(processImg, cv2.COLOR_RGB2GRAY)
    contours, hierarchy = cv2.findContours(processImg1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_list = []
    canny_center = []
    radius_list = []
    area = 0
    k = -1
    ap = 0
    rad = 0

    # find the contours
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.1 * cv2.arcLength(contour, True), True)

        area = cv2.contourArea(contour)
        ((x, y), radius) = cv2.minEnclosingCircle(contour)

        # We assume there won't be an object with bigger circle than the ball in the field
        if (len(approx) > ap) & (radius > rad):
            ap = len(approx)
            rad = radius

            M = cv2.moments(contour)

            if M["m00"] > 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]), radius)
                canny_center.append(center)

    # Canny will not detect all objects, we need other methods to appyl, here we use Hough Transform
    circles = cv2.HoughCircles(erosion, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=20, minRadius=0, maxRadius=0)

    circles_rounded = []
    storeRadius = []
    if circles is not None:
        circles_rounded = np.uint16(np.around((circles)))
        if len(circles_rounded) > 0:
            storeRadius = circles_rounded[0]
            for cir in range(len(circles_rounded)):
                circle = circles_rounded[cir]
                if storeRadius[0][2] < circle[0][2]:
                    storeRadius = circle
                    
    # compare circles found by hough and canny
    save_hough = []
    save_canny = []
    index = -1
    temp = 0
    if len(canny_center) > 0 & len(storeRadius) >0:
        for cann in canny_center:
            for c in cann:
                for sr in storeRadius:
                    tempDisx = abs(sr[0] - cann[0])
                    tempDisy = abs(sr[1] - cann[1])
                    tempDisr = abs(sr[2] - cann[2])

                    if tempDisx <= 5:
                        if tempDisy <= 5:
                            if tempDisr <= 5:
                                check = any((sr == x).all() for x in save_hough)
                                if check == False:
                                    save_hough.append(sr)
                                    save_canny.append(cann)

    # Find the optimal value to label the center point
    obj_canny = len(save_canny)
    obj_hough = len(save_hough)
    contlen = len(contours)

    if k == -1 | k >= obj_canny:
        k = 0

    if len(save_canny) >= 1 & len(save_hough) >= 1:
        canny_area = 2 * np.pi * save_canny[0][2]
        hough_area = 2 * np.pi * save_hough[0][2]

        if (contlen > 1) & (obj_canny == 1):
            ball = cv2.circle(imc.copy(), (save_canny[0][0], save_canny[0][1]), 30, (10, 200, 200), 3)
            center = save_canny[0]
        elif (obj_hough == 1 & obj_canny > 1):
            ball = cv2.circle(imc.copy(), (save_hough[0][0], save_hough[1][1]), 30, (10, 200, 200), 3)
            center = save_hough[0]
        elif (obj_canny == 1 & obj_hough > 1):
            ball = cv2.circle(imc.copy(), (save_canny[0][0], save_canny[0][1]), 30, (10, 200, 200), 3)
        elif obj_hough == 1 & (hough_area > canny_area):
            ball = cv2.circle(imc.copy(), (save_hough[0][0], save_hough[0][1]), 30, (10, 200, 200), 3)
            center = save_hough[0]
        elif obj_canny == 1 & (canny_area > hough_area):
            ball = cv2.circle(imc.copy(), (save_canny[0][0], save_canny[0][1]), 30, (10, 200, 200), 3)
            center = save_canny[0]
        elif obj_hough > obj_canny:
            ball = cv2.circle(imc.copy(), (save_hough[0][1], save_hough[1][1]), 30, (10, 200, 200), 3)
            center = save_hough[0]
        else:
            ball = cv2.circle(imc.copy(), (save_canny[0][0], save_canny[0][1]), 30, (10, 200, 200), 3)
            center = save_canny[0]
    elif len(canny_center) > 0:
        ball = cv2.circle(imc.copy(), (canny_center[0][0], canny_center[0][1]), 30, (10, 200, 200), 3)
        center = canny_center[0]

    yVal = abs(int(center[1]) - int(yCord))
    xVal = abs(int(xCord - center[0]))

    # Angle
    if xVal > 0:
        angle = np.abs(yVal) / xVal
        angle = math.atan(angle)
        angle = int(np.round(math.degrees(angle)))
        txt = "Angle=" + str(np.round(angle, 4))
    else:
        txt=""
        
    xCord = center[0]
    yCord = center[1]
    
    if k == -1:
        k=0

    # Speed
    spd = 0
    xySpd = 0
    if xDis == 0:
        xySpd = abs(yDis - center[1])
    elif yDis == 0:
        xySpd = abs(xDis - center[0])
    else:
        ySpd = yVal*yVal
        xSpd = xVal*xVal
        xySpd = np.sqrt(ySpd + xSpd)

    savetime = abs(curtime - time.time())
    curtime = time.time() # save this time
    
    speed = np.round(np.round(xySpd)/ savetime)

    if speed >= 1000 & (speed < 1000000):
        speed = str(np.round(speed/1000,2))+"k"
    elif speed >= 1000000:
        speed = str(np.round(speed/1000000,2)) +"M"
        
    txt = txt + ", Speed="+ str(speed) + " px/sec"

    # Update the fram
    cv2.putText(ball, txt, (center[0] - 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("mat", ball)
    
    key = cv2.waitKey(1) & 0xFF
    
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

