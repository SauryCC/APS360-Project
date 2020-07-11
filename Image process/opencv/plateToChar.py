import cv2
import numpy as np
print("cv2 imported")

# ------------------detect shape----------------------------------------


def cropout(img):
    # (image, retrive mwthod, approximation)
    # cv2.RETR_EXTERNAL finds the outer contour
    # cv2.CHAIN_APPROX_NONE gets all the detailes without compress
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # (img, contour to draw, contour index, color, thickness)
        # -1 means all contours
        #cv2.drawContours(imgcpy, cnt, -1, (255, 0, 0),3)

        # to cutoff useless contours, test etc
        # try it out to find area limit
        if area > 100:
            #print(area)
            #cv2.drawContours(imgcpy, cnt, -1, (255, 0, 0), 1)
            # find arc length, (target, if  target is enclosed)
            para = cv2.arcLength(cnt, True)
            # get vertex approximation for each contour
            # (contour, epsilon*length, if it is closed)
            approx = cv2.approxPolyDP(cnt, 0.05*para, True)
            # to predict shape:
            objContour = len(approx)
            # draw a box around the shape
            x, y, w, h = cv2.boundingRect(approx)
            # draw on imgcpy, (target to draw on, start, end, color, width)
            #cv2.rectangle(imgcpy, (x, y), (x+w, y+h), (0, 255, 0),2)

            # crop the rectangle out

            # [height,width]
            imggCropped = imgcpy[y:y+h, x:x+w]


            return imggCropped



def findColor(img, myColors, penColors):
    # covert to HSV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    colorCount = 0
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        mask1 = cv2.bitwise_not(mask)
        # show all masks for each color
        cv2.imshow(str(color[0]), mask1)


        # draw a line with center top of color contour
        centerx, centery = getContour(mask1)
        #cv2.circle(imgContour, (centerx, centery), 10, penColors[colorCount], cv2.FILLED)
    return mask1


# -------------------------------  find contour for the color tracked --------------------------
def getContour(img):
    # (image, retrive mwthod, approximation)
    # cv2.RETR_EXTERNAL finds the outer contour
    # cv2.CHAIN_APPROX_NONE gets all the detailes without compress
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # define variables
    x, y, w, h = 0, 0, 0, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # (img, contour to draw, contour index, color, thickness)
        # -1 means all contours
        #cv2.drawContours(enhanced, cnt, -1, (255, 0, 0),15)

        # to cutoff useless contours, test etc
        # try it out to find area limit
        print(area)
        if 1000 < area < 7000:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 10)
            # find arc length, (target, if  target is enclosed)
            para = cv2.arcLength(cnt, True)
            # get vertex approximation for each contour
            # (contour, epsilon*length, if it is closed)
            approx = cv2.approxPolyDP(cnt, 0.02*para, True)
            # to predict shape:
            objContour = len(approx)
            # draw a box around the shape
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(enhanced, (x, y), (x+w, y+h), (0, 255, 0), 3)
            myPoints.append([x,y,w,h])



    # to draw a line with the center top of the contour
    return x+w//2, y


def getIndividualPic(imgSrc, points):
    print(len(points))
    picList = []
    # covert to HSV
    imgHSV = cv2.cvtColor(imgSrc, cv2.COLOR_BGR2HSV)
    colorCount = 0
    for point in points:
        x, y, w, h = point
        for color in myColors:
            lower = np.array(color[0:3])
            upper = np.array(color[3:6])
            mask = cv2.inRange(imgHSV, lower, upper)
            mask1 = cv2.bitwise_not(mask)
            imggCropped = mask1[y:y + h, x:x + w]
            picList.append(imggCropped)
    print(len(picList))
    return picList


# ------------------------------------------------------------------------------------------------------------
img = cv2.imread("imgs/1.png")
# a list of mask values
# [[HUE Min, SAT Min, VALUE Min, HUE Max, SAT Max, VALUE Max],...]
myColors = [[0,0,137,179,255,255]]
# a list of drawing colors corresponding to myColors in BGR
penColors = [[255, 0, 0]]
# a list of points with color track
myPoints = []
# draw contours onto image

# convert to 1 channel
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# use blur if needed
imgBlur = cv2.GaussianBlur(img, (3,3), 1)
imgCanny = cv2.Canny(imgBlur, 50, 50)

# copy original image to draw onto it
imgcpy = img.copy()
# crop img out
imggCropped = cropout(imgCanny)
enhanced = cv2.resize(imggCropped,(600,200))



imgContour = enhanced.copy()

findColor(imgContour, myColors, penColors)
print(myPoints)
#cv2.imwrite( "imgs/test.png", enhanced )


imgSrc = enhanced.copy()
outputs = getIndividualPic(imgSrc, myPoints)
for i in range(len(outputs)):
    outputs[i] = cv2.resize(outputs[i] ,(280,280))


cv2.imshow("1", outputs[0])
cv2.imshow("2", outputs[1])
cv2.imshow("3", outputs[2])
cv2.imshow("4", outputs[3])
cv2.imshow("5", outputs[4])
cv2.imshow("6", outputs[5])
cv2.imshow("7", outputs[6])
cv2.imshow("8", outputs[7])


cv2.imshow('original image', img)
#cv2.imshow('imggCropped', imggCropped)
cv2.imshow('enhanced', enhanced)
#cv2.imshow("origin", imgContour)
#cv2.imshow("img Contour", imgcpy)
# this only displays for 1ms, to keep showing:
# 0 for infinite delay, else ms
cv2.waitKey(0)