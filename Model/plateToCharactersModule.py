import cv2
import numpy as np
print("cv2 imported")

# ------------------detect shape----------------------------------------


def cropout(img, imgcpy):
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



def findColor(img, myColors, penColors, imgContour, enhanced, myPoints):
    # covert to HSV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    colorCount = 0
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        mask1 = cv2.bitwise_not(mask)
        # show all masks for each color
        #cv2.imshow(str(color[0]), mask1)
        # draw a line with center top of color contour
        centerx, centery = getContour(mask1, imgContour, enhanced, myPoints)
        #cv2.imshow('Cropped out on stretched picture', enhanced)
    return mask1


# -------------------------------  find contour for the color tracked --------------------------
def getContour(img, imgContour, enhanced, myPoints):
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
        #print(area)
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
    return x+w//2, y


def getIndividualPic(imgSrc, points, myColors):
    #print(len(points))
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
            # invert color
            picList.append(cv2.bitwise_not(imggCropped))
    #sprint(len(picList))
    return picList


# --------------------------------------- core control function-------------------------------------------------------------
def slicePic(img):
    # a list of mask values
    # [[HUE Min, SAT Min, VALUE Min, HUE Max, SAT Max, VALUE Max],...]
    myColors = [[0,0,137,179,255,255]]
    # a list of drawing colors corresponding to myColors in BGR
    penColors = [[255, 0, 0]]
    # a list of points with color track
    myPoints = []
    # convert to 1 channel
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # use blur if needed
    imgBlur = cv2.GaussianBlur(img, (3,3), 1)
    imgCanny = cv2.Canny(imgBlur, 50, 50)
    # copy original image to draw onto it
    imgcpy = img.copy()
    # crop img out
    imggCropped = cropout(imgCanny,imgcpy )
    enhanced = cv2.resize(imggCropped,(600,200))
    imgContour = enhanced.copy()
    findColor(imgContour, myColors, penColors, imgContour, enhanced, myPoints)
    #print(myPoints)
    #cv2.imwrite( "imgs/test.png", enhanced )
    imgSrc = enhanced.copy()
    outputs = getIndividualPic(imgSrc, myPoints, myColors)
    for i in range(len(outputs)):
        outputs[i] = cv2.resize(outputs[i] ,(280,280))
    return outputs


# --------------------------------------- main block ------------------------------------------------
'''
This part is what to change when implementing this openCV module

change the imread input to the source of your target image

The outputs variable is a list of sliced images from input

The code only works for licenses with white background and black characters

The order in outputs is still messed up

In some cases there will be missing numbers or characters, even random slices without any characters.
'''

if __name__ == "__main__":
    # change this input image
    import matplotlib.pyplot as plt
    img = cv2.imread(r"C:\Users\Sam\Documents\GitHub\APS360-Project\Picture\2017-IWT4S-CarsReId_LP-dataset\s01_l02\74_12.png")
    # slice numbers and characters out of thage
    skip=False
    try:
        outputs = slicePic(img)
    except:
        print("An exception occurred")
        skip=True
    #if (skip==False):
    # print out images
        #imageName = 0
        #for image in outputs:
            #print(image.shape)
           # imgplot = plt.imshow(image)
            #cv2.imshow(str(imageName), image)
      #      imageName += 1
    # print original image
        #cv2.imshow('original image', img)
    # keep pictures displayed
     #   cv2.waitKey(0)
