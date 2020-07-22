# Main.py

import cv2
import numpy as np
import os
import pytesseract
import DetectChars
import DetectPlates
import PossiblePlate

# module level variables ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False

###################################################################################################
def main(img):

    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()         # attempt KNN training

    if blnKNNTrainingSuccessful == False:                               # if KNN training was not successful
        print("\nerror: KNN traning was not successful\n")  # show error message
        return                                                          # and exit program
    # end if

    imgOriginalScene  = img             # open image

    if imgOriginalScene is None:                            # if image was not read successfully
        print("\nerror: image not read from file \n\n")  # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit program
    # end if

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           # detect plates

    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # detect chars in plates

    #cv2.imshow("imgOriginalScene", imgOriginalScene)            # show scene image

    if len(listOfPossiblePlates) == 0:                          # if no plates were found
        print("\nno license plates were detected\n")  # inform user no plates were found
    else:                                                       # else
                # if we get in here list of possible plates has at leat one plate

                # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

                # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
        licPlate = listOfPossiblePlates[0]

        #cv2.imshow("imgPlate", licPlate.imgPlate)           # show crop of plate and threshold of plate
        #cv2.imshow("imgThresh", licPlate.imgThresh)

        if len(licPlate.strChars) == 0:                     # if no chars were found in the plate
            print("\nno characters were detected\n\n")  # show message
            return                                          # and exit program
        # end if

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)             # draw red rectangle around plate

        #print("\nlicense plate read from image = " + licPlate.strChars + "\n")  # write license plate text to std out
        #("----------------------------------------")

        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)           # write license plate text on the image

        #cv2.imshow("imgOriginalScene", imgOriginalScene)                # re-show scene image

        #cv2.imwrite("imgOriginalScene.png", imgOriginalScene)           # write image out to file

    # end if else

    cv2.waitKey(0)					# hold windows open until user presses a key

    return licPlate.strChars
# end main

###################################################################################################
def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            # get 4 vertices of rotated rect

    #cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)         # draw 4 red lines
    #cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    #cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    #cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)
# end function

###################################################################################################
def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0                             # this will be the center of the area the text will be written to
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0                          # this will be the bottom left of the area that the text will be written to
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX                      # choose a plain jane font
    fltFontScale = float(plateHeight) / 30.0                    # base font scale on height of plate area
    intFontThickness = int(round(fltFontScale * 1.5))           # base font thickness on font scale

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)        # call getTextSize

            # unpack roatated rect into center point, width and height, and angle
    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)              # make sure center is an integer
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)         # the horizontal location of the text area is the same as the plate

    if intPlateCenterY < (sceneHeight * 0.75):                                                  # if the license plate is in the upper 3/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))      # write the chars in below the plate
    else:                                                                                       # else if the license plate is in the lower 1/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))      # write the chars in above the plate
    # end if

    textSizeWidth, textSizeHeight = textSize                # unpack text size width and height

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))           # calculate the lower left origin of the text area
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))          # based on the text area center, width, and height

            # write the text on the image
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)
# end function

###################################################################################################




licenseGuess = ''

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files (x86)\Tesseract-OCR\\tesseract.exe'
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

def furtherCut(img, myColors):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
    # cut off horizontal bias
    count = 10
    for i in range(1,10):
        binaryImg = mask[:,(i-1)*5:i*5]
        height, width = binaryImg.shape[:2]
        #("a-------------------", height, width)
        whitePercent = cv2.countNonZero(binaryImg)/(height*width)
        if whitePercent> 0.7:
            count = i
            #print('percentage:', whitePercent)
            break
    #img = img[:, count * 5:]

    # cut off vertical bias
    countvertop = 10
    for i in range(1, 10):
        binaryImg = mask[(i - 1) * 5:i * 5, :]
        height, width = binaryImg.shape[:2]
        whitePercent = cv2.countNonZero(binaryImg) / (height * width)
        if whitePercent > 0.5:
            countvertop = i
            #print('percentage:', whitePercent)
            break
    # cut top and left
    img = img[countvertop*5:,count*5:]

    countverbottom = 10
    for i in range(1, 10):
        height, width = img.shape[:2]
        binaryImg = mask[height-(i * 5): height- ((i - 1) * 5), :]

        whitePercent = cv2.countNonZero(binaryImg) / (height * width)
        if whitePercent < 0.5:
            countverbottom = i
            #print('percentage:', whitePercent)
            break

    img = img[:(height - countverbottom*5), :]


    return img




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

        licenseGuess =pytesseract.image_to_string(mask)
        #print(licenseGuess)



        # ----------------------------------------------------------
        pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files (x86)\Tesseract-OCR\\tesseract.exe'

        imggg = mask.copy()

        imggg = cv2.cvtColor(imggg, cv2.COLOR_BGR2RGB)

        hImg, wImg, _ = imggg.shape
        boxes = pytesseract.image_to_boxes(imggg)
        for b in boxes.splitlines():
            #print(b)
            b = b.split(' ')
            print(b)
            x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            cv2.rectangle(imggg, (x, hImg - y), (w, hImg - h), (50, 50, 255), 2)
            cv2.putText(imggg, b[0], (x, hImg - y + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)

        #cv2.imshow('imggg', imggg)

        # ----------------------------------------------------------------------------------------

    return mask1, licenseGuess


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
        if 1000 < area < 9000:
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
            # if a bounding box has width > height, it must be not a letter
            if w < h:
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
    #print(len(picList))
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

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)

    imggCropped = cropout(mask, imgcpy)

    if imggCropped is None:
        imggCropped = imgcpy.copy()
    enhanced = cv2.resize(imggCropped,(600,200))
    enhanced= furtherCut(enhanced, myColors)
    enhanced = cv2.resize(enhanced, (600, 200))
    #cv2.imshow('further enhanced', enhanced)




    imgContour = enhanced.copy()
    mask1, licenseGuess = findColor(imgContour, myColors, penColors, imgContour, enhanced, myPoints)
    #print(myPoints)
    #cv2.imwrite( "imgs/test.png", enhanced )
    imgSrc = enhanced.copy()
    outputs = getIndividualPic(imgSrc, myPoints, myColors)
    for i in range(len(outputs)):
        outputs[i] = cv2.resize(outputs[i] ,(280,280))
    return outputs, licenseGuess


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
    img = cv2.imread("imgs/10.png")
    dalao_predict = main(img)
    # slice numbers and characters out of the image
    outputs, licenseGuess = slicePic(img)
    if len(licenseGuess) != 0:
        licenseGuess = licenseGuess.replace(" ", "")
        licenseGuess = licenseGuess.upper()
    if len(dalao_predict) != 0:
        dalao_predict = dalao_predict.replace(" ", "")
        dalao_predict = dalao_predict.upper()
    # prevent head over reading
    if len(licenseGuess)>7:
        licenseGuess = licenseGuess[-7:]
    if len(dalao_predict) > 7:
        dalao_predict = dalao_predict[-7:]

    # check our model predict vs dalao's

    if len(licenseGuess) == len(dalao_predict):
        temp = ""
        for i in range(len(licenseGuess)):
            if not dalao_predict[i].isalnum():
                if licenseGuess[i].isalnum():
                    temp += licenseGuess[i]
                else:
                    temp += dalao_predict[i]
            else:
                temp+=dalao_predict[i]
        dalao_predict = temp

    if len(dalao_predict) < 7 and len(licenseGuess) == 7:
        temp = list(licenseGuess)
        for i in range(len(dalao_predict)-1,-1,-1):
            if not licenseGuess[i].isalnum():
                temp[i] = dalao_predict[i]
        dalao_predict = temp.join()
    elif len(dalao_predict) < 7:
        dalao_predict = licenseGuess[:7-len(dalao_predict)] + dalao_predict

    # print out images
    imageName = 0
    for image in outputs:
        #cv2.imshow(str(imageName), image)
        # cv2.imwrite( "imgs/tests", enhanced )
        imageName += 1
    prediction = dalao_predict

    # print original image
    #cv2.imshow('original image', img)
    print(prediction)
    # keep pictures displayed
    cv2.waitKey(0)
























