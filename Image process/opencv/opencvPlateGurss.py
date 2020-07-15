import cv2
import numpy as np
print("cv2 imported")
import pytesseract

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
            print(area)
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
        print("a-------------------", height, width)
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
            print('percentage:', whitePercent)
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
            print('percentage:', whitePercent)
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
        cv2.imshow(str(color[0]), mask1)
        # draw a line with center top of color contour
        centerx, centery = getContour(mask1, imgContour, enhanced, myPoints)
        cv2.imshow('Cropped out on stretched picture', enhanced)

        licenseGuess =pytesseract.image_to_string(mask)
        print(licenseGuess)



        # ----------------------------------------------------------
        pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files (x86)\Tesseract-OCR\\tesseract.exe'

        imggg = mask.copy()

        imggg = cv2.cvtColor(imggg, cv2.COLOR_BGR2RGB)

        hImg, wImg, _ = imggg.shape
        boxes = pytesseract.image_to_boxes(imggg)
        for b in boxes.splitlines():
            print(b)
            b = b.split(' ')
            print(b)
            x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            cv2.rectangle(imggg, (x, hImg - y), (w, hImg - h), (50, 50, 255), 2)
            cv2.putText(imggg, b[0], (x, hImg - y + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)

        cv2.imshow('imggg', imggg)

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
    return x+w//2, y


def getIndividualPic(imgSrc, points, myColors):
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
            # invert color
            picList.append(cv2.bitwise_not(imggCropped))
    print(len(picList))
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
    cv2.imshow('further enhanced', enhanced)




    imgContour = enhanced.copy()
    mask1, licenseGuess = findColor(imgContour, myColors, penColors, imgContour, enhanced, myPoints)
    print(myPoints)
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
    img = cv2.imread("imgs/12.png")
    # slice numbers and characters out of the image
    outputs, licenseGuess = slicePic(img)
    licenseGuess = licenseGuess.replace(" ", "")
    # print out images
    imageName = 0
    for image in outputs:
        cv2.imshow(str(imageName), image)
        # cv2.imwrite( "imgs/tests", enhanced )
        imageName += 1

    # print original image
    cv2.imshow('original image', img)
    print(licenseGuess)
    # keep pictures displayed
    cv2.waitKey(0)
