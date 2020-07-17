# This file is for generating random plate images

# random select number
# select fonts of the number
# generate plates
# attach plates to a background
# do some linear transformation

import os
import random
import numpy as np
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt

Numerical_Types = (
'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' ')

FONT_PATH = "C:/Users/bowen/Documents/APS360/project/APS360-project/APS360-Project/Model/Font/"
FONT_LIST = [f for f in os.listdir(FONT_PATH) if f.endswith('.ttf')]
font_name = "Driver Gothic W01 Regular.ttf" #random.choice(FONT_LIST)
def select_random_number():
    alphaNumerical_Types = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M','N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' ')
    #np.random.seed(1000)
    plate_num = np.random.randint(0, 36, size = 7)
    _str= ""
    for i in range(8):
        if i == 3:
            _str += alphaNumerical_Types[36]
        else:
            _str+=alphaNumerical_Types[plate_num[i-(i>4)]]
    return _str

print(select_random_number())

def load_font(font_name, out_height):
    font_size = out_height*4 # approximately
    font_path = os.path.join(FONT_PATH, font_name)
    font = ImageFont.truetype(font_path, font_size) # *.ttf files load by truetype
    height = max(font.getsize(c)[1] for c in Numerical_Types)   # get maximum height in the char tuple
    for c in Numerical_Types:
        width = font.getsize(c)[0]  # getsize return size of char tuple(width, height)
        im = Image.new("RGBA", (width,height), (0,0,0))  # construct a black RGBA picture
        draw = ImageDraw.Draw(im)
        draw.text((0,0), c, (255,255,255), font = font)
        scale = float(out_height) /height
        im = im.resize((int(width*scale), out_height), Image.ANTIALIAS)
        yield c, np.array(im)[:,:,0].astype(np.float32)/255

loader = dict(load_font(font_name, 80))

f1 = plt.figure(figsize=(36,6))
i = 0
for c, img in loader:
    print(img.shape)
    ax = f1.add_subplot(6, 36/6, i+1, xticks=[], yticks=[])
    plt.imshow(img)
    ax.set_title(c)
    i+=1

#def pick_color():
    #text_color =

def num_to_plate(str, loader, out_shape):
    right_h_pad = 40
    left_h_pad = 30
    vertical_pad = 60
    #np.zeros()
    text_mask=loader[str[0]]
    for i in range(1, len(str)):
        text_mask = np.concatenate((text_mask, loader[str[i]]),axis =1)
    plt.imshow(text_mask)
    text_mask = np.dstack((text_mask,text_mask,text_mask))
    plt.imshow(text_mask)
    print(text_mask[0][0][0])
    l_h_pad = np.zeros(( text_mask.shape[0],left_h_pad, 3))
    r_h_pad = np.zeros(( text_mask.shape[0],right_h_pad, 3))
    text_mask=np.concatenate((l_h_pad, text_mask,r_h_pad), axis =1)
    v_pad = np.zeros((vertical_pad, text_mask.shape[1], 3))
    text_mask = np.concatenate((v_pad, text_mask, v_pad), axis = 0)
    plate_bg = Image.open(FONT_PATH+"../plate/blank_ontario-plate.jpg")
    plate_bg=plate_bg.resize((text_mask.shape[1],text_mask.shape[0]), Image.ANTIALIAS)
    plate_bg = np.array(plate_bg)[:, :, :].astype(np.float32) / 255
    print(text_mask[0][0][0])
    _mask = (text_mask==0)
    plate = np.copy(text_mask)
    plate[_mask] = plate_bg[_mask]
    plt.imshow(plate)
    #plate = np.ones(out_shape)
num_to_plate("ONT JOBS", loader, 1)


# edit picture to plate backgound
from PIL import Image
im = Image.open(FONT_PATH+"../plate/ontario-plate.jpg")
print(im.width)
im.show()