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
from PIL import Image, ImageFilter
from PIL import ImageDraw
import matplotlib.pyplot as plt
import shutil
import transforms
import PIL.ImageOps

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


def num_to_plate(str, loader, out_shape):
    right_h_pad = 40
    left_h_pad = 30
    vertical_pad = 60
    #np.zeros()
    rainbow = np.array([255,0,0, 255,127,0, 255,255,0, 255,255,255,0,255,0, 0,0,255, 46,43,95,139,0,255]).reshape((8,3))
    rainbow = rainbow.astype(np.float32)/255
    text_mask=loader[str[0]]
    shit_mask = np.stack(((loader[str[0]]!=0)*rainbow[0][0], (loader[str[0]]!=0)*rainbow[0][1],(loader[str[0]]!=0)*rainbow[0][2]))
    print(text_mask.shape, shit_mask.shape)
    for i in range(1, len(str)):
        text_mask = np.concatenate((text_mask, loader[str[i]]),axis =1)
        tmp = np.stack(((loader[str[i]]!=0) * rainbow[i][0], (loader[str[i]]!=0) * rainbow[i][1],(loader[str[i]]!=0) * rainbow[i][2]))
        shit_mask = np.concatenate((shit_mask,tmp), axis = 2)
    # plt.imshow(text_mask)
    shit_mask = np.transpose(shit_mask, [1,2,0])
    #plt.imshow(shit_mask)
    text_mask = np.dstack((text_mask,text_mask,text_mask))
    #plt.imshow(text_mask)
    l_h_pad = np.zeros(( text_mask.shape[0],left_h_pad, 3))
    r_h_pad = np.zeros(( text_mask.shape[0],right_h_pad, 3))
    text_mask=np.concatenate((l_h_pad, text_mask,r_h_pad), axis =1)
    shit_mask = np.concatenate((l_h_pad, shit_mask, r_h_pad), axis=1)
    v_pad = np.zeros((vertical_pad, text_mask.shape[1], 3))
    text_mask = np.concatenate((v_pad, text_mask, v_pad), axis = 0)
    shit_mask = np.concatenate((v_pad, shit_mask, v_pad), axis=0)
    plate_bg = Image.open(FONT_PATH+"../plate/blank_ontario-plate.jpg")
    plate_bg=plate_bg.resize((text_mask.shape[1],text_mask.shape[0]), Image.ANTIALIAS)
    plate_bg = np.array(plate_bg)[:, :, :].astype(np.float32) / 255
    _mask = (text_mask==[0,0,0])
    plate = np.copy(text_mask)
    plate[_mask] = plate_bg[_mask]
    mask_bg = np.zeros(plate.shape)
    _mask = (shit_mask == [0,0,0])
    #print(_mask)
    ma = np.copy(shit_mask)
    ma[_mask] = mask_bg[_mask]
    return plate, ma

import cv2
from scipy.spatial.transform import Rotation as R
# from skimage import transform as t
# from skimage import io
# import math
# def random_rotation(photo):
#     image = io.imread(photo)
#     afine_tf = t.AffineTransform(rotation=math.pi/6, shear=0.2)
#     # Apply transform to image data
#     modified = t.warp(image, inverse_map=afine_tf)
#     # Display the result
#     io.imshow(modified)
#     io.show()
#
#
#
# photo_path = "C:/Users/bowen/Documents/APS360/project/APS360-project/APS360-Project/Model/Image/PNGImages"
# photo = photo_path + "/0AE IIT0.png"
# img = np.asarray(Image.open(photo))
# random_rotation(photo)
#
# def photo_transform(photo):
#     cv2.getRotationMatrix2D()

# from torchvision.transforms import functional as F
# shear = [5,5]
# ret0 = (30, (0, 0), 0.5, shear)
# _rd = select_random_number()
# plate, ma = num_to_plate(_rd, loader, 1)
# print(plate.shape, ma.shape)
# im = Image.fromarray((plate * 255).astype(np.uint8))
# im2 = Image.fromarray((ma * 255).astype(np.uint8))
# img_tuple = (im,im2)
# img1= F.affine(img_tuple[0], *ret0, resample=0, fillcolor=0)
# img2 = F.affine(img_tuple[1], *ret0, resample=0, fillcolor=0)
# img1.show()
# img2.show()
#
# shear = [5,5,5,5]
# transform = transforms.RandomAffine(30, scale=(0.3, 1.1), shear=shear)
# print(transform)
# _rd = select_random_number()
# plate, ma = num_to_plate(_rd, loader, 1)
# print(plate.shape, ma.shape)
# im = Image.fromarray((plate * 255).astype(np.uint8))
# im2 = Image.fromarray((ma * 255).astype(np.uint8))
# img = transform((im,im2))
# print(img[0])
# img[1].show()
# img[2].show()
# #im.save('IMAGE/PNGImages/{}.png'.format(_rd))

# def add_background():
#
#
#

def perspect_blur_bright_contrast(img, mask,target_shape, blur, b_range, c_range):
    size = img.size
    t_shape = (target_shape[0], int(size[1] / (size[0] / target_shape[0]) * target_shape[1]))
    resized_im = img.resize(t_shape, Image.ANTIALIAS)
    resized_im2 = mask.resize(t_shape, 0)
    # img = transform((im,im2))
    im_blur = resized_im.filter(ImageFilter.GaussianBlur(radius=blur))
    perspect = transforms.RandomPerspective(distortion_scale=0.5, p=1, interpolation=False, fill=0)
    perspective_img = perspect([im_blur, resized_im2])
    # perspective_img[0].show()
    # perspective_img[1].show()
    bright = random.uniform(b_range[0], b_range[1])
    contrast = random.uniform(c_range[0], c_range[1])
    bright_img = F.adjust_brightness(perspective_img[0], bright)
    contrast_img = F.adjust_contrast(bright_img, contrast)
    return contrast_img, perspective_img[1]

import torchvision
import torch
from torchvision.transforms import functional as F
if __name__ == "__main__":
    regen = False
    root_dir = "C:/Users/bowen/Documents/APS360/project/APS360-project/APS360-Project/Model/"
    image_dir = "Image_pers_blur/"
    original = "PNGImages/"
    mask = "PNGMasks/"
    target_shape = [150,1]
    if os.path.exists(root_dir+image_dir):
        if regen:
            shutil.rmtree(root_dir+image_dir)
            os.makedirs(root_dir+image_dir)
            os.makedirs(root_dir + image_dir+original)
            os.makedirs(root_dir + image_dir+mask)
    else:
        os.makedirs(root_dir + image_dir)
        os.makedirs(root_dir + image_dir + original)
        os.makedirs(root_dir + image_dir + mask)
    for i in range(500):
        #shear = [3, 3, 5, 5]
        #transform = transforms.RandomAffine(30, scale=(0.3, 1.1), shear=shear)
        _rd = select_random_number()
        plate, ma = num_to_plate(_rd, loader, 1)
        im = Image.fromarray((plate * 255).astype(np.uint8))
        im2 = Image.fromarray((ma * 255).astype(np.uint8))
        img, mask = perspect_blur_bright_contrast(im, im2, target_shape, blur=1.2, b_range=[0.8,1.5], c_range=[0.8,1.5])
        # img.show()
        # mask.show()
        #break
        img.save(image_dir+'/PNGImages/{}.png'.format(_rd))
        mask.save(image_dir+'/PNGMasks/{}_mask.png'.format(_rd))

