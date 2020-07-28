import glob
import cv2
import numpy as np
from skimage.util import random_noise
import random
from random import randrange




# ------------------------------------------------------------- change rotation
def changeRotate(images):
    new_set = []
    for image in images:
        for angle in range(-20,20,2):
            (h1, w1) = image.shape[:2]
            center = (w1 / 2, h1 / 2)
            Matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_image = cv2.warpAffine(image, Matrix, (w1, h1))
            new_set.append(rotated_image)
    return new_set

# ------------------------------------------------------------- change crop
def changeCrop(images):
    new_set = []
    for img in images:
        for i in range(20):
            height = img.shape[0]
            width = img.shape[1]

            width_left = int(randrange(int(width*0.13)))
            width_right = width - int(randrange(int(width*0.13)))
            height_top = int(randrange(int(height*0.2)))
            height_bottom = height - int(randrange(int(height * 0.2)))

            cropped = img[height_top:height_bottom , width_left: width_right]
            new_set.append(cropped)
    return new_set

# ------------------------------------------------------------- change noise

def changeNoise(images):
    new_set = []
    # Load the image
    for img in images:
        for noise_loop in range(1, 20):
            noise = noise_loop/100
            # Add salt-and-pepper noise to the image.
            noise_img = random_noise(img, mode='s&p', amount=0.01)

            # The above function returns a floating-point image
            # on the range [0, 1], thus we changed it to 'uint8'
            # and from [0,255]
            noise_img = np.array(255 * noise_img, dtype='uint8')
            new_set.append(noise_img)
    return new_set
# ------------------------------------------------------------- Explosure and contrast
def changeExplosureAndContrast(images):
    newSet = []
    for image in images[0:1]:

        #Enter the alpha value [1.0-3.0]
        for alpha1 in range(10, 20, 1):
            alpha = alpha1/10
            #Enter the beta value [0-100]
            for beta in range (30, 70, 5):
                new_image = np.zeros(image.shape, image.dtype)

                for y in range(image.shape[0]):
                    for x in range(image.shape[1]):
                        for c in range(image.shape[2]):

                            new_image[y, x, c] = np.clip(alpha * image[y, x, c] + beta, 0, 255)
                            newSet.append(new_image)
    return newSet


# ------------------------------------------------------------- blur
def changeBlur(images):
    blurred = []
    for img in images:
        for i in range(1,17, 4):
            for j in range(1, 17, 4):
                for k in range(0, 2, 1):
                    blurred.append(cv2.GaussianBlur(img, (i, j), k))
    return blurred



# ------------------------------------------------------------- Main function
# 请确保 Crop 文件夹 与本文件在同一root里
# num指的是像在各个文件夹里添加多少张新的image， 因直接处理image过多， 大于30w张，故提供张数选择
# option --------->
#   if option == 1, 在crop文件夹的每个子文件夹中直接添加写入新生成的图片
#   if option == 2, 不写入新图片，return一个numpy array，格式为 [[0的图像]，[1的图像]... ,[Y的图像] [Z的图像]]
#   if option == 3, option 1 和 2 都进行， 写入新图片，同时return一个numpy array，格式为 [[0的图像]，[1的图像]... ,[Y的图像] [Z的图像]]

def main(num, option):
    all = []
    for letter in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        #01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ
        path = "./Crop/" + letter
        images = [cv2.imread(file) for file in glob.glob(path + "/*.png")]
        original = images[:]

        # ------------------------------------------------------- 制造image set
        images.extend(changeExplosureAndContrast(images))
        random.shuffle(images)
        random.shuffle(images)
        random.shuffle(images)
        images = images[:num]
        images.extend(changeBlur(images))
        random.shuffle(images)
        images = images[:num]
        images.extend(changeNoise(images))
        random.shuffle(images)
        images = images[:num]
        images.extend(changeCrop(images))
        random.shuffle(images)
        images = images[:num]
        images.extend(changeRotate(images))
        random.shuffle(images)
        images = images[:num]

        if option == 1:
            for i in range(len(images)):
                cv2.imwrite(path + "/"  + str(i) + "_NEW_"  +  ".png", images[i])
            print(letter + " done")
        elif option == 2:
            original.extend(images)
            all.append(original)
        elif option == 3:
            for i in range(len(images)):
                cv2.imwrite(path + "/" + str(i) + "_NEW_" + ".png", images[i])
            print(letter + " done")
            original.extend(images)
            all.append(original)

    if option == 2 or option == 3:
        return all




# ----------------------------------------- 运行程序
# 请确保解压后的Crop 文件夹 与本文件在同一root里
# num指的是像在各个文件夹里添加多少张新的image， 因直接处理image过多， 大于30w张，故提供张数选择
# 例如如果想要1000张新图像， 则在选择option后， 在num处填1000
# option --------->
#   if option == 1, 在crop文件夹的每个子文件夹中直接添加写入新生成的图片
#   if option == 2, 不写入新图片，return一个numpy array，格式为 [[0的图像]，[1的图像]... ,[Y的图像] [Z的图像]]
#   if option == 3, option 1 和 2 都进行， 写入新图片，同时return一个numpy array，格式为 [[0的图像]，[1的图像]... ,[Y的图像] [Z的图像]]
if __name__ == "__main__":
    all = main(10,3)
    print(all)