import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

Numerical_Types = (
'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' ')

def check_3_chanel(array, rgb):
    #print(rgb.shape)
    _tmp = np.zeros((rgb.shape[0],array.shape[0], array.shape[1]))
    for t in range(rgb.shape[0]):
        #print(array.shape)
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array[i][j][0] == rgb[t][0] and  array[i][j][1] == rgb[t][1] and array[i][j][2] == rgb[t][2] :
                    _tmp[t][i][j] = 1
    return _tmp


class ImageMaskDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PNGMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PNGMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        #print(mask.shape)
        tmp = mask.reshape(-1, 3)
        # instances are encoded as different colors
        obj_ids = np.unique(tmp, axis = 0)
        #print(len(obj_ids))
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = check_3_chanel(mask,obj_ids)#tmp == obj_ids[0,None]#, None, None

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        _str = img_path.split('\\')[-1]
        _str = (_str.replace(' ', '')).split(".")[0]
        #print(_str)
        labels = []
        for i in range(len(_str)):
            labels.append(Numerical_Types.index(_str[i]))
        labels = torch.ones((num_objs,), dtype=torch.int64)
        #labels = torch.IntTensor(labels).type(dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            #img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.imgs)



def show_img_with_boxes(img, target = None, prediction =None):
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    if target is not None:
        box = target['boxes']
        for i in range(box.shape[0]):
            rect = patches.Rectangle((box[i, 0], box[i, 1]), box[i, 2] - box[i, 0], box[i, 3] - box[i, 1],
                                     fill=False, edgecolor="red")
            ax.add_patch(rect)
    if prediction is None:
        plt.show()
    else:
        box = prediction['boxes']
        scores = prediction["scores"]
        top_7_idx = torch.topk(scores, 7)[1]
        for t in range(top_7_idx.shape[0]):
            i = top_7_idx[t]
            if scores[i]>0.5:
                rect = patches.Rectangle((box[i, 0], box[i, 1]), box[i, 2] - box[i, 0], box[i, 3] - box[i, 1],
                                         fill=False, edgecolor="yellow")
                ax.add_patch(rect)
        plt.savefig("result.png")
        plt.show()

if __name__ == "__main__":
    root = "C:/Users/bowen/Documents/APS360/project/APS360-project/APS360-Project/Model/Image"
    data = ImageMaskDataset(root, None)
    print(len(data))
    for img, label in data:
        img = np.asarray(img)
        # plt.imshow(img)
        print(img.shape, label)
        show_img_with_boxes(img, label)
        break