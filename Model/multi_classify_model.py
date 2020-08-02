from ImageMaskdataset import *
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch
from torchvision import transforms
import math
import tkinter
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import  evaluate
import utils
import matplotlib
import transforms as T
import sys
import statistics

# load a model pre-trained pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 2  # 1 class (person) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



# load a pre-trained model for classification and return
# only the features
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# FasterRCNN needs to know the number of
# output channels in a backbone. For mobilenet_v2, it's 1280
# so we need to add it here
backbone.out_channels = 1280

# let's make the RPN generate 5 x 3 anchors per spatial
# location, with 5 different sizes and 3 different aspect
# ratios. We have a Tuple[Tuple[int]] because each feature
# map could potentially have different sizes and
# aspect ratios
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

# let's define what are the feature maps that we will
# use to perform the region of interest cropping, as well as
# the size of the crop after rescaling.
# if your backbone returns a Tensor, featmap_names is expected to
# be [0]. More generally, the backbone should return an
# OrderedDict[Tensor], and in featmap_names you can choose which
# feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=7,
                                                sampling_ratio=2)

# put the pieces together inside a FasterRCNN model
model = FasterRCNN(backbone,
                   num_classes=2,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)




def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomInvert(0.5))
    transforms.append(T.Grayscale())
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    root = "C:/Users/bowen/Documents/APS360/project/APS360-project/APS360-Project/Model/Image_pers_blur"
    trans = transforms.Compose([transforms.ToTensor()])
    dataset = ImageMaskDataset(root, get_transform(train=True))
    dataset_test = ImageMaskDataset(root, get_transform(train=False))
    print(len(dataset), len(dataset_test))
    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50]) #-50
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:]) #-50
    print(len(dataset), len(dataset_test))
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
         dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0005,
                                momentum=0.5, weight_decay=0.005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=1,
                                                   gamma=0.5)

    # let's train it for 5 epochs
    num_epochs = 5

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")
    return model



def show_result(model, data_loader, cuda):
    device = torch.device('cuda') if torch.cuda.is_available() and cuda else torch.device('cpu')
    model.to(device)
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(10)
    for images, targets in metric_logger.log_every(data_loader, 10, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        for image, target in zip(images,targets):
            plt.imshow(np.transpose(image.cpu().numpy(),[1,2,0]))
            #image = np.transpose(image, [1, 2, 0])
            # print(image.shape)
            loss_dict = model([image])#, [target])
            # print(len(loss_dict[0]['boxes']))
            # print(len(loss_dict[0]['masks']))
            # print(loss_dict[0])
            show_img_with_boxes(np.transpose(image.cpu().numpy(),[1,2,0]), target,loss_dict[0])
            #break
        break
# model = main()
# root = "C:/Users/bowen/Documents/APS360/project/APS360-project/APS360-Project/Model/Image"
# trans = transforms.Compose([transforms.ToTensor()])
# dataset = ImageMaskDataset(root, get_transform(train=True))
# dataset_test = ImageMaskDataset(root, get_transform(train=False))
#
# # split the dataset in train and test set
# indices = torch.randperm(len(dataset)).tolist()
# dataset = torch.utils.data.Subset(dataset, indices[:-50])
# dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
# data_loader = torch.utils.data.DataLoader(
#     dataset, batch_size=2, shuffle=True, num_workers=0,collate_fn=utils.collate_fn)
#
# data_loader_test = torch.utils.data.DataLoader(
#      dataset_test, batch_size=1, shuffle=False, num_workers=0,
#     collate_fn=utils.collate_fn)
# show_result(model, data_loader, 1)
#
# dirname = "C:/Users/bowen/Documents/APS360/project/APS360-project/APS360-Project/Model/"
# model_path = "detection_2"
# model_path = os.path.join(dirname,model_path)
# torch.save(model.state_dict(), model_path)

def image_transform(img):
    # take Image object return tensor
    img = np.asarray(img)
    print(img.shape)
    if len(img.shape) == 3:
        img = np.transpose(img,[2,0,1])
    elif len(img.shape) == 2:
        img = img
    else :
        print("only support RGB or L")
        sys.exit(1)
    return torch.Tensor(img.astype(np.float32)/255)

def test_on_none_generated_data(model, img_path,label):
    img = Image.open(img_path)
    # img = Image.open(img_path).convert('RGBA')
    # background = Image.new('RGBA', img.size, (255, 255, 255))
    #
    # img = Image.alpha_composite(background, img)
    image = torch.Tensor(image_transform(img))
    print(image.shape)
    model.eval()
    loss_dict = model([image])
    #print(loss_dict)
    print(image.shape)
    show_img_with_boxes(np.transpose(image.cpu().numpy(), [1, 2, 0]), target=None, prediction=loss_dict[0])
    print(loss_dict[0]['boxes'].shape)
    crop_plate_number(image, loss_dict[0],label )

import copy
def sort_array(boxes, top_7_idx):
    box = boxes[top_7_idx]
    box = box.detach().numpy()
    top_7_idx =top_7_idx.numpy()
    changed = True
    while changed:
        changed = False
        for i in range(box.shape[0]-1):
            if box[i][0]> box[i+1][0]:
                tmp = copy.deepcopy(box[i])
                _t = copy.deepcopy(top_7_idx[i])
                box[i]=box[i+1]
                top_7_idx[i] = top_7_idx[i + 1]
                #print(tmp)
                box[i + 1] = tmp
                top_7_idx[i+1] = _t
                changed = True
                #print(box)
    return box, top_7_idx
# import torch
# boxes = torch.Tensor([[5,1],[4,2],[3,4],[2,4],[1,0]])
# top_7_idx = torch.Tensor([0,3,1])
# sort_array(boxes, top_7_idx.long())

def decide_crop_area(boxes, scores, top=5, dev = 1,th = 20):
    top_idx = torch.topk(scores, top)[1]
    _b = boxes[top_idx].detach().numpy()
    area = []
    loop = True
    for i in range(_b.shape[0]):
        a, b, c, d = _b[i, 0], _b[i, 2], _b[i, 1], _b[i, 3]
        _a = (d-c) * (b-a)
        area.append(_a.astype(np.float64))
        a+=_a
    avg = sum(area)/len(area)
    print(area, type(area[0]))
    while loop:
        if len(area) == 0:
            break
        if statistics.stdev(area) <= th*avg*0.01:
            return avg
        else:
            print(statistics.stdev(area))
            value1 = min(area)
            value2 = max(area)
            remove = value1 if abs(value1-avg)>abs(value2-avg) else value2
            area.remove(remove)
            avg = sum(area)/len(area)



def crop_plate_number(img, pred, label, th = 10, target_dir=None):

    mul = 1
    if img.max() <= 1:
        mul = 255
    boxes = pred['boxes']
    scores = pred['scores']
    # ref = decide_crop_area(boxes, scores)
    # print(ref)
    # top = min( boxes.shape[0], 20)
    top = 7
    top_7_idx = torch.topk(scores, top)[1]
    # print(scores[top_7_idx])
    img = img.numpy()
    boxes = boxes[top_7_idx]
    boxes, top_7_idx = sort_array(boxes, top_7_idx.long())
    # boxes = boxes.detach().numpy().astype(np.int32)
    # count = 0
    for i in range(boxes.shape[0]):
        # i = top_7_idx[t]
        # print(type(boxes))
        # if count == 7:
        #     break
        boxes = boxes.astype(np.uint32)
        if scores[top_7_idx[i]] > 0.5:
            a,b,c,d = boxes[i, 0],boxes[i,2],boxes[i,1],boxes[i,3]
            #print(img.shape)
            # _a = (d - c) * (b - a)
            # if abs(_a-ref)<=th*ref:
            #     count +=1
            crop = img[:, c:d,a:b]
            #print(crop.shape)
            _img = np.transpose((crop*mul).astype(np.uint8), [1,2,0])
            #print(_img)
            im = Image.fromarray(_img)
            im.save('Crop/{}.png'.format(i))


if __name__ == "__main__":
    matplotlib.use('TkAgg')
    train = False
    dirname = "C:/Users/bowen/Documents/APS360/project/APS360-project/APS360-Project/Model/"
    model_path = "detection_pers_blur"
    model_path = os.path.join(dirname,model_path)
    root = "C:/Users/bowen/Documents/APS360/project/APS360-project/APS360-Project/Model/Image_pers_blur"
    dataset = ImageMaskDataset(root, get_transform(train=True))
    dataset_test = ImageMaskDataset(root, get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
    #print(len(dataset), len(dataset_test))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
         dataset_test, batch_size=3, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)
    if train is False:
        model1 = get_model_instance_segmentation(2)
        state = torch.load(model_path)
        model1.load_state_dict(state)
        # show_result(model1, data_loader_test, 1)
        new_data = "C:/Users/bowen/Documents/APS360/project/APS360-project/APS360-Project/Model/plate/LBYRNTH.jpg"
        new_data = "C:/Users/bowen/Documents/APS360/project/2017-IWT4S-CarsReId_LP-dataset/s01_l01/4_17.png"
        test_on_none_generated_data(model1, new_data, label = "BE082AB")
    else:
        model = main()
        dirname = "C:/Users/bowen/Documents/APS360/project/APS360-project/APS360-Project/Model/"
        #model_path = "detection_affine"
        model_path = os.path.join(dirname,model_path)
        torch.save(model.state_dict(), model_path)