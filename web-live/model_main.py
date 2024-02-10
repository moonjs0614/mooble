import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from utils.plots import Annotator, colors
from utils.general import non_max_suppression


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(main_img, device, size=None):
    img = main_img.copy()
    if size is not None:
        img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.moveaxis(img, -1, 0)
    img = torch.from_numpy(img).to(device)
    img = img.float()/255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


def detect(model, img):
    """
      Perform prediction of an image with given torch model

      :param torch.nn.Module model: prediction model
      :param PIL.Image.Image img: an image to predict
      :return: an array of predictions [x0,y0,x1,y1,conf,index_label]
    """
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.25)
    labels = model.names

    items = []
    if len(pred) and pred[0] is not None:
        for p in pred[0]:
            row = []
            x0, y0, x1, y1 = p[:4].tolist()
            conf = float(p[4])
            idx_label = int(p[-1])
            row = [x0, y0, x1, y1, conf, idx_label, labels[idx_label]]
            items.append(row)
    return items

def box_label(pred, img, show_label=False):
    img_pil = Image.fromarray(img)
    boxes = []

    for p in pred:
        box = tuple(map(int, p[:4]))
        conf = p[4]
        c = int(p[5])
        label = p[-1]
        text = f'{label} {conf:.2f}'
        if show_label == True:
            annotator = Annotator(img)
            annotator.box_label(box, text, colors(c, True))
            result_img = annotator.result()
        else:
            boxes.append(box)

    for box in boxes:
        box_region = img_pil.crop(box)
        box_blurred = box_region.filter(ImageFilter.GaussianBlur(radius=50))
        img_pil.paste(box_blurred, box)
    img_np = np.array(img_pil)
            
    return img_np if not show_label else result_img