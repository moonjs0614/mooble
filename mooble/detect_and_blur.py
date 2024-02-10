#Object Crop Using YOLOv7
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def detect(source='0', blurratio=50):

    weights='C:/Users/user/Desktop/yolov7/yolov7/project.pt'
    device = '0'
    img_size = 640
    conf_thres=0.25
    iou_thres=0.45
    view_img=True
    classes=None
    agnostic_nms=False
    augment=False
    no_trace=False
    hidedetarea=False
    blurratio_0 = 1
    
    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(img_size, s=stride)  # check img_size

    if no_trace == True:
        model = TracedModel(model, device, img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    if source == '0':
        webcam = True
        stride = 32
        source = str(source)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        webcam = False
        source = str(source)
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                p, s, im1, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                p, s, im2, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    
                    #Add Object Blurring Code
                    #..................................................................
                    crop_obj_0 = im0[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]
                    crop_obj_1 = im1[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]
                    crop_obj_2 = im2[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]
                    blur_0 = cv2.blur(crop_obj_0,(blurratio_0,blurratio_0))
                    blur_1 = cv2.blur(crop_obj_1,(blurratio_0,blurratio_0))
                    blur_2 = cv2.blur(crop_obj_2,(blurratio,blurratio))
                    im0[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])] = blur_0
                    im1[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])] = blur_1
                    im2[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])] = blur_2
                    #..................................................................

                    if view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        if not hidedetarea:
                            plot_one_box(xyxy, im1, label=label, color=colors[int(cls)], line_thickness=3)
            
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow("original", im0)
                cv2.imshow('detection', im1)
                cv2.imshow('blurring', im2)
                cv2.waitKey(1)  # 1 millisecond

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    
    detect()