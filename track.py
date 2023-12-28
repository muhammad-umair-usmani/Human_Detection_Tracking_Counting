import numpy as np
import cv2
import random
import os
import torch
from boxmot import tracker_zoo
from pathlib import Path
import argparse
from boxmot.utils import ROOT, WEIGHTS




def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)
# plot_one_box(bboxes, overlayImage, label=label, color=color, line_thickness=line_thickness, bottom_label=bottom_label)
def plot_one_box(x, img, color=None, label=None, line_thickness=None, bottom_label=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1# line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    
    tf = max(tl - 1, 1)# font thickness
    if label:
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c4 = c1[0] + t_size[0], c1[1] - t_size[1] - 3# filled
        cv2.rectangle(img, c1, c4, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    if bottom_label:
        a_size = cv2.getTextSize(bottom_label, 0, fontScale=tl / 4, thickness=tf)[0]
        c3 = c1[0] + a_size[0], c2[1] + a_size[1] + 3
        cv2.rectangle(img, (c1[0], c2[1]), c3, color, -1, cv2.LINE_AA)
        cv2.putText(img, bottom_label, (c1[0], c2[1] + 12), 0, tl / 4, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
def resize_img(im, target_width = 640):
    h,w,_  = im.shape
    target_height = int(h / w * target_width)
    im = cv2.resize(im , (target_width , target_height), interpolation = cv2.INTER_AREA)  
    return im,target_height,target_width
def xyxy_xywh(x1,y1,x2,y2):
    w = abs(x2-x1)
    h = abs(y2-y1)
    x,y = x1+w/2,y1+h/2
    return x,y,w,h
def xywh_xyxy(x,y,w,h):
    x1 = x-w/2
    y1 = y-h/2
    x2 = x+w/2
    y2 = y+h/2
    return int(x1),int(y1),int(x2),int(y2)

def scale_con(x,y,w,h,old,new):
    # old : tuple(height,width)
    x = int(x* old[0]/new[0])
    y = int(y*old[1]/new[1])
    h = int(h*old[0]/new[0])
    w = int(w*old[1]/new[1])
    return x,y,w,h 

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default='./yolov5/runs/train/exp3/weights/best.pt',
                        help='yolov5m weights path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x1_0_dukemtmcreid.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='botsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--input-path', type=str, default='./video/crowd_street.mp4',
                        help='video file input path')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--output-path', default='./video/output_crowd_street.mp4',
                        help='save results to path')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')

    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    args = parse_opt()
    display = args.show
    # Model Initialization for detection

    model = torch.hub.load("./yolov5", 'custom', path=args.yolo_model, source='local')
    model.conf = args.conf
    model.iou = args.iou
    if args.device != "cpu":
        model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Tracker Initialization
    path = args.reid_model
    tracker = tracker_zoo.create_tracker(args.tracking_method, "./boxmot/configs/{}.yaml".format(args.tracking_method),
        path, args.device , False, args.per_class)
    # Video Reading
    cap = cv2.VideoCapture(args.input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Video writer

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'VP80')
    output = cv2.VideoWriter(args.output_path, fourcc, fps, (frame_width, frame_height))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            img,res_height,res_width = resize_img(frame)
            results = model(img)
            preds = np.array(results.xyxy[0])
            bboxes = []
            for ind,(x1,y1,x2,y2,conf,cls) in enumerate(preds):
                # print(ind,x1,y1,x2,y2,conf,cls)
                x1,y1,x2,y2 = scale_con(x1,y1,x2,y2,(frame_height,frame_width),(res_height,res_width))
                bboxes.append([x1,y1,x2,y2,conf,cls])
            if len(bboxes) == 0:
                bboxes = [[0,0,0,0,0.0,0]]
            outputs = tracker.update(np.array(bboxes),frame)
            for x1,y1,x2,y2,tid,conf,_,_ in  outputs:
                plot_one_box(x=(x1,y1,x2,y2),img=frame,color=compute_color_for_id(int(tid))
                            ,label=str(int(tid)),line_thickness=1)
            output.write(frame)
            
            if args.show:
                cv2.imshow('Frame',frame)
                # press esc to quit video
                if cv2.waitKey(25) == 27:
                    break
        else:
            break
    cap.release() 
    if(args.show):
        cv2.destroyAllWindows() 
    output.release()

            



