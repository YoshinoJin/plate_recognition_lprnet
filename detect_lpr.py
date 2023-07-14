import argparse
import time
import onnxruntime as ort
import numpy as np
import cv2
import os
import re
# from utils.datasets import letterbox
from utils.plots import *
from models.LPRNet import CHARS

def reg_postprocess(prebs, inference_cfg=None):
    preb_labels = list()
    for i in range(prebs.shape[0]):
        preb = prebs[i, :, :]  # 对每张图片 [68, 18]
        preb_label = list()
        for j in range(preb.shape[1]):  # 18  返回序列中每个位置最大的概率对应的字符idx  其中'-'是67
            preb_label.append(np.argmax(preb[:, j], axis=0))
        no_repeat_blank_label = list()
        pre_c = preb_label[0]
        if pre_c != len(CHARS) - 1:  # 记录重复字符
            no_repeat_blank_label.append(pre_c)
        for c in preb_label:  # 去除重复字符和空白字符'-'
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        preb_labels.append(no_repeat_blank_label)  # 得到最终的无重复字符和无空白字符的序列
    
    return preb_labels

def detect_onnx(save_img=True, save_conf=True):
    source, clas_weights, save_dir, view_img, save_txt, imgsz = opt.source, opt.classifi_weights, opt.save_dir, opt.view_img, opt.save_txt, opt.img_size
    iou_thres, conf_thres = opt.iou_thres, opt.conf_thres
    if save_img:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    # read input file path
    imgs_path = []
    if os.path.isfile(source):
        with open(source, 'r') as f:
            lines = f.readlines()
        for line in lines:
            imgs_path.append(line.split("\n")[0])
    elif os.path.isdir(source):
        imgs_path = os.listdir(source)
    else:
        raise Exception("the input file is file or dir")


    # load detection and classfication onnx model
    ort_class_session = ort.InferenceSession(clas_weights, providers=['CUDAExecutionProvider'])

    for img in imgs_path:
        print(img)
        image_ori = cv2.imread(os.path.join(source, img))
        # proprocess
        image_clas = image_ori
        print(image_clas.shape)
        image_clas = image_clas.astype('float32')
        image_clas -= 127.5
        image_clas *= 0.0078125
        image_clas = np.transpose(image_clas, (2, 0, 1))
        # recognization inference
        print(image_clas.shape)
        probs = ort_class_session.run(output_names=['138'], input_feed={'input.1': [image_clas]})[0]
        # print(probs)
        probs = reg_postprocess(probs)

        for prob in probs:
            lb = ""
            for i in prob:
                lb += CHARS[i]
            cls = lb


        label = f'names{[str(cls)]}'
        print(label)
        # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)


            # Save results (image with detections)
    if save_img:
        img_path = os.path.join(save_dir, img.split("/")[-1])
        cv2.imwrite(img_path, image_ori)


                


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifi_weights', nargs='+', type=str,
                        default=r"/home/zj/plate_detection_recognization/code_plate_detection_recognization/weights/LPRNet_Simplified.onnx",
                        help='classification model path(s)')
    parser.add_argument('--source', type=str, default=r"/home/zj/plate_detection_recognization/code_plate_detection_recognization/demo/images",
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--save_dir', type=str, default=r"/home/zj/plate_detection_recognization/code_plate_detection_recognization/demo/output",
                        help='source')  # folder,
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.3, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=False, help='display results')
    parser.add_argument('--save-txt', default=True, help='save results to *.txt')
    parser.add_argument('--save-conf', default=True, help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='final_predict', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))
    with torch.no_grad():
        startt = time.time()
        detect_onnx()
        print(time.time() - startt)