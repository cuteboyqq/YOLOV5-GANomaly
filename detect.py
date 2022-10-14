# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s.xml                # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box, get_crop_image
from utils.torch_utils import select_device, smart_inference_mode
from models.model_GANomaly import *


#=========================================================================
import tensorflow as tf
from tensorflow.keras import layers
print(tf.__version__)
import os
import time
import numpy as np
import cv2
#from model import GANomaly

from absl import app
from absl import flags
from absl import logging

from torch.autograd import Variable
#=======================================================================
#===========================================================================
'''
FLAGS = flags.FLAGS
flags.DEFINE_integer("shuffle_buffer_size", 10000,
                     "buffer size for pseudo shuffle")
flags.DEFINE_integer("batch_size", 64, "batch_size")
flags.DEFINE_integer("isize", 32, "input size")
flags.DEFINE_string("ckpt_dir", 'ckpt', "checkpoint folder")
flags.DEFINE_integer("nz", 100, "latent dims")
flags.DEFINE_integer("nc", 3, "input channels")
flags.DEFINE_integer("ndf", 64, "number of discriminator's filters")
flags.DEFINE_integer("ngf", 64, "number of generator's filters")
flags.DEFINE_integer("extralayers", 0, "extralayers for both G and D")

flags.DEFINE_list("encdims", None, "Layer dimensions of the encoder and in reverse of the decoder."
                                   "If given, dense encoder and decoders are used.")
flags.DEFINE_integer("niter",50,"number of training epochs")
flags.DEFINE_float("lr", 2e-4, "learning rate")
flags.DEFINE_float("w_adv", 1., "Adversarial loss weight")
flags.DEFINE_float("w_con", 50., "Reconstruction loss weight")
flags.DEFINE_float("w_enc", 1., "Encoder loss weight")
flags.DEFINE_float("beta1", 0.5, "beta1 for Adam optimizer")
flags.DEFINE_string("dataset", r'/home/ali/GitHub_Code/YOLO/YOLOV5/runs/detect/f_384_2min/crops_1cls', "name of dataset")
#flags.DEFINE_string("dataset", 'cifar10', "name of dataset")
flags.DEFINE_string("dataset_test", r'/home/ali/GitHub_Code/YOLO/YOLOV5/runs/detect/f_384_2min/crops_2cls', "name of dataset")
flags.DEFINE_string("dataset_infer", r'/home/ali/GitHub_Code/YOLO/YOLOV5/runs/detect/f_384_2min/crops_1cls', "name of dataset")
flags.DEFINE_string("dataset_infer_abnormal", r'/home/ali/GitHub_Code/cuteboyqq/GAN/GAN-Pytorch/implementations/discogan/fake_images', "name of dataset")
DATASETS = ['mnist', 'cifar10']

flags.register_validator('dataset',
                         lambda name: name in DATASETS,
                         message='--dataset must be {}'.format(DATASETS))

flags.DEFINE_integer("anomaly", 5, "the anomaly idx")
flags.mark_flag_as_required('anomaly')
flags.mark_flag_as_required('isize')
flags.mark_flag_as_required('nc')
'''
#=========================================================================



@smart_inference_mode()
def run(weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        isize=32,
        nz=100,
        nc=3,
        ndf=64,
        ngf=64,
        w_adv=1,
        w_con=50,
        w_enc=1,
        extralayers=0,
        encdims=None,
        dataset=r'/home/ali/GitHub_Code/YOLO/YOLOV5-old/runs/detect/f_384_2min/crops_1cls',
        batch_size=64,
        lr=2e-4,
        beta1=0.5,
        GANOMALY=False, #use GANoamly defeat detection
        LOSS=1.2,
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    print('Start load model')
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    print('load model successfull')
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    #===========================
    # Alister 2022-10-13 add 
    # Create GANomaly instance
    #==========================
    
    #opt_tf = FLAGS
    if GANOMALY:
        ganomaly = GANomaly(opt,
                            train_dataset=opt.dataset,
                            valid_dataset=opt.dataset,
                            test_dataset=None)
    
    
    
    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow()
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        print('Start LoadImages')
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        print('LoadImages successful')
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            print('Start inference model')
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            print('inference model successful')
        # NMS
        with dt[2]:
            print('Start NMS')
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            print('NMS done')
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        if GANOMALY:
                            if not c==0 and not c==1:
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))
                        else:
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    
                    #==================================
                    # GANomaly Alister 2022-10-13 add
                    #==================================
                    if GANOMALY:
                        abnormal = 0
                        crop = get_crop_image(xyxy, imc, BGR=True)
                        
                        crop = cv2.resize(crop,(32,32),interpolation=cv2.INTER_AREA)
                        #crop = crop / 255.0
                        crop = np.expand_dims(crop, axis=0)
                        print(crop.shape)
                        crop=tf.convert_to_tensor(crop,dtype=tf.float32)
                        crop=tf.cast(crop/255.0,dtype=tf.float32)
                        #cuda = True if torch.cuda.is_available() else False
                        #Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
                        #crop = Variable(crop.type(Tensor))
                        loss = ganomaly.infer_cropimage(crop)
                        loss = int(loss*100)
                        loss = float(loss/100.0)
                        #print(loss)
                        if loss<=LOSS:
                            print('normal-normal-normal-normal-normal-normal-normal {}-- {}'.format(loss,loss))
                        else:
                            print('ab-normal line--ab-normal line--ab-normal line--ab-normal line {}--{}'.format(loss,loss))
                        
                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            if c==0 or c==1:
                                if loss<=LOSS:
                                    annotator.box_label(xyxy, "normal "+str(loss), color=(255,0,0))
                                else:
                                    annotator.box_label(xyxy, "abnormal "+str(loss), color=(0,255,255))
                                
                        

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    #========================================================================================
    parser.add_argument('--isize', type=int, default=32, help='GANomaly input size')
    parser.add_argument('--nz', type=int, default=100, help='GANomaly latent dim')
    parser.add_argument('--nc', type=int, default=3, help='GANomaly num of channels')
    parser.add_argument('--ndf', type=int, default=64, help='GANomaly number of discriminator filters')
    parser.add_argument('--ngf', type=int, default=64, help='GANomaly number of generator filters')
    parser.add_argument('--w_adv', type=int, default=1, help='GANomaly Adversarial loss weight')
    parser.add_argument('--w_con', type=int, default=50, help='GANomaly Reconstruction loss weight')
    parser.add_argument('--w_enc', type=int, default=1, help='GANomaly Encoder loss weight')
    parser.add_argument('--extralayers', type=int, default=0, help='GANomaly extralayers for both G and D')
    parser.add_argument('--encdims', type=str, default=None, help='GANomaly Layer dimensions of the encoder and in reverse of the decoder.')
    parser.add_argument('--dataset', type=str, default=r'/home/ali/GitHub_Code/YOLO/YOLOV5/runs/detect/f_384_2min/crops_1cls', help='dataset dir')
    parser.add_argument('--batch_size', type=int, default=64, help='GANomaly batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='GANomaly learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='GANomaly beta1 for Adam optimizer')
    parser.add_argument('--GANOMALY', action='store_true', help='enable GANomaly')
    parser.add_argument('--LOSS', type=float, default=1.2, help='GANomaly generator loss threshold')
    #========================================================================================
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
