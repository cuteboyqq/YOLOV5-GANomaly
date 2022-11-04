#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 18:09:55 2022

@author: ali
"""

import platform
import subprocess
import warnings
from pathlib import Path
import cv2
import numpy as np
import os

import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image


class GANomaly_Detect():
    def __init__(self,
                 model_dir=r'/home/ali/Desktop/GANomaly-tf2/export_model',
                 model_file=r'G-uint8-20221104_edgetpu.tflite',
                 save_image=False,
                 show_log=False):
        super().__init__()
        self.model_dir = model_dir
        self.tflite = False
        self.edgetpu = True
        self.model_file = model_file
        self.model_path = os.path.join(self.model_dir, model_file)
        self.interpreter = self.get_interpreter()
        self.save_image = save_image
        self.show_log = show_log
        
    def Pycoral_Edgetpu(self):
        # Specify the TensorFlow model, labels, and image
        #script_dir = pathlib.Path(__file__).parent.absolute()
        #model_file = os.path.join(self.model_dir, 'G-uint8-20221104_edgetpu.tflite')
        
        # Initialize the TF interpreter
        print('Start interpreter')
        self.interpreter = edgetpu.make_interpreter(self.model_path)
        print('End interpreter')
    
        print('Start allocate_tensors')
        self.interpreter.allocate_tensors()
        print('End allocate_tensors')
    
        self.input_details = self.interpreter.get_input_details()  # inputs
        self.output_details = self.interpreter.get_output_details()  # outputs 
        print('input details : \n{}'.format(self.input_details))
        print('output details : \n{}'.format(self.output_details))
        # Resize the image
        #size = common.input_size(interpreter)
        #image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)
    
        # Run an inference
        #common.set_input(interpreter, image)
        #interpreter.invoke()
        #classes = classify.get_classes(interpreter, top_k=1)
    
        # Print the result
        #labels = dataset.read_label_file(label_file)
        #for c in classes:
          #print('%s: %.5f' % (labels.get(c.id, c.id), c.score))
    def get_interpreter(self):
        if self.tflite or self.edgetpu:# https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
            try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                from tflite_runtime.interpreter import Interpreter, load_delegate
                print('try successful')
            except ImportError:
                print('ImportError')
                import tensorflow as tf
                self.Interpreter, self.load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,
            if self.edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                print(f'Loading {self.model_path} for TensorFlow Lite Edge TPU inference...')
                
                self.delegate = {
                    'Linux': 'libedgetpu.so.1',
                    'Darwin': 'libedgetpu.1.dylib',
                    'Windows': 'edgetpu.dll'}[platform.system()]
                self.interpreter = self.Interpreter(model_path=self.model_path, experimental_delegates=[self.load_delegate(self.delegate)])
                
                
                # Initialize the TF interpreter
                #print('Start interpreter')
                #interpreter = edgetpu.make_interpreter(w)
                #print('End interpreter')
                
            else:  # TFLite
                print(f'Loading {self.model_path} for TensorFlow Lite inference...')
                self.interpreter = self.Interpreter(model_path=self.model_path)  # load TFLite model
            self.interpreter.allocate_tensors()  # allocate
            self.input_details = self.interpreter.get_input_details()  # inputs
            self.output_details = self.interpreter.get_output_details()  # outputs
            print('input details : \n{}'.format(self.input_details))
            print('output details : \n{}'.format(self.output_details))
        return self.interpreter
    
    def detect_image(self, w, im, cnt=1):
        #SHOW_LOG=False
        self.INFER=False
        self.ONLY_DETECT_ONE_IMAGE=True
        if self.interpreter is None:
            print('interpreter is None, get interpreter now')
            interpreter = self.get_interpreter(w,self.tflite,self.edgetpu)
            self.interpreter.allocate_tensors()  # allocate
            self.input_details = self.interpreter.get_input_details()  # inputs
            self.output_details = self.interpreter.get_output_details()  # outputs 
            #print('input details : \n{}'.format(input_details))
            #print('output details : \n{}'.format(output_details))
        self.input_details = self.interpreter.get_input_details()  # inputs
        self.output_details = self.interpreter.get_output_details()  # outputs 
       
        #import tensorflow as tf
        #from PIL import Image
        #from matplotlib import pyplot as plt
        # Lite or Edge TPU
        os.makedirs('./runs/detect/ori_images',exist_ok=True)
        os.makedirs('./runs/detect/gen_images',exist_ok=True)
        if self.INFER:
            self.input_img = im
            #im = tf.transpose(im, perm=[0,1,2,3])
            #im = tf.squeeze(im)
            #plt.imshow(im)
            #plt.show()
        elif self.ONLY_DETECT_ONE_IMAGE:
            im = cv2.imread(im)
            im = cv2.resize(im, (64, 64))
            #input_img = im
            if self.save_image:
            #cv2.imshow('ori_image',im)
                filename = 'ori_image_' + str(cnt) + '.jpg'
                file_path = os.path.join('./runs/detect/ori_images', filename)
                cv2.imwrite(file_path,im)
                #cv2.waitKey(10)
            
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        #im = im/255.0
        #im = (im).astype('int32')
        #image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
        #img = img[np.newaxis, ...].astype(np.float32)
        #print("calibration image {}".format(img[i]))
        #img = img / 255.0
        
        #im = Image.fromarray((im * 255).astype('uint8'))
        im = im[np.newaxis, ...].astype(np.float32)
        if self.show_log:
            print('im : {}'.format(im.shape))
        self.input_img = im
        im = im/255.0
        #im = tf.expand_dims(im, axis=0)
        #im = im.numpy()
        
        #print('im:{}'.format(im.shape))
        #print('im: {}'.format(im))
        input = self.input_details[0]
        self.int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model (np.uint8)
        #int32 = input['dtype'] == np.int32  # is TFLite quantized uint8 model (np.uint8)
        #print('input[dtype] : {}'.format(input['dtype']))
        if self.int8:
            #print('is TFLite quantized uint8 model')
            self.scale, self.zero_point = input['quantization']
            im = (im / self.scale + self.zero_point).astype(np.uint8)  # de-scale
            #im = im.astype(np.uint8)
            if self.show_log:
                print('after de-scale {}'.format(im))
        interpreter.set_tensor(input['index'], im)
        interpreter.invoke()
        y = []
        self.gen_img = None
        for output in self.output_details:
            x = self.interpreter.get_tensor(output['index'])
            #print(x.shape)
            #print(x)
            if x.shape[1]==64:
                #print('get out images')
                
                self.scale, self.zero_point = output['quantization']
                
                x = (x.astype(np.float32)-self.zero_point) * self.scale  # re-scale
                #x = x.astype(np.float32)
                #x = tf.squeeze(x)
                #x = x.numpy()
                self.gen_img = x*255
                self.gen_img = np.squeeze(self.gen_img)
                #print('after squeeze & numpy x : {}'.format(x))
                if self.save_image:
                    #cv2.imshow('out_image',gen_img)
                    filename = 'out_image_' + str(cnt) + '.jpg'
                    file_path = os.path.join('./runs/detect/gen_images/',filename)
                    cv2.imwrite(file_path,self.gen_img)
                    #cv2.waitKey(10)
                #gen_img = renormalize(gen_img)
                #gen_img = tf.transpose(gen_img, perm=[0,1,2])
                #plt.imshow(gen_img)
                #plt.show()
            if self.int8:
                self.scale, self.zero_point = output['quantization']
                x = (x.astype(np.float32)-self.zero_point) * self.scale  # re-scale
                #x = x.astype(np.float32)
                #gen_img = tf.squeeze(gen_img)
                #gen_img = gen_img.numpy()
            y.append(x)
        y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
        #gen_img = y[0]
        
        if self.show_log:
            print('input image : {}'.format(self.input_img))
            print('input image : {}'.format(self.input_img.shape))
            print('gen_img : {}'.format(self.gen_img))
            print('gen_img : {}'.format(self.gen_img.shape))
        self.latent_i = y[1]
        self.latent_o = y[2]
        if self.show_log:
            print('latent_i : {}'.format(self.latent_i))
            print('latent_o : {}'.format(self.latent_o))
        self._g_loss = self.g_loss(self.input_img/255.0, self.gen_img/255.0, self.latent_i, self.latent_o)
        #_g_loss = 888
        if self.show_log:
            print('g_loss : {}'.format(self._g_loss))
        #print(y)
        return self._g_loss, self.gen_img
    
    def g_loss(self):
        
        def l1_loss(A,B):
            return np.mean((abs(A-B)).flatten())
        def l2_loss(A,B):
            return np.mean(np.sqrt((A-B)*(A-B)).flatten())
        # tf loss
        #l2_loss = tf.keras.losses.MeanSquaredError()
        #l1_loss = tf.keras.losses. MeanAbsoluteError()
        #bce_loss = tf.keras.losses.BinaryCrossentropy()
        
        # adversarial loss (use feature matching)
        #l_adv = l2_loss
        # contextual loss
        self.l_con = l1_loss
        # Encoder loss
        self.l_enc = l2_loss
        # discriminator loss
        #l_bce = bce_loss
        
        #err_g_adv = l_adv(feat_real, feat_fake)
        self.err_g_con = self.l_con(self.input_img, self.gen_img)
        #err_g_enc = l_enc(latent_i, latent_o)
        self.err_g_enc = self.l_enc(self.latent_i,self.latent_o)
        self.g_loss = self.err_g_con * 50 + \
                 self.err_g_enc * 1
        return self.g_loss
    
    def plot_loss_distribution(self, SHOW_MAX_NUM,positive_loss,defeat_loss):
        # Importing packages
        import matplotlib.pyplot as plt2
        # Define data values
        x = [i for i in range(SHOW_MAX_NUM)]
        y = positive_loss
        z = defeat_loss
        print(x)
        print('positive_loss len: {}'.format(len(positive_loss)))
        print('defeat_loss len: {}'.format(len(defeat_loss)))
        #print(positive_loss)
        #print(defeat_loss)
        # Plot a simple line chart
        #plt2.plot(x, y)
        # Plot another line on the same chart/graph
        #plt2.plot(x, z)
        plt2.scatter(x,y,s=1)
        plt2.scatter(x,z,s=1) 
        os.makedirs('./runs/detect',exist_ok=True)
        file_path = os.path.join('./runs/detect','loss_distribution_100.jpg')
        plt2.savefig(file_path)
        plt2.show()
    
    def infer_python(self, img_dir,SHOW_MAX_NUM):
        import glob
        self.image_list = glob.glob(os.path.join(img_dir,'*.jpg'))
        self.loss_list = []
        self.cnt = 0
        for image_path in self.image_list:
            print(image_path)
            self.cnt+=1
            
            if self.cnt<=SHOW_MAX_NUM:
                self.loss,self.gen_img = self.detect_image(self.model_path, image_path, interpreter=self.interpreter, 
                                                           tflite=self.tflite,edgetpu=self.edgetpu, 
                                                           save_image=self.save_image, cnt=self.cnt)
                print('{} loss: {}'.format(self.cnt,self.loss))
                self.loss_list.append(self.loss)
        
        return self.loss_list
'''
if __name__=="__main__":
    PYCORAL = False
    DETECT = False
    DETECT_IMAGE = False
    INFER = True
    if DETECT:
        w=r'/home/ali/Desktop/GANomaly-tf2/export_model/G-uint8-20221104_edgetpu.tflite'
        #w=r'/home/ali/GitHub_Code/cuteboyqq/GANomaly/GANomaly-tf2/export_model/G-uint8-new.tflite'
        get_interpreter(w,tflite=False,edgetpu=True)
    if PYCORAL:
        Pycoral_Edgetpu()
        
    if DETECT_IMAGE:
        save_image = True
        im = r'/home/ali/Desktop/factory_data/crops_1cls/line/ori_video_ver2121.jpg'
        #im = r'/home/ali/Desktop/factory_data/crops_2cls_small/noline/ori_video_ver244.jpg'
        #w=r'/home/ali/GitHub_Code/cuteboyqq/GANomaly/GANomaly-tf2/export_model/G-uint8-new_edgetpu.tflite'
        #w=r'/home/ali/Desktop/GANomaly-tf2/export_model/G-uint8-20221104.tflite'
        w = r'/home/ali/Desktop/GANomaly-tf2/export_model/G-uint8-20221104_edgetpu.tflite'
        loss, gen_image = detect_image(w, im, tflite=False,edgetpu=True, save_image=True)
        
        
    if INFER:
        #import tensorflow as tf
        test_data_dir = r'/home/ali/Desktop/factory_data/crops_2cls_small/line'
        abnormal_test_data_dir = r'/home/ali/Desktop/factory_data/crops_2cls_small/noline'
        (img_height, img_width) = (64,64)
        batch_size_ = 1
        shuffle = False
        SHOW_MAX_NUM = 10
        save_image=True
        w = r'/home/ali/Desktop/GANomaly-tf2/export_model/G-uint8-20221104_edgetpu.tflite'
        interpreter = get_interpreter(w,tflite=False,edgetpu=True)
        line_loss = infer_python(test_data_dir,interpreter,SHOW_MAX_NUM,save_image=save_image)
        #noline_loss = infer_python(abnormal_test_data_dir,interpreter,SHOW_MAX_NUM)
        #plot_loss_distribution(SHOW_MAX_NUM,line_loss,noline_loss)
        #=================================================
        #if plt have QT error try
        #pip uninstall opencv-python
        #pip install opencv-python-headless
        #=================================================
'''