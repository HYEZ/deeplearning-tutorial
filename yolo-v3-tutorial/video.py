import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot as plt 
from yolo3_one_file_to_detect_them_all import *

options = {
 'model': '/mnt/c/Users/Z440_user1/Desktop/dataset/yolo/model.h5',
 'load': '/mnt/c/Users/Z440_user1/Desktop/dataset/yolo//yolov3.weights',
 'threshold': 0.3
    
}


weights_path = '/mnt/c/Users/Z440_user1/Desktop/dataset/yolo/yolov3.weights'
image_path   = './data/zebra.jpg'

# set some parameters
net_h, net_w = 416, 416
obj_thresh, nms_thresh = 0.5, 0.45
anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
          "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
          "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
          "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# make the yolov3 model to predict 80 classes on COCO
yolov3 = make_yolov3_model()

# load the weights trained on COCO into the model
weight_reader = WeightReader(weights_path)
weight_reader.load_weights(yolov3)

model = make_yolov3_model()

cap = cv2.VideoCapture('/mnt/c/Users/Z440_user1/Desktop/dataset/yolo/data/bjj.mp4')

# colors=[tuple(255 * np.random.rand(3)) for i in range(5)]
while(cap.isOpened()):
    (grabbed, frame) = cap.read()

    if not grabbed:
        break

    cv2.imshow('video',frame)

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

    # stime= time.time()
    # ret, frame = cap.read()
    # results = tfnet.return_predict(frame)
    # if ret:
    #     for color, result in zip(colors, results):
    #         tl = (result['topleft']['x'], result['topleft']['y'])
    #         br = (result['bottomright']['x'], result['bottomright']['y'])
    #         label = result['label']
    #         frame= cv2.rectangle(frame, tl, br, color, 7)
    #         frame= cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,0), 2)
    #     cv2.imshow('frame', frame)
    #     print('FPS {:1f}'.format(1/(time.time() -stime)))
    #     if cv2.waitKey(1)  & 0xFF == ord('q'):
    #         break
    # else:
    #     break
cap.release()
cv2.destroyAllWindows()