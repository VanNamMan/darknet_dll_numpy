import cv2
import numpy as np
import threading,time
from PyQt5.QtCore import pyqtSignal,QObject

class YoloOpenCv(QObject):
    detectionResult = pyqtSignal(object)
    videocaptureError = pyqtSignal() 
    def __init__(self,cfg_file,weights_file,names_file):
        super(YoloOpenCv,self).__init__()
        self.net = cv2.dnn_DetectionModel(cfg_file,weights_file)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.net.setInputSize(416, 416)
        self.net.setInputScale(1.0 / 255)
        self.net.setInputSwapRB(True)

        self.classes = []
        with open(names_file, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        np.random.seed(42)
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        self.bloop_detection = False

    def detect(self,mat,conf_threshold=0.4,nms_threshold=0.3):
        '''
        :return : classes, confidences, boxes
        '''
        classes, confidences, boxes = self.net.detect(mat, confThreshold=conf_threshold, nmsThreshold=nms_threshold)
        if not len(classes):
            return None
        classes = classes.reshape((len(classes),))
        confidences = confidences.reshape((len(confidences),))
        return classes, confidences, boxes

    def detect_realtime(self,cap,conf_threshold,nms_threshold,dont_show=True):
        self.bloop_detection = True
        thread = threading.Thread(target=self.loop_detect_realtime,args=(cap,conf_threshold,nms_threshold,dont_show))
        thread.start()
        pass

    def loop_detect_realtime(self,cap,conf_threshold,nms_threshold,dont_show=True):
        '''
        :cap : cv2.VideoCapture
        '''
        while self.bloop_detection:
            ret,mat = cap.read()
            if not ret:
                self.videocaptureError.emit()
                break
            pred = self.detect(mat,conf_threshold,nms_threshold)
            self.detectionResult.emit(pred)
            if not dont_show:
                mat = yolo.draw_pred(mat,pred)
                cv2.imshow("",mat)
                cv2.waitKey(1)
            else:
                time.sleep(0.001)
        pass

    def draw_pred(self,mat,pred):
        '''
        :pred: detection output
        '''
        if pred is None:
            return mat
        classes,confidences,boxes = pred
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            label = str(self.classes[classes[i]])
            color = self.colors[classes[i]]
            cv2.rectangle(mat, (x, y), (x + w, y + h), color,1)
            cv2.putText(mat, label + " " + str(round(confidences[i], 2)), (x, y-5), self.font,1, color,1)
        
        return mat

if __name__ == "__main__":
    yolo = YoloOpenCv("cfg/yolov3-tiny-custom.cfg","weights/yolov3-tiny-custom_5000.weights","data/obj.names")
    # mat = cv2.imread("data/1.jpg")
    # print(mat.shape)
    # pred = yolo.detect(mat,0.4,0.3)
    # mat = yolo.draw_pred(mat,pred)
    # cv2.imwrite("output.jpg",mat)
    cap = cv2.VideoCapture("data/bienso.mp4")
    yolo.detect_realtime(cap,0.2,0.3,False)
    pass