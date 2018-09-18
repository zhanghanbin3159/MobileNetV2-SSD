import cv2,sys,os
import numpy as np
from imutils.video import FPS
sys.path.append('/media/ziwei/Harddisk02/ziwei/SSD/caffe/python')
import caffe
cap = cv2.VideoCapture('../images/12.avi')

# # V2
net_file= '/media/ziwei/Harddisk02/ziwei/SSD/caffe/examples/MobileNet-SSD-zhb/example/V1_4_sp/MobileNetSSD_deploy.prototxt'
caffe_model='../0910_mobilenetv1_4_87000.caffemodel'

if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.caffemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()

caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net(net_file,caffe_model,caffe.TEST)
#
# CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
#     "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#     "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
#     "sofa", "train", "tvmonitor"]

CLASSES = ['background',
           'person'
           # ,'face'
           ]
fps = FPS().start()
def preprocess(src):
    img = cv2.resize(src, (300, 300))
    img = img - 127.5
    img = img * 0.007843
    return img


def postprocess(img, out):
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0, 0, :, 3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0, 0, :, 1]
    conf = out['detection_out'][0, 0, :, 2]
    print(cls,conf)
    return (box.astype(np.int32), conf, cls)


def detect(imgfile):
    # origimg = cv2.imread(imgfile)
    img = preprocess(imgfile)

    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()
    box, conf, cls = postprocess(imgfile, out)

    # for i in range(len(box)):
    #     if conf[i] < 0.3: continue
    #
    #     p1 = (box[i][0], box[i][1])
    #     p2 = (box[i][2], box[i][3])
    #     cv2.rectangle(imgfile, p1, p2, (0, 255, 0))
    #     p3 = (max(p1[0], 15), max(p1[1], 15))
    #     title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
    #     cv2.putText(imgfile, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
    #
    # cv2.imshow("SSD", imgfile)



while True:
    flag, frame = cap.read()
    if flag is False:
        break
    detect(frame)
    fps.update()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

fps.stop()
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cap.release()
cv2.destroyAllWindows()