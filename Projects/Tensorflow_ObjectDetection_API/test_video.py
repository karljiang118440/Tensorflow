import numpy as np
import tensorflow as tf
import cv2 as cv
import time
 
# Read the graph.
with tf.gfile.FastGFile('frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
 
with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
 
    # Read and preprocess an image.
    # video 
    capture = cv.VideoCapture("test.mp4")
    fps = 0.0 
    while(True):
        t1 = time.time()
        ret,img = capture.read()
        
    #img = cv.imread('example.jpg')
    inp = cv.resize(img, (300, 300))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
 
    # Run the model
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
 
    # Visualize detected bounding boxes.
    num_detections = int(out[0][0])
    for i in range(num_detections):
        classId = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]
        if score > 0.3:
            x = bbox[1] * img.shape[1]
            y = bbox[0] * img.shape[0]
            right = bbox[3] * img.shape[1]
            bottom = bbox[2] * img.shape[0]
            cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (255, 255, 0), thickness=1)
    
    fps = ( fps + (1./(time.time() - t1)) ) / 2
    print("fps = %.2f" %(fps))

cv.imshow('TensorFlow MobileNet-SSD', img)
cv.waitKey()