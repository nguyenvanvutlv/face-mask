# --------------------------------------------- #
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from insightface.app import FaceAnalysis
# --------------------------------------------- #



def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)



# --------------------------------------------- #
# model for face detection
app = FaceAnalysis(name='buffalo_sc')
app.prepare(ctx_id= 0, det_size=(640, 640))
# model for mask detection
"""
    MobileNetV2
        input shape (1, 224, 224, 3)
        
        output  [[ mask  withoutmask ]]    
"""
maskNet = load_model("mask_detector.model")
# --------------------------------------------- #


# --------------------------------------------- #
# read frame from camera, id = 0
cam = cv2.VideoCapture(0)
while True:
    
    _, frame = cam.read()
    
    # flip face from camera
    frame = cv2.flip(frame, 1)
    
    # detect face
    faces = app.get(frame)
    (h, w) = frame.shape[:2]
    for face in faces:
        frame_crop = frame.copy()
        
        # bounding box where there is a face
        """
            bounding box (startX, startY, endX, endY)
        """
        bbox = face['bbox'].astype(np.int64)
        (bbox[1], bbox[3]) = (max(0, bbox[1]), max(0, bbox[3]))
        (bbox[0], bbox[2]) = (min(w - 1, bbox[0]), min(h - 1, bbox[2]))
        
        
        # crop image from bounding box
        frame_crop = frame_crop[bbox[1]: bbox[3], bbox[0]: bbox[2]]
        
        if frame_crop.any():
            # resize image because input shape MobileNetV2 is (1, 224, 224, 3)
            frame_crop = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2RGB)
            frame_crop = cv2.resize(frame_crop, (224, 224))
            frame_crop = img_to_array(frame_crop)
            frame_crop = preprocess_input(frame_crop)
            
            frame_crop = np.array(frame_crop, dtype= "float32")
            
            # current size image is (224, 224, 3)
            
            # expand dims image (224, 224, 3) --> (1, 224, 224, 3)
            frame_crop = np.expand_dims(frame_crop, axis=0)
            
            # detect mask
            pred = maskNet.predict(frame_crop, batch_size = 32)
            mask, withoutmask = pred[0]
            print(pred)
            label = "Mask" if mask > withoutmask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            
            # draw image and put text
            draw_border(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2, 10, 10)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        
    cv2.imshow("cam", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
# --------------------------------------------- #



cam.release()
cv2.destroyAllWindows()