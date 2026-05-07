import numpy as np
import cv2 as cv
import time
import os


def get_yolo_input_size(cfg_path='model/model.cfg'):
    width = 416
    height = 416
    try:
        with open(cfg_path, 'r') as cfg:
            for line in cfg:
                stripped = line.strip()
                if stripped.startswith('width='):
                    width = int(stripped.split('=', 1)[1].strip())
                elif stripped.startswith('height='):
                    height = int(stripped.split('=', 1)[1].strip())
    except Exception:
        pass
    return width, height

def detectObject(CNNnet, total_layer_names, image_height, image_width, image, name_colors, class_labels,  
            Boundingboxes=None, confidence_value=None, class_ids=None, ids=None, detect=True):
    
    if detect:
        width, height = get_yolo_input_size()
        blob_object = cv.dnn.blobFromImage(image, 1/255.0, (width, height), swapRB=True,crop=False)
        CNNnet.setInput(blob_object)
        cnn_outs_layer = CNNnet.forward(total_layer_names)
        Boundingboxes, confidence_value, class_ids = listBoundingBoxes(cnn_outs_layer, image_height, image_width, 0.5)
        if not Boundingboxes or not confidence_value or class_ids is None:
            ids = None
        else:
            ids = cv.dnn.NMSBoxes(Boundingboxes, confidence_value, 0.5, 0.3)
        image, cls = labelsBoundingBoxes(image, Boundingboxes, confidence_value, class_ids, ids, name_colors, class_labels)

    return image, cls, Boundingboxes, confidence_value, class_ids, ids


def labelsBoundingBoxes(image, Boundingbox, conf_thr, classID, ids, color_names, predicted_labels):
    cls = []
    if ids is None or len(ids) == 0:
        return image, cls
    for i in ids.flatten():
        # draw boxes
        xx, yy = Boundingbox[i][0], Boundingbox[i][1]
        width, height = Boundingbox[i][2], Boundingbox[i][3]
        class_color = (255, 0, 0)

        cv.rectangle(image, (xx, yy), (xx+width, yy+height), class_color, 2)
        text_label = "{}: {:4f}".format(predicted_labels[classID[i]], conf_thr[i])
        cv.putText(image, text_label, (xx, yy-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, class_color, 2)
        cls.append(predicted_labels[classID[i]])
    return image, cls


def listBoundingBoxes(image, image_height, image_width, threshold_conf):
    box_array = []
    confidence_array = []
    class_ids_array = []

    for img in image:
        for obj_detection in img:
            objectness = float(obj_detection[4])
            class_scores = obj_detection[5:]
            class_id = np.argmax(class_scores)
            class_confidence = float(class_scores[class_id])
            confidence_value = objectness * class_confidence
            if confidence_value > threshold_conf:
                Boundbox = obj_detection[0:4] * np.array([image_width, image_height, image_width, image_height])
                center_X, center_Y, box_width, box_height = Boundbox.astype('int')

                xx = int(center_X - (box_width / 2))
                yy = int(center_Y - (box_height / 2))

                box_array.append([xx, yy, int(box_width), int(box_height)])
                confidence_array.append(float(confidence_value))
                class_ids_array.append(class_id)

    return box_array, confidence_array, class_ids_array

def displayImage(image):
    cv.imshow("Final Image", image)
    cv.waitKey(0)


