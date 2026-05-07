import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import os

print('CHECK START')
print('Python version:', __import__('sys').version)
print('OpenCV version:', cv2.__version__)
try:
    net = cv2.dnn.readNetFromDarknet('model/model.cfg', 'model/model.weights')
    print('YOLO loaded')
    ln = net.getLayerNames()
    out = net.getUnconnectedOutLayers()
    print('out layers:', out)
    try:
        names = [ln[i[0] - 1] for i in out]
    except Exception:
        names = [ln[i - 1] for i in out]
    print('names len', len(names))
    blob = cv2.dnn.blobFromImage(np.zeros((608,608,3), np.uint8), 1/255.0, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(names)
    print('outs shapes', [o.shape for o in outs])
except Exception as e:
    print('YOLO ERROR', type(e).__name__, e)

try:
    if os.path.exists('model/cnnmodel.json'):
        with open('model/cnnmodel.json', 'r') as f:
            model = model_from_json(f.read())
        model.load_weights('model/cnnmodel_weights.h5')
        print('Emotion model loaded')
    else:
        print('Emotion model JSON missing')
except Exception as e:
    print('Emotion ERROR', type(e).__name__, e)
