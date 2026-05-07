from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
from ObjectDetection import detectObject, displayImage
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.models import model_from_json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import os


main = tkinter.Tk()
main.title("Smart Artificial Intelligence Based Online Proctoring System")
main.geometry("1300x1200")

class_labels = open('model/model-labels').read().strip().split('\n') #reading labels from model
cnn_model = cv2.dnn.readNetFromDarknet('model/model.cfg', 'model/model.weights') #reading model
cnn_layer_names = cnn_model.getLayerNames() #getting layers from cnn model
out_layers = cnn_model.getUnconnectedOutLayers()
try:
    cnn_layer_names = [cnn_layer_names[i[0] - 1] for i in out_layers]
except IndexError:
    cnn_layer_names = [cnn_layer_names[i - 1] for i in out_layers]
label_colors = (0, 0, 255)
emotion_model = None

faceCascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

def is_yolo_weights_valid():
    try:
        cfg_exists = os.path.exists('model/model.cfg')
        weights_exists = os.path.exists('model/model.weights')
        if not cfg_exists or not weights_exists:
            return False
        size = os.path.getsize('model/model.weights')
        return size >= 240_000_000
    except Exception:
        return False


def loadModel():
    global emotion_model
    text.delete('1.0', END)

    model_json_path = 'model/cnnmodel.json'
    model_weights_path = 'model/cnnmodel_weights.h5'
    x_path = 'model/X.txt.npy'
    y_path = 'model/Y.txt.npy'

    if not os.path.exists(model_json_path) or not os.path.exists(model_weights_path):
        text.insert(END, "Pre-trained CNN model not found. Please provide cnnmodel.json and cnnmodel_weights.h5 in model/.\n")
        text.update_idletasks()
        return

    with open(model_json_path, "r") as json_file:
        loaded_model_json = json_file.read()
        emotion_model = model_from_json(loaded_model_json)
    emotion_model.load_weights(model_weights_path)
    text.insert(END, "CNN Model loaded successfully.\n")

    if os.path.exists(x_path) and os.path.exists(y_path):
        X = np.load(x_path)
        Y = np.load(y_path)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        predict = emotion_model.predict(X_test)
        predict = np.argmax(predict, axis=1)
        y_test1 = np.argmax(y_test, axis=1)
        a = accuracy_score(y_test1, predict) * 100
        p = precision_score(y_test1, predict, average='macro') * 100
        r = recall_score(y_test1, predict, average='macro') * 100
        f = f1_score(y_test1, predict, average='macro') * 100
        algorithm = "CNN"
        text.insert(END, algorithm + " Accuracy  :  " + str(a) + "\n")
        text.insert(END, algorithm + " Precision : " + str(p) + "\n")
        text.insert(END, algorithm + " Recall    : " + str(r) + "\n")
        text.insert(END, algorithm + " FScore    : " + str(f) + "\n\n")
    else:
        missing = []
        if not os.path.exists(x_path):
            missing.append('X.txt.npy')
        if not os.path.exists(y_path):
            missing.append('Y.txt.npy')
        missing_text = ', '.join(missing)
        text.insert(END, f"Evaluation data missing: {missing_text}. Model was still loaded successfully.\n")

    text.update_idletasks()

def detectEmotion(image):
    global emotion_model
    if emotion_model is None:
        return 'Neutral'
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
    predict = 4
    for (x, y, w, h) in faces:
        img = image[y:y+h, x:x+w]
        #cv2.imwrite("face.jpg", img)    
        img = cv2.resize(img, (32,32))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,32,32,3)
        img = np.asarray(im2arr)
        img = img.astype('float32')
        img = img/255
        preds = emotion_model.predict(img)
        predict = np.argmax(preds)        
    return labels[predict]

def getPose(image):
    pose = ""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        center_x = (x + x + w) // 2
        center_y = (y + y + h) // 2
        distance_x = center_x - image.shape[1] // 2
        distance_y = center_y - image.shape[0] // 2
        if distance_x > 0:
            pose = "Right"
        elif distance_x < 0:
            pose = "Left"
    return pose        


def webcamVideo():
    if not is_yolo_weights_valid():
        size_text = ''
        try:
            size = os.path.getsize('model/model.weights')
            size_text = f"\nCurrent model/model.weights size: {size} bytes"
        except Exception:
            pass
        messagebox.showerror(
            "YOLO Weights Error",
            "model/model.weights looks incomplete or corrupted.\nPlease replace it with the full YOLOv3 weights file (~248MB)." + size_text
        )
        return

    webcamera = cv2.VideoCapture(0)

    alert_shown = False   # prevents continuous popup spam

    while True:
        (grab, frame) = webcamera.read()
        if not grab:
            break

        frame_height, frame_width = frame.shape[:2]

        try:
            frames, cls, Boundingboxes, confidence_value, class_ids, ids = detectObject(
                cnn_model, cnn_layer_names, frame_height, frame_width,
                frame, label_colors, class_labels)
        except Exception as e:
            messagebox.showerror("Detection Error", f"Object detection failed: {e}")
            break

        person_count = 0

        if ids is not None and len(ids) > 0:
            for i in ids.flatten():
                xx, yy = Boundingboxes[i][0], Boundingboxes[i][1]

                detected_label = class_labels[class_ids[i]]

                # 🔴 Person Detection
                if detected_label == "person":
                    person_count += 1
                    pose = getPose(frame)
                    emotion = detectEmotion(frame)

                    cv2.putText(frames, "Head Pose: " + pose,
                                (xx, yy-20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 0, 0), 2)
                    cv2.putText(frames, "Emotion: " + emotion,
                                (xx, yy-40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 0, 0), 2)

                    # 🚨 Head Movement Alert
                    if pose == "Left" or pose == "Right":
                        cv2.putText(frames,
                                    "ALERT: Head Movement Detected!",
                                    (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0, 0, 255), 3)

                        if not alert_shown:
                            messagebox.showwarning("Proctoring Alert",
                                                   "Head Movement Detected!")
                            alert_shown = True

                    # 🚨 Negative Emotion Alert
                    if emotion in ['Angry', 'Disgusted', 'Fearful', 'Sad']:
                        cv2.putText(frames,
                                    "ALERT: Negative Emotion Detected!",
                                    (50, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0, 0, 255), 3)

                        if not alert_shown:
                            messagebox.showwarning("Proctoring Alert",
                                                   f"Negative Emotion ({emotion}) Detected!")
                            alert_shown = True

                # 🔴 Prohibited Objects
                elif detected_label in ["cell phone", "book", "laptop"]:
                    cv2.putText(frames,
                                "ALERT: Prohibited Object Detected!",
                                (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 255), 3)

                    if not alert_shown:
                        messagebox.showwarning("Proctoring Alert",
                                               f"{detected_label} Detected!")
                        alert_shown = True

        # 🚨 Multiple Persons
        if person_count > 1:
            cv2.putText(frames,
                        "ALERT: Multiple Persons Detected!",
                        (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 3)

            if not alert_shown:
                messagebox.showwarning("Proctoring Alert",
                                       "Multiple Persons Detected!")
                alert_shown = True

        cv2.imshow("Detected Objects", frames)

        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            break

        alert_shown = False   # reset alert for next frame

    webcamera.release()
    cv2.destroyAllWindows()
   
	


def exit():
    main.destroy()

    
font = ('times', 16, 'bold')
title = Label(main, text='Smart artificial intelligence online proctoring with neuro-inspired behavior intelligence')
title.config(bg='light cyan', fg='pale violet red')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Generate & Load CNN Model", command=loadModel)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='light cyan', fg='pale violet red')  
pathlabel.config(font=font1)           
pathlabel.place(x=460,y=100)

webcamButton = Button(main, text="Webcam Based Proctoring System", command=webcamVideo)
webcamButton.place(x=50,y=150)
webcamButton.config(font=font1) 

exitButton = Button(main, text="Exit", command=exit)
exitButton.place(x=460,y=150)
exitButton.config(font=font1) 


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='snow3')
main.mainloop()
