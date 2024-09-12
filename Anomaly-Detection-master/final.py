#import pandas as pd 
from datetime import timedelta
import os
import cv2
import numpy as np
from numpy import expand_dims, zeros, ones, vstack
from numpy.random import randn, randint
from keras.datasets.mnist import load_data
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, Input, UpSampling2D
#from matplotlib import pyplot
from keras.utils.vis_utils import plot_model
import cv2
import tensorflow as tf
#from keras.layers import LeakyReLU
from keras import backend as K
from keras import initializers
#from tqdm import tqdm
import time
from datetime import timedelta
import torch
import torchvision.transforms as transforms

#import yolo
'''
def start(video_path):
    cap = cv2.VideoCapture(video_path)
    count=0
    while True:
        ret, frame = cap.read()
        if count%2 == 0:
            if video_path == 0:
                frame=cv2.resize(frame, (224,224))
                gan(frame)
                cv2.imshow("Image", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if ret == True:
                    gan(frame)
                else:
                    break
        count+=1
    cap.release()


def preprocessing(frame):
    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hist = cv2.calcHist(frame2,[0,1,2],None,[8,8,8],[0,256,0,256,0,256])
    hist = cv2.normalize(hist,None).flatten()
    index = hist

    OPENCV_METHODS = (
	(cv2.HISTCMP_CORREL ),
	(cv2.HISTCMP_CHISQR),
	(cv2.HISTCMP_INTERSECT), 
	(cv2.HISTCMP_BHATTACHARYYA))

    for method in OPENCV_METHODS:       
        results = {}
        reverse = False        
        if method in (cv2.HISTCMP_CORREL, cv2.HISTCMP_INTERSECT ):
            reverse = True
    
    d = cv2.compareHist(index, hist, cv2.HISTCMP_INTERSECT)
    results = d
    mean__ = np.mean(index, dtype=np.float64)
    variance = np.var(index, dtype=np.float64)
    #print("variance", variance)
    standard_deviation = np.sqrt(variance)
    th = mean__ + standard_deviation + 3
    #print("threshold value", th)

    cap.set(cv2.CAP_PROP_POS_FRAMES,0)

    os.makedirs('keyframes')
    cframe1=0
    for (k,hist) in index.items():
            d = cv2.compareHist(index[k], hist, cv2.HISTCMP_INTERSECT)
            ret, keyframe = cap.read()
            print(d)
            if not ret:
                break
            if (d > th):
                    name = './keyframes/' + str(cframe1) + '.jpg'
                    print("creating" +name)
                    cv2.imwrite(name, keyframe )
                    cframe1+=1
'''

def sum_of_residual(y_true, y_pred):
    #print("in sum of residual")
    return tf.reduce_sum(abs(y_true - y_pred))

def feature_extractor(d):
    #print("in feature extractor")
    intermediate_model = Model(inputs=d.layers[0].input, outputs=d.layers[-5].output)
    intermediate_model.compile(loss='binary_crossentropy', optimizer='adam')
    return intermediate_model

d_model = tf.keras.models.load_model("models/discriminator_model.h5", custom_objects={'sum_of_residual': sum_of_residual}) 

def compute_anomaly_score(model, x):   
    #print("computing anomaly score")
    
    z = np.random.uniform(0, 1, size=(1, 100))
    intermediate_model = feature_extractor(d_model)
    d_x = intermediate_model.predict(x)
    loss = model.fit(z, [x, d_x], epochs=50, verbose=0)
    similar_data, _ = model.predict(z)
    return loss.history['loss'][-1], similar_data

ano_model = tf.keras.models.load_model("models/anogan.h5", custom_objects={'sum_of_residual': sum_of_residual})
#print("ano model loaded")

def gan(frame,cframe):
    #print("in gan")
    #ano_model = tf.keras.models.load_model("gan/anogan.h5", custom_objects={'sum_of_residual': sum_of_residual})
    #g_model = tf.keras.models.load_model("gan/g_model.h5", custom_objects={'sum_of_residual': sum_of_residual})
    #int_model = tf.keras.models.load_model("gan/int_model.h5", custom_objects={'sum_of_residual': sum_of_residual})
    test_img = frame.astype(np.float32)/255.
    #test_img = np.array(test_img)
    ano_score, similar_img = compute_anomaly_score(ano_model, test_img.reshape(1, 224, 224, 1))
    print(ano_score)
    #image = np.dstack([frame]*3)
    evaluate(ano_score,cframe)
    return

def evaluate(ano_score,cframe):
    #print("in evaluate")
    if ano_score > 3000:
        print("in yolo")
        yolo_res = eval_yolo(cframe)
        print("YOLO:",yolo_res)
        fighting_res = eval_fight(cframe)
        print("FIGHT:",fighting_res)

        #if false, then sus
        return 
    else:
        return 

net = cv2.dnn.readNet("models/yolov4-obj_last_2.weights", "models/yolov4-obj.cfg")
classes = []
with open("models/obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layers_names = net.getLayerNames()
#output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()] 

#print("reading yolo net complete")
counter = 0
nocounter = 0
prev_label = ''
#print("glob vars assigned")

def eval_yolo(frame):
    #print("in eval yolo")
    output_layers = [layers_names[i-1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    height, width, channels = frame.shape

    global counter, nocounter, prev_label

    boxes = []
    confidences = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(conf))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    label=''

    for i in indexes:
       # i = i[0]
        #label2 = '%.2f' % (confidences[i]*100)
            
        # Get the label for the class name and its confidence
        if classes:
            assert(class_ids[i] < len(classes))
            label = str(classes[class_ids[i]])

    print(label, counter)
    if prev_label!=label:
        nocounter+=1	
        if nocounter==3:
            prev_label=label
            nocounter=0
            counter=0
    else:
        if label=='Gun' or label == 'Rifle':
            cv2.imwrite(os.path.join('app/static/anoframes','Weapon_'+str(counter)+'.jpg'), img=frame)
            counter+=1
            return True
        elif label=='Fire':
            cv2.imwrite(os.path.join('app/static/anoframes','Fire_'+str(counter)+'.jpg'), img=frame)
            counter+=1
            return True
        else:
            counter+=1
            return False

fight_model = torch.load('models/alexnet2.pth')
transform = transforms.ToTensor()

def eval_fight(image):
    global counter, nocounter, prev_label
    input = transform(image)
    # unsqueeze batch dimension, in case you are dealing with a single image
    input = input.unsqueeze(0)
    idx_to_class = {
        0: 'Fight',
        1: 'No Fight'}
    # Set model to eval
    fight_model.eval()
    output = fight_model(input)
    print(output)
    pred = torch.argmax(output, 1)
    # Get prediction
    for p in pred:
        label = idx_to_class[p.item()]
    print(label)
    if label=='Fight':
        cv2.imwrite(os.path.join('app/static/anoframes','Fight_'+str(counter)+'.jpg'), img=image)
        return True
    else: 
        return False

if __name__ == '__main__':

    video_path = 'testdata/fire1.mp4'

    cap = cv2.VideoCapture(video_path)
    count=0
    
    start_time = time.monotonic()
    

    while True:
        ret, frame = cap.read()
        if count%2 == 0:
            if video_path == 0:
                gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gframe = cv2.resize(gframe, dsize=(224, 224),interpolation=cv2.INTER_CUBIC)
                #cframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #print(frame.shape)
                gan(gframe,frame)
                cv2.imshow("Image", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if ret == True:
                    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gframe = cv2.resize(gframe, dsize=(224, 224),interpolation=cv2.INTER_CUBIC)
                    #cframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    #print(frame.shape)
                    gan(gframe,frame)
                    #cv2.imshow("Image", frame)
                else:
                    break
        count+=1
    end_time = time.monotonic()
    print(timedelta(seconds = end_time-start_time))
    cap.release()

cv2.destroyAllWindows()