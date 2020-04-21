from dog_breed_web_app import app
import json, plotly
from flask import render_template , request
import cv2
import numpy as np
import sys


import keras
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50, preprocess_input,decode_predictions
from keras import regularizers
import tensorflow as tf
import os




face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_alt.xml')


def face_detector(img_path):
    """
    detects whether a human face is in an image.
    :param img_path: string the path of the image we want to classify
    :return:
    bool: True if we detect a human face and false if we can't detect a face
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def ResNet50_predict_labels(img_path):
    """
    predict class label of an image using pre-trained resnet50

    :param img_path: string. image path we want to classify
    :return:
    int. predicted class index
    """
    # returns prediction vector for image located at img_path
    keras.backend.clear_session()
    graph = tf.get_default_graph()
    with graph.as_default():

        ResNet50_model_dogs = ResNet50(weights="imagenet")
        #ResNet50_model_dogs._make_predict_function()
        img = preprocess_input(path_to_tensor(img_path))
        return np.argmax(ResNet50_model_dogs.predict(img))

def extract_Resnet50(tensor):
   """
   extract bottle neck features from resnet50
   :param tensor: tensor. a tensor containing the image or images to get their bottle neck features of
   :return:
   tensor. bottleneck features of the tensor passed
   """
   from keras.applications.resnet50 import ResNet50, preprocess_input
   return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))


def dog_detector(img_path):
   """
   Detect whether a dog is in a given image or not
   :param img_path: string. path containing the image we want to classify
   :return:
   bool. True of a dog was detected and false if a dog was not detected
   """
   prediction = ResNet50_predict_labels(img_path)
   return ((prediction <= 268) & (prediction >= 151))

def path_to_tensor(img_path):
    """
    Reads in an image and preprocess it to be fed to a nueral network

    :param img_path: string. image path to read and process
    :return:
    numpy array. the read and processed image
    """

    # loads RGB image as PIL.Image.Image type
    img = cv2.imread(img_path)#image.load_img(img_path, target_size=(224, 224))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img ,(224,224))

    img = img.astype("float64")

    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(img, axis=0)

def resnet_predict_breed(img_path):
   """
   predict which dog breed in an image
   :param img_path: string containing the image path
   :return:
   string.  dog breed name
   """

   #reset graph
   keras.backend.clear_session()
   #define model architecture
   resnet50_model = Sequential()
   resnet50_model.add(Flatten(input_shape=(1,1,2048)))
   resnet50_model.add(Dropout(0.3))
   resnet50_model.add(Dense(256, activation='relu'))
   resnet50_model.add(Dropout(0.4))
   resnet50_model.add(Dense(133, activation='softmax'))
   #load weights to the model
   resnet50_model.load_weights('../saved_models/weights.best.resnet50.hdf5')

    # extract bottleneck features
   bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
   # obtain predicted vector
   predicted_vector = resnet50_model.predict(bottleneck_feature)
   # return dog breed that is predicted by the model
   dog_names = np.load("dog_names.npy").tolist()

   return dog_names[np.argmax(predicted_vector)]


def predict_breed(image_path):
   """
   predicts breed of a dog if a dog is present or the dog breed a human resembels if a human is present
   and throws and error message if neither is detected
   :param image_path:string.  image path of the image to process
   :return:
   string.  output message
   """
   if face_detector(image_path):
       return ("This human resembels a {} ".format(resnet_predict_breed(image_path)))
   elif dog_detector(image_path):
       return ("This dog is a {} ".format(resnet_predict_breed( image_path)))
   else:
       return ("we couldn't find a dog or a human in this picture")


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/upload',methods=["POST"])
def upload():
    file = request.files["inputFile"]

    img_path = os.path.join(os.path.abspath(os.getcwd()), 'dog_breed_web_app/static/img',file.filename)

    file.save(img_path)
    str = predict_breed(img_path)
    
    return render_template('upload.html', dog_breed = str, img_path="static/img/"+file.filename)
