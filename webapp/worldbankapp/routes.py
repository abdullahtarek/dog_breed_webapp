from worldbankapp import app
import json, plotly
from flask import render_template , request
from wrangling_scripts.wrangle_data import return_figures
import cv2
import numpy as np

import keras

#from keras.applications.resnet50 import preprocess_input, decode_predictions
#from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
#from keras.layers import Dropout, Flatten, Dense
#from keras.models import Sequential
#from keras import regularizers


#face_cascade = cv2.CascadeClassifier('../../haarcascades/haarcascade_frontalface_alt.xml')


#def face_detector(img_path):
#    img = cv2.imread(img_path)
#    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    faces = face_cascade.detectMultiScale(gray)
#    return len(faces) > 0

#def ResNet50_predict_labels(img_path):
#    # returns prediction vector for image located at img_path
#    img = preprocess_input(path_to_tensor(img_path))
#    return np.argmax(ResNet50_model.predict(img))

#def dog_detector(img_path):
#    prediction = ResNet50_predict_labels(img_path)
#    return ((prediction <= 268) & (prediction >= 151)) 

#resnet50_model = Sequential()
#resnet50_model.add(Flatten(input_shape=train_reset50.shape[1:]))
#resnet50_model.add(Dropout(0.3))
#resnet50_model.add(Dense(256, activation='relu'))
#resnet50_model.add(Dropout(0.4))
#resnet50_model.add(Dense(133, activation='softmax'))

#resnet50_model.load_weights('../../saved_models/weights.best.resnet50.hdf5')

#def resnet_predict_breed(img_path):
#    # extract bottleneck features
#    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
#    # obtain predicted vector
#    predicted_vector = resnet50_model.predict(bottleneck_feature)
#    # return dog breed that is predicted by the model
#    return dog_names[np.argmax(predicted_vector)].split("/")[-1].split(".")[-1]


#def predict_breed(image_path):
#    img = cv2.imread(image_path)
#    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#    plt.imshow(img)
#    plt.show()
#    if face_detector(image_path):
#        print("This human resembels a {} ".format(resnet_predict_breed(image_path)))
#    elif dog_detector(image_path):
#        print("This dog is a {} ".format(resnet_predict_breed(image_path)))
#    else:
#        print("we couldn't find a dog or a human in this picture")


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/upload',methods=["POST"])
def upload():
    file = request.files["inputFile"]
    
    
    file.save(file.filename)
    
    img = cv2.imread(file.filename)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #r= request
    
    #nparr = np.fromstring(r.data, np.uint8)
    # decode image
    #img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    
    
    return file.filename