import os
import pickle
import numpy as np
from tqdm.notebook import tqdm
import tensorflow
from tensorflow import keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils import load_img, img_to_array
from keras.models import Model

BASE_DIR = './Dataset'
WORKING_DIR = './Outputs'

def load_new_model():
    model = VGG16()
    # restructure the model
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    model.save("vgg16.h5")
    return model

def load_saved_model(model_path):
    from keras.models import load_model
    model = load_model(model_path)
    return model

def extract_features(BASE_DIR, model, WORKING_DIR):
    features = {}
    directory = os.path.join(BASE_DIR, 'Images')

    for img_name in tqdm(os.listdir(directory)):
        img_path = directory + '/' + img_name
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = img_name.split('.')[0]
        features[image_id] = feature
    
    # store features in pickle
    pickle.dump(features, open(os.path.join(WORKING_DIR, 'features.pkl'), 'wb'))
    
    return features