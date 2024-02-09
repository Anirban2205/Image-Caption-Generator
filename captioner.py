import os
import pickle
import numpy as np
from tqdm.notebook import tqdm
import tensorflow
from tensorflow import keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils import load_img, img_to_array
from keras.models import Model
from preprocess import vocab_creation
from data_loader import load_features, data_generator
from keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import nltk
from nltk.translate.bleu_score import corpus_bleu
from tester import predict_caption

BASE_DIR = './Dataset'
WORKING_DIR = './Outputs'


def generate_local_caption(image_name, model, tokenizer, mapping, max_length, features):
    image_id = image_name.split('.')[0]
    img_path = os.path.join(BASE_DIR, "Images", image_name)
    image = Image.open(img_path)
    captions = mapping[image_id]
    print('---------------------Actual---------------------')
    for caption in captions:
        print(caption)

    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    print('--------------------Predicted--------------------')
    print(y_pred)
    plt.imshow(image)
    return y_pred

def real_img_caption(image_path, model, tokenizer, max_length):
    vgg_model = VGG16()
    vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
    # image_path = "./flagged/image/tmp95u0_plp.jpg"
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = vgg_model.predict(image, verbose=0)
    cap_tag = predict_caption(model, feature, tokenizer, max_length)

    s = ''
    # print(cap_tag.split())
    for w in cap_tag.split():
        if w == "startseq" or w == "endseq":
            continue
        else:
            s = s + w + " "
    print(s.capitalize())

if __name__ == "__main__":
    model = load_model(WORKING_DIR+'/best_model.h5')
    features, mapping = load_features()
    tokenizer, vocab_size, max_length = vocab_creation(mapping)
    img_loc = input("Give the local image name: ")
    generate_local_caption(img_loc, model, tokenizer, mapping, max_length, features)
    img_path = input("Enter a image path: ")
    real_img_caption(img_path, model, tokenizer, max_length)

