import os
import pickle
import numpy as np
from tqdm.notebook import tqdm
import tensorflow
from tensorflow import keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils import load_img, img_to_array
from keras.utils import pad_sequences
from keras.models import Model
from preprocess import vocab_creation
from data_loader import load_features, data_generator
from keras.models import load_model
import nltk
from nltk.translate.bleu_score import corpus_bleu

BASE_DIR = './Dataset'
WORKING_DIR = './Outputs'

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break

    return in_text

def metric(model, tokenizer, mapping, test_data, max_length, features):
    actual, predicted = list(), list()

    for key in tqdm(test_data):
        captions = mapping[key]
        y_pred = predict_caption(model, features[key], tokenizer, max_length) 
        actual_captions = [caption.split() for caption in captions]
        y_pred = y_pred.split()
        actual.append(actual_captions)
        predicted.append(y_pred)
    
    # calcuate BLEU score
    print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print("BLEU-3: %f" % corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0)))
    print("BLEU-4: %f" % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


if __name__ == "__main__":
    model = load_model(WORKING_DIR+'/best_model.h5')
    features, mapping = load_features()
    tokenizer, vocab_size, max_length = vocab_creation(mapping)
    image_ids = list(mapping.keys())
    split = int(len(image_ids) * 0.90)
    test_data = image_ids[split:]
    scores = metric(model, tokenizer, mapping, test_data, max_length, features)