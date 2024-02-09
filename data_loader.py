import os
import pickle
import numpy as np
from tqdm.notebook import tqdm
import tensorflow
from tensorflow import keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Model
from keras.utils import to_categorical, plot_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from feature_extractor import load_saved_model, extract_features 
from preprocess import load_saved_captions, map_img_to_cap, clean, vocab_creation

BASE_DIR = './Dataset'
WORKING_DIR = './Outputs'

def load_features():
    model = load_saved_model("./vgg16.h5")

    features = extract_features(BASE_DIR, model, WORKING_DIR)
    captions = load_saved_captions()
    mapping = map_img_to_cap(captions)

    mapping = clean(mapping)

    return features, mapping


def data_generator(data_keys,features, mapping, tokenizer, vocab_size, max_length, batch_size):
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            # process each caption
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield [X1, X2], y
                X1, X2, y = list(), list(), list()
                n = 0

