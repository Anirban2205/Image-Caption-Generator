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
from preprocess import vocab_creation
from data_loader import load_features, data_generator
from model import Caption_model

BASE_DIR = './Dataset'
WORKING_DIR = './Outputs'
def train():
    features, mapping = load_features()
    tokenizer, vocab_size, max_length = vocab_creation(mapping)

    image_ids = list(mapping.keys())
    split = int(len(image_ids) * 0.90)
    train = image_ids[:split]
    test = image_ids[split:]

    model = Caption_model(max_length, vocab_size)
    epochs = 1
    batch_size = 32
    steps = len(train) // batch_size

    for i in range(epochs):
        generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
        model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

    model.save(WORKING_DIR+'/best_model.h5')

if __name__ == "__main__":
    train()

