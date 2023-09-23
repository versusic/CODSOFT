# import modules
import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image
import graphviz
import pandas as pd
import os
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from keras import layers, Input, Model
from keras.layers import Dense, Flatten, Dropout, Embedding, LSTM, add
from keras.models import Sequential
from keras.optimizers import Adam
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import image_dataset_from_directory
from keras.utils import to_categorical, plot_model
import pathlib
import cv2

# Load data sets
base_dir = "D:\My Data Sets\\vehicles\\all images"
save_path = "D:\My Data Sets\\vehicles\Saved"

# choose the model
resnet_model = Sequential()
pretrained_model = tf.keras.applications.ResNet50(include_top=False,
                                                  input_shape=(180, 180, 3),
                                                  pooling='avg',
                                                  weights='imagenet')

for layer in pretrained_model.layers:
    layer.trainable = False

# modify the model
resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))

# extract features
features = {}
directory = os.path.join(base_dir, 'images')
for img_name in tqdm(os.listdir(directory)):
    img_path = directory + "/" + img_name
    image = load_img(img_path, target_size=(180, 180))
    image = img_to_array(image)
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    image = preprocess_input(image)
    feature = resnet_model.predict(image, verbose=0)
    img_id = img_name.split('.')[0]
    features[img_id] = feature

# save features to save time
pickle.dump(features, open(os.path.join(save_path, 'features.pkl'), 'wb'))
with open(os.path.join(save_path, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)
with open(os.path.join(base_dir, 'captions.txt'), 'r') as f:
    captions_doc = f.read()

# map features to captions
mapping = {}
for line in tqdm(captions_doc.split('\n')):
    tokens = line.split(',')
    image_id, caption = tokens[0], tokens[1:]
    image_id = image_id.split('.')[0]
    caption = " ".join(caption)
    if image_id not in mapping:
        mapping[image_id] = []
    mapping[image_id].append(caption)

# function to preprocess captions
def preproccesing_mapping(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower()
            caption = caption.replace('[^A-Za-z]', '')
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
            captions[i] = caption
# apply the function
preproccesing_mapping(mapping)

# make all modified captions in one set and fit them with tokenizer
all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in all_captions)

# make training and validations parts
image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90)
train = image_ids[:split]
test = image_ids[split:]

# make a function to preprocess the whole data to make it faster to work
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key in data_keys:
            n+=1
            captions = mapping[key]
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
                X1, X2, y =np.array(X1), np.array(X2), np.array(y)
                yield [X1, X2], y
                X1, X2, y = list(), list(), list()
                n = 0

# modify the model and make the inputs
inputs1 = Input(shape=(512,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation="relu")(fe1)
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

# make the decoder and outputs
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation="relu")(decoder1)
outputs = Dense(vocab_size, activation="softmax")(decoder2)

# finalize the model
resnet_model = Model(inputs=[inputs1, inputs2], outputs= outputs)
resnet_model.compile(loss='categorical_crossentropy', optimizer='adam')

# start training
epochs = 50
batch_size = 64
step = len(train) // batch_size
for i in range(epochs):
    generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
    resnet_model.fit(generator, epochs= 1, steps_per_epoch=step, verbose=1)

# function to change photos index
def index_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# predict captions
def predict_caption(model, image, tokenizer, max_length):
    in_text ='startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        next_y = model.predict([image, sequence], verbose= 0)
        next_y = np.argmax(next_y)
        word = index_to_word(next_y, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    in_text = in_text[9:-7]
    return in_text

# check accuracy of the model
actual, predicted = list(), list()
for key in tqdm(test):
    captions = mapping[key]
    y_pred = predict_caption(resnet_model, features[key], tokenizer, max_length)
    actual_captions = [caption.split() for caption in captions]
    y_pred = y_pred.split()
    actual.append(actual_captions)
    predicted.append(y_pred)
print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))

# define the prediction function
def pred_caption(image_name):
    image_id = image_name.split('.')[0]
    image_path = os.path.join(base_dir, "Images", image_name)
    image = Image.open(image_path)
    captions = mapping[image_id]
    y_pred = predict_caption(resnet_model, features[image_id], tokenizer, max_length)
    print(y_pred)
    plt.imshow(image)
    plt.show()

# test
pred_caption("0009.jpg")
pred_caption("0021.jpg")
pred_caption("0061.jpg")
pred_caption("0045.jpg")

