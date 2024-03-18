#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from nltk.translate.bleu_score import corpus_bleu
import os
import glob
import random

# Load pre-trained VGG16 model (without top FC layers)
image_model = VGG16(weights='imagenet', include_top=False)

# Define image feature extractor model
image_input = Input(shape=(224, 224, 3))
x = image_model(image_input)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
image_features = Dense(256, activation='relu')(x)
image_features_extractor = Model(inputs=image_input, outputs=image_features)

# Load and preprocess image
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return img

# Load and preprocess captions
def load_captions(captions_path):
    captions = {}
    with open(captions_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            img_name, caption = parts[0], parts[1]
            if img_name not in captions:
                captions[img_name] = []
            captions[img_name].append(caption)
    return captions

# Define tokenizer and preprocess captions
def tokenize_captions(captions):
    tokenizer = Tokenizer(oov_token='<unk>')
    tokenizer.fit_on_texts(captions)
    return tokenizer

# Generate sequences from captions
def generate_sequences(tokenizer, captions, max_sequence_length):
    sequences = tokenizer.texts_to_sequences(captions)
    sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
    return sequences

# Define LSTM model for caption generation
def define_model(vocab_size, max_sequence_length, embedding_dim):
    image_features_input = Input(shape=(256,))
    image_features_dropout = Dropout(0.5)(image_features_input)
    image_features_dense = Dense(256, activation='relu')(image_features_dropout)

    caption_input = Input(shape=(max_sequence_length,))
    caption_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(caption_input)
    caption_dropout = Dropout(0.5)(caption_embedding)
    caption_lstm = LSTM(256)(caption_dropout)

    decoder = tf.keras.layers.add([image_features_dense, caption_lstm])
    decoder_dropout = Dropout(0.5)(decoder)
    output = Dense(vocab_size, activation='softmax')(decoder_dropout)

    model = Model(inputs=[image_features_input, caption_input], outputs=output)
    return model

# Load images and captions
image_dir = 'images/'
captions_path = 'captions.txt'
captions = load_captions(captions_path)

# Tokenize captions
all_captions = [caption for img_captions in captions.values() for caption in img_captions]
tokenizer = tokenize_captions(all_captions)
vocab_size = len(tokenizer.word_index) + 1

# Generate sequences
max_sequence_length = max(len(caption.split()) for caption in all_captions)
sequences = generate_sequences(tokenizer, all_captions, max_sequence_length)

# Train-test split
random.seed(42)
all_image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
random.shuffle(all_image_paths)
split = int(0.8 * len(all_image_paths))
train_image_paths = all_image_paths[:split]
test_image_paths = all_image_paths[split:]

# Define model
embedding_dim = 256
model = define_model(vocab_size, max_sequence_length, embedding_dim)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train model
num_epochs = 10
batch_size = 32
checkpoint_path = 'model_checkpoint.h5'
checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
for epoch in range(num_epochs):
    for img_path in train_image_paths:
        img_name = os.path.basename(img_path)
        img_features = image_features_extractor(load_and_preprocess_image(img_path))
        for caption in captions[img_name]:
            target_sequence = tokenizer.texts_to_sequences([caption])[0]
            for i in range(1, len(target_sequence)):
                input_sequence, output_sequence = target_sequence[:i], target_sequence[i]
                input_sequence = pad_sequences([input_sequence], maxlen=max_sequence_length, padding='post')[0]
                output_sequence = to_categorical([output_sequence], num_classes=vocab_size)[0]
                model.fit([np.array([img_features]), np.array([input_sequence])], np.array([output_sequence]), 
                          verbose=0, batch_size=batch_size, callbacks=[checkpoint_callback], validation_split=0.2)

# Evaluate model
model.load_weights(checkpoint_path)
bleu_scores = []
for img_path in test_image_paths:
    img_name = os.path.basename(img_path)
    img_features = image_features_extractor(load_and_preprocess_image(img_path))
    caption = generate_caption(img_features)
    references = [caption.split()]
    hypothesis = captions[img_name][0].split()
    bleu_score = corpus_bleu(references, hypothesis)
    bleu_scores.append(bleu_score)
average_bleu_score = np.mean(bleu_scores)
print("Average BLEU Score:", average_bleu_score)

