# RNN Model
import pandas as pd
import tensorflow as tf
import numpy as np

import tensorflow as tf
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.layers import Embedding, LSTM, Dense, Dropout, LayerNormalization
from sklearn.model_selection import train_test_split
import re
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import wandb
from wandb.keras import WandbCallback
# wandb.login()


class Build_Models():
    def __init__(self, DataHandler,
                 category_name = None,
                 model_name = 'Model_name',
                 max_tokens=1000,
                 output_sequence_length=200,
                 pad_to_max_tokens=True,
                 batch_size=62,
                 embedding_dimension=200,
                 epochs_n=10):
    
        self.filename =  model_name
        self.category_name = category_name
        self.model = None
        self.encoder = None

    
        # Hyperparameters                                    
        self.epochs_n = epochs_n                             
        self.max_tokens = max_tokens                         
        self.output_sequence_length = output_sequence_length 
        self.pad_to_max_tokens = pad_to_max_tokens           
        self.embedding_dimension = embedding_dimension       
        self.batch_size = batch_size                         
        self.epochs_n = epochs_n                             


        # Data 
        self.DataHandler = DataHandler
        if category_name is None:
            # If no category is specified, use the combined training data
            self.data_train = self.DataHandler.get_combined_train_data()
            self.data_val = self.DataHandler.get_combined_val_data()
        else:
            # If a category is specified, use that category's training and validation data
            self.data_train = self.DataHandler.get_category_data(category_name, 'train')
            self.data_val = self.DataHandler.get_category_data(category_name, 'val')

        # Encoder
        self._adapt_encoder()
        
        # Ds data
        self.ds_train = self._preprocess_data(self.data_train['text'], self.data_train['overall']) 
        self.ds_val = self._preprocess_data(self.data_val['text'], self.data_val['overall']) 


    def build_model(self):
        early_stopping = EarlyStopping(monitor='accuracy',  # or 'val_accuracy' depending on your preference
                                       patience=3,  # Number of epochs with no improvement after which training will be stopped
                                       restore_best_weights=True)

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(input_dim=len(self.encoder.get_vocabulary()), 
                                      output_dim=self.embedding_dimension,
                                      input_length=self.output_sequence_length,
                                      trainable=True, # Embeddings Trainable (Defulat in Tensor) 
                                      name="embedding"), 
            Dropout(0.5),
            LayerNormalization(axis=-1), # Normalize the Embedding 
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, return_sequences=True)),
            tf.keras.layers.LSTM(100, return_sequences=True,
                                 kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.LSTM(50),
            Dropout(0.5),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        
        # Note legacy Adam optimizer used instead of 'adam'
        self.model.compile(optimizer = tf.keras.optimizers.legacy.Adam(),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def _adapt_encoder(self):
        self.encoder = tf.keras.layers.TextVectorization(max_tokens=self.max_tokens, 
                                            output_sequence_length=self.output_sequence_length, 
                                            pad_to_max_tokens=self.pad_to_max_tokens)
        text_ds = tf.data.Dataset.from_tensor_slices(self.data_train['text']).batch(self.batch_size)
        self.encoder.adapt(text_ds)

    def _preprocess_data(self, text_data, label_data):
        # Create the full dataset with text and labels
        ds = tf.data.Dataset.from_tensor_slices((text_data, label_data)).batch(self.batch_size)
        # Apply TextVectorization to the text data in the dataset
        ds = ds.map(lambda x, y: (self.encoder(x), y))
        # Configure the dataset for performance
        AUTOTUNE = tf.data.AUTOTUNE 
        ds = ds.cache().prefetch(buffer_size=AUTOTUNE)
        return ds

    def train(self, early_stopping_patience=3):
        early_stopping = EarlyStopping(monitor='accuracy',
                                       patience=early_stopping_patience,
                                       restore_best_weights=True)

        # Use WandB callback for logging
        wandb_callback = wandb.keras.WandbCallback()

        self.model.fit(self.ds_train, 
                       epochs=self.epochs_n, 
                       validation_data=self.ds_val,
                       callbacks=[early_stopping, wandb_callback], 
                       verbose=1)


    def save_model(self):
        self.model.save(self.filename)

    def evaluate(self, test_data):
        # Evaluate the model on the test set
        test_loss, test_accuracy = self.model.evaluate(test_data)
            
        print(f'Test Loss: {test_loss}')
        print(f'Test Accuracy: {test_accuracy}')






if __name__ == "__main__":
    quit()

