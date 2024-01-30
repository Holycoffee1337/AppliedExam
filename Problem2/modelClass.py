"""
File: modelClass.py, provides, the class
structure for the RNN model.
"""

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
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ModelCheckpoint


# Build RNN Model
class Build_Models():
    """
    Build_Models is a class to build a RNN model,
    for Amazon reviews classification.

    Precondition:
        Class DataHandler: './RESULTS/DATA_CLASSES/'
        If no data class, create with DataHandler from 
        data_preprocessing.py

    Parameters:
        Paremeters: (Best parameter Hardcoded) can take wandb.config from sweep
        
    """
    def __init__(self, DataHandler,
                 category_name = None,
                 model_name = 'Model_name',
                 max_tokens=1000,
                 output_sequence_length=100,
                 pad_to_max_tokens=True,
                 batch_size=62,
                 embedding_dimension=400,
                 lr = 0.0001,
                 epochs_n=1,
                 optimizer_name = 'sgd',
                 lr_scheduler = 'decay'):
    
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
        self.lr = lr
        self.optimizer_name = optimizer_name 
        self.lr = lr_scheduler

        
        # Sweep Config initialization 
        self.max_tokens = wandb.config.get('max_tokens', 1000)
        self.output_sequence_length = wandb.config.get('output_sequence_length', 200)
        self.embedding_dimension = wandb.config.get('embedding_dimension', 200)
        self.batch_size = wandb.config.get('batch_size', 64)
        self.epochs_n = wandb.config.get('n_epochs', 5)
        self.lr = wandb.config.get('lr', 0.001)
        self.optimizer = wandb.config.get('optimizer', 'adam')  
        self.lr_scheduler = wandb.config.get('lr_scheduler', 'constant')
    

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


    # Building the model, with Bidirectional GRU layers + Dense layers
    def build_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(
                input_dim=len(self.encoder.get_vocabulary()),
                output_dim=self.embedding_dimension,
                input_length=self.output_sequence_length,
                trainable=True,
                name="embedding"
            ),

            tf.keras.layers.LayerNormalization(axis=-1), 
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.01))), # Note: The last RNN layer should not return sequences
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(5, activation='softmax') # Assuming you have 5 output classes for the ratings

        ])
    
        # Compile the model
        if self.optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        elif self.optimizer_name == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr)
        
        # Creating optimizer 
        self.model.compile(optimizer= optimizer, 
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        return self.model


    def _adapt_encoder(self):
        self.encoder = tf.keras.layers.TextVectorization(max_tokens=self.max_tokens, 
                                            output_sequence_length=self.output_sequence_length, 
                                            pad_to_max_tokens=self.pad_to_max_tokens)
        text_ds = tf.data.Dataset.from_tensor_slices(self.data_train['text']).batch(self.batch_size)
        self.encoder.adapt(text_ds)


    def _preprocess_data(self, text_data, label_data):
        # Create the full dataset with text and labels
        ds = tf.data.Dataset.from_tensor_slices((text_data, label_data)).batch(self.batch_size) #  Maybe wandb.
        ds = ds.map(lambda x, y: (self.encoder(x), y))
        AUTOTUNE = tf.data.AUTOTUNE 
        ds = ds.cache().prefetch(buffer_size=AUTOTUNE)
        return ds


    def train(self, early_stopping_patience=3):
        early_stopping = EarlyStopping(monitor='accuracy',
                                       patience=early_stopping_patience,
                                       restore_best_weights=True)

        # Use WandB callback for logging
        model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
        wandb_callback = wandb.keras.WandbCallback()


        scheduler = ExponentialDecay(self.lr, 10000, 0.9, staircase=True)
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
        callbacks= [early_stopping, wandb_callback] 
        if self.lr_scheduler == 'decay':
            callbacks.append(lr_scheduler)
           
        
        self.model.fit(self.ds_train, 
                       epochs= self.epochs_n, 
                       validation_data=self.ds_val,
                       callbacks= callbacks, 
                       verbose = 1)


    def save_model(self):
        path = './RESULTS/RNN/'
        self.model.save(self.filename)


    def evaluate(self):

        # Loading model
        self.model.load_weights('path/to/best_model.h5')

        # 
        DataClass = self.DataHandler
        test = DataClass.get_test_data() 
        test_data = test['text']
        test_labels = test['overall'] - 1
        
        self._adapt_encoder()
        test_ds = self._preprocess_data(test_data, test_labels)
        test_loss, test_accuracy = self.model.evaluate(test_ds)
            
        print(f'Test Loss: {test_loss}')
        print(f'Test Accuracy: {test_accuracy}')


        # Save the evaluation
        report_content = f'Test Loss: {test_loss}\nTest Accuracy: {test_accuracy}'
        filename = 'RNN_TEST_EVALUATION_REPORT_2_BROADER.txt'

        with open(filename, 'w') as file:
            file.write(report_content)



if __name__ == "__main__":
    quit()

