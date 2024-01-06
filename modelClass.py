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

# Optimizer
import wandb
from wandb.keras import WandbCallback
wandb.login()

# Our files
import data
import os


# Set your WandB API key
os.environ["WANDB_API_KEY"] = '76fdf9acb6a4a334b5b8c8f89c8a63c1c2b5135e'


##################################### 
### Note: Leagacy Adam optimizer used ###
##################################### 

##########
## Fixe: Optimizeren WANDb - fejl i det 
#######


path = './Data/Training/'
item = 'AMAZON_FASHION' 
# specific_path = './Data/Locally/AMAZON_FASHION_5.json'

X_train = pd.read_csv(path + item + '_X_train.csv')
X_val = pd.read_csv(path + item + '_X_val.csv')
Y_train = pd.read_csv(path + item + '_Y_train.csv')
Y_val = pd.read_csv(path + item + '_Y_val.csv')

df_train = pd.concat([Y_train, X_train], axis=1)
df_val = pd.concat([Y_val, X_val], axis=1)

# Ensure the text column is of string type and handle NaN values
df_train['overall'] = df_train['overall'] - 1  # to make it 0-4
df_train['text'] = df_train['text'].fillna('').astype(str)

df_val['overall'] = df_val['overall'] - 1  # to make it 0-4
df_val['text'] = df_val['text'].fillna('').astype(str)





class Build_Models():
    def __init__(self, df_train, df_val, filename,
                 max_tokens=1000,
                 output_sequence_length=200,
                 pad_to_max_tokens=True,
                 batch_size=62,
                 embedding_dimension=200,
                 epochs_n=5):

        # Data 
        self.filename = filename,
        self.dataTrain = df_train
        self.dataVal = df_val
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

    def adapt_encoder(self):
        self.encoder = tf.keras.layers.TextVectorization(max_tokens=self.max_tokens, 
                                            output_sequence_length=self.output_sequence_length, 
                                            pad_to_max_tokens=self.pad_to_max_tokens)
        text_ds = tf.data.Dataset.from_tensor_slices(self.dataTrain['text']).batch(self.batch_size)
        self.encoder.adapt(text_ds)

    def preprocess_data(self, text_data, label_data):
        # Create the full dataset with text and labels
        ds = tf.data.Dataset.from_tensor_slices((text_data, label_data)).batch(self.batch_size)
        # Apply TextVectorization to the text data in the dataset
        ds = ds.map(lambda x, y: (self.encoder(x), y))
        # Configure the dataset for performance
        AUTOTUNE = tf.data.AUTOTUNE 
        ds = ds.cache().prefetch(buffer_size=AUTOTUNE)
        return ds

    def train(self, train_ds, val_ds, early_stopping_patience=3):
        early_stopping = EarlyStopping(monitor='accuracy',
                                       patience=early_stopping_patience,
                                       restore_best_weights=True)

        # Use WandB callback for logging
        wandb_callback = wandb.keras.WandbCallback()

        self.model.fit(train_ds, 
                       epochs=self.epochs_n, 
                       validation_data=val_ds,
                       callbacks=[early_stopping, wandb_callback], 
                       verbose=1)


    def save_model(self, filename='saved_model'):
        self.model.save(filename)

    def evaluate(self, test_data):
        # Evaluate the model on the test set
        test_loss, test_accuracy = self.model.evaluate(test_data)
            
        print(f'Test Loss: {test_loss}')
        print(f'Test Accuracy: {test_accuracy}')


##### Example usage #### For Only class
# builder = Build_Models(df_train, df_val, 'RNN_Model')
# builder.adapt_encoder()
# train_ds = builder.preprocess_data(df_train['text'], df_train['overall'])
# builder.build_model()
# builder.train(train_ds)


##### Optimizer ##### 
wandb.init(project="your_project_name", name="your_run_name")

builder = Build_Models(df_train, df_val, 'File_Name')
builder.adapt_encoder()
train_ds = builder.preprocess_data(df_train['text'], df_train['overall'])
val_ds = builder.preprocess_data(df_val['text'], df_val['overall'])
builder.build_model()







# Optimizer Weight and Bias - Maybe add to class?
optimizer = tf.keras.optimizers.Adam()  # Use any optimizer of your choice
# https://docs.wandb.ai/tutorials/tensorflow Docs 
i = 0
for epoch in range(builder.epochs_n):
    print(i)
    i = i + 1
    # Training
    train_loss = []
    train_acc = []
    for batch_data, batch_labels in train_ds:
        with tf.GradientTape() as tape:
            predictions = builder.model(batch_data, training=True)
            loss = builder.model.compiled_loss(batch_labels, predictions)

        gradients = tape.gradient(loss, builder.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, builder.model.trainable_variables))

        acc = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(batch_labels, predictions))
        train_loss.append(loss.numpy())
        train_acc.append(acc.numpy())

    # Validation
    val_loss = []
    val_acc = []
    for val_data, val_labels in val_ds:
        val_predictions = builder.model(val_data, training=False)
        val_loss.append(builder.model.compiled_loss(val_labels, val_predictions).numpy())
        val_acc.append(tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(val_labels, val_predictions)).numpy())

    # Log metrics using WandB
    wandb.log({'epochs': epoch,
               'loss': np.mean(train_loss),
               'acc': float(np.mean(train_acc)), 
               'val_loss': np.mean(val_loss),
               'val_acc': float(np.mean(val_acc))})

# Save the model
builder.save_model()









if __name__ == "__main__":
    quit()

