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
import data


##### Load data #####
# data.Dataloader()


# Constants
max_tokens = 1000 # the maximum number of words to keep, based on word frequency. Only the most common `max_tokens-1` words will be kept.
output_sequence_length = 200 # the maximum length of the sequence to keep. Sequences longer than this will be truncated.
pad_to_max_tokens = True # whether to pad to the `output_sequence_length`.
batch_size = 62
embedding_dimension = 200
epochs_n = 5


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


#############
## Encoder ##
#############


# Initialize the TextVectorization layer
encoder = tf.keras.layers.TextVectorization(max_tokens=max_tokens, 
                                            output_sequence_length=output_sequence_length, 
                                            pad_to_max_tokens=pad_to_max_tokens)




# Adapt the encoder to the training data
text_ds = tf.data.Dataset.from_tensor_slices(df_train['text']).batch(batch_size)
encoder.adapt(text_ds)
vocab = np.array(encoder.get_vocabulary())

# Word Counts
# Create the full train dataset with text and labels
train_ds = tf.data.Dataset.from_tensor_slices((df_train['text'], df_train['overall'])).batch(batch_size)
# Apply TextVectorization to the text data in the dataset
train_ds = train_ds.map(lambda x, y: (encoder(x), y))

# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE 
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Similarly, adapt the encoder to the validation data
df_val = df_val.dropna() # Necessary, when loading from JSON
val_ds = tf.data.Dataset.from_tensor_slices((df_val['text'], df_val['overall'])).batch(batch_size)
# Apply TextVectorization to the text data in the dataset
val_ds = val_ds.map(lambda x, y: (encoder(x), y))
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



###############################
############ Model ############
###############################
# Define early stopping callback
early_stopping = EarlyStopping(monitor='accuracy',  # or 'val_accuracy' depending on your preference
                               patience=3,  # Number of epochs with no improvement after which training will be stopped
                               restore_best_weights=True)


embedding_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=len(vocab), 
                              output_dim=embedding_dimension,
                              input_length=200,
                              trainable=True, # Embeddings Trainable (Defulat in Tensor) 
                              name="embedding"), 
    Dropout(0.5),
    LayerNormalization(axis=-1), # Normalize the Embedding 
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, return_sequences=True)),
    tf.keras.layers.LSTM(100, return_sequences=True,
                         kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.LSTM(50),
    Dropout(0.5),
    tf.keras.layers.Dense(5, activation='softmax')
])

embedding_model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])





embedding_model.summary()
embedding_model.fit(train_ds, epochs=epochs_n, callbacks=[early_stopping], verbose=1)

# Make predictions
predictions = embedding_model.predict(val_ds)
# The 'predictions' array will contain the probabilities of each class for each sample

# Convert probabilities to class labels
predicted_labels = np.argmax(predictions, axis=1)+1
print(predicted_labels)

# 'predicted_labels' now contains the class label (1 to 5) for each sample in your test dataset
# print(predicted_labels)

y_test_hat_pd = pd.DataFrame({
    'Id': list(range(len(predicted_labels))),
    'Predicted': predicted_labels.reshape(-1),
})
y_test_hat_pd.to_csv('y_test_hat.csv', index=False)

# Add Plotting + Saving mordel  -> Data in another file 

embedding_model.save('change_1_saved_model')


def myEvaluate(test_data, model):
    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_data)
    
    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_accuracy}')














if __name__ == "__main__":
    quit()

