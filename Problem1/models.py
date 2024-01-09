from data import DataLoader
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout,Conv2DTranspose,BatchNormalization,Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
import numpy as np
import wandb
from wandb.keras import WandbCallback
import random
class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = self.encoder_build()
        self.latent_space_encoder = self.latent_space_build()
        self.decoder_first_layer = self.decoder_first_layer_build()
        self.decoder = self.decoder_build()
        
    def encoder_build(self):
        encoder = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),strides=2, activation='relu',padding='same', input_shape=(96, 96, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),strides=2, activation='relu',padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1),strides=1, activation='relu',padding='same'),
            tf.keras.layers.BatchNormalization(),
        ])
        #encoder.summary()
        return encoder
    def latent_space_build(self):
        latent_space = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape = (24,24,1)),   
            tf.keras.layers.Dense(2 * self.latent_dim), # 2 since we encode mean and standard deviation
        ])
        #latent_space.summary()
        return latent_space

    def decoder_first_layer_build(self):
        decoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=24*24*1, activation='relu', input_shape=(self.latent_dim,)),
            tf.keras.layers.Reshape(target_shape=(24, 24,1)), # To get in "image format"
        ])
        #decoder.summary()
        return decoder
    
    def decoder_build(self):
        decoder = tf.keras.models.Sequential([
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3),strides=2, activation='relu',padding='same', input_shape=(24, 24, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3),strides=2, activation='relu',padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=1,strides=1, activation='relu'),
        ])
        #decoder.summary()
        return decoder
    
    
    def encode(self, x):
        return self.encoder(x)
    
    def latent_space(self,x):
        param = self.latent_space_encoder(x)
        return tf.split(param, num_or_size_splits=2, axis=1) # mean, logvar    
    
    def decode_first_layer(self,z):
        return self.decoder_first_layer(z)
    
    def decode(self, z):
        return self.decoder(z)
    
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean #sigma= sqrt(exp(logvar))
    
    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return tf.sigmoid(self.decode(self.decode_first_layer(eps)))
   
class TrainVAE():
    def __init__(self,latent_dim = 24*24):
        self.data = DataLoader()
        self.model = VAE(latent_dim)
        """
        self.sweep_config = {
                'method': 'grid',
                'metric': {
                    'name': 'val_loss',
                    'goal': 'minimize'
                },
                'early_terminate':{
                    'type': 'hyperband',
                    'min_iter': 5
                },  
                'parameters': {
                    'learning_rate':{
                        'values': [0.01, 0.0001]
                        }
                }
            }
        self.sweep_id = wandb.sweep(self.sweep_config)
        """
    def log_normal_pdf(self,sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        vals = -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi)
        return tf.reduce_sum(vals, axis=raxis)


    def compute_loss(self,model, x):
        # Output from encoder
        encoded = model.encode(x)
        mean, logvar = model.latent_space(encoded)
        # The reparameterization trick
        z = model.reparameterize(mean, logvar)    
        # We assume that p(x|z) is multivariate Bernoulli, ie. the final dense layer 
        # has a sigmoid activation function, see page. 11
        # in Kingma, D. P., & Welling, M. (2013).
        z_prime = model.decode_first_layer(z)
        x_logit = model.decode(z_prime)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, 
                                                            labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])    
        # Assume normaility of p(z)
        logpz = self.log_normal_pdf(z, 0., 0.)    
        # Assume normality of q(z|x)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        # -tf.reduce_mean(decoder + sampler - encoder)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)
    
    
    @tf.function
    def train_step(self, model, x,optimizer):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(model, x)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    def run(self, epochs=1):
        train_dataset, _, val_dataset = self.data.create_batches_image()
        #config_defaults = {
        #    "learning_rate": 0.0001
        #}
        #wandb.init(config=config_defaults, project="Train VAE")
        #wandb.config.epochs = epochs
        #self.optimizer = tf.keras.optimizers.Adam(learning_rate=wandb.config.learning_rate)
        optimizer = tf.keras.optimizers.Adam(0.0001)
        

        #wandb_callback = WandbCallback()
        for epoch in range(epochs):
            for train_x in train_dataset:
                self.train_step(self.model, train_x,optimizer)
                
            loss = tf.keras.metrics.Mean()
            for val_x in val_dataset:
                loss(self.compute_loss(self.model, val_x))
                
            variational_lower_bound = -loss.result()

    
            print(f'Epoch: {epoch}, Test set variational lower bound: {variational_lower_bound}')
        """
        for epoch in range(epochs):
            train_loss = []
            val_loss = []

            # Training
            for train_x in train_dataset:
                self.train_step(self.model, train_x)
                #train_loss.append(float(loss_value))
            
            # Validation
            loss = tf.keras.metrics.Mean()
            for val_x in val_dataset:
                loss_value = loss(self.compute_loss(self.model, val_x))
                val_loss.append(float(loss_value))

            # Calculate and log metrics
            mean_train_loss = np.mean(train_loss)
            mean_val_loss = np.mean(val_loss)

            variational_lower_bound = -mean_val_loss  # Assuming you want to log the validation loss
            """
        """
            wandb.log({
                'epochs': epoch,
                'loss': mean_train_loss,
                'val_loss': mean_val_loss,
            })

            # Manually call WandbCallback methods
            wandb_callback.on_epoch_end(epoch, {
                'loss': mean_train_loss,
                'val_loss': mean_val_loss,
            })
            """

        # Save model manually if needed
        #self.model.save_weights('your_model_weights.h5')
        #print(f'Epoch: {epoch}, Test set variational lower bound:{variational_lower_bound}')
    
    def train(self, epochs = 2):
        wandb.agent(self.sweep_id, function=lambda: self.run(epochs), count = 1)
            
    def latent_space(self):
        train_dataset,_,_ = self.data.create_batches_image()
        for train_x in train_dataset:
                return self.model.encode(train_x)
   
    def encode_decode(self, image):
        encode = self.model.encoder.predict(image, verbose = 0)
        decode = self.model.decoder.predict(encode, verbose = 0)
        return decode
        
    def plot(self,n):
        train_dataset,_,_ = self.data.create_batches_image()
        plt.figure(figsize=(20, 4))
        for i, images in enumerate(train_dataset.take(n)):  # Take 1 batch for visualization
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(images[i].numpy())
            plt.title("original")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
            ax = plt.subplot(2, n, i + 1 + n)
            encoded_image = self.model.encode(images)
            mean, logvar = self.model.latent_space(encoded_image)
            
            z = self.model.reparameterize(mean, logvar)
            predictions = self.model.sample(z)
            
            
            plt.imshow(predictions[i][:, :, 0])
            plt.title("reconstructed")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

class VAE2(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE2, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = self.encoder_build()
        self.decoder = self.decoder_build()
        
    def encoder_build(self):
        encoder = tf.keras.models.Sequential([
            Conv2D(32, 3, 2, activation='relu',padding='same', input_shape=(96, 96, 3)),
            Conv2D(32, 3, 2, activation='relu',padding='same'),
            Flatten(),
            Dense(2 * self.latent_dim), # 2 since we encode mean and standard deviation
        ])
        #encoder.summary()
        return encoder

    def decoder_build(self):
        decoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=24*24*32, activation='relu', input_shape=(self.latent_dim,)),
            tf.keras.layers.Reshape(target_shape=(24, 24, 32)), # To get in "image format"
            tf.keras.layers.Conv2DTranspose(32, 3, 2, padding='same', activation='relu'),
            tf.keras.layers.Conv2DTranspose(32, 3, 2, padding='same', activation='relu'),
            tf.keras.layers.Conv2DTranspose(3, 3, 1, padding='same'),
        ])
        #decoder.summary()
        return decoder
    
    def encode(self, x):
        params = self.encoder(x)
        return tf.split(params, num_or_size_splits=2, axis=1) # mean, logvar
        
    def decode(self, z):
        return self.decoder(z)
    
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean #sigma= sqrt(exp(logvar))

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return tf.sigmoid(self.decode(eps))

    
class TrainVAE2():
    def __init__(self,latent_dim = 2):
        self.data = DataLoader()
        self.model = VAE2(latent_dim)
        """
        self.sweep_config = {
                'method': 'grid',
                'metric': {
                    'name': 'val_loss',
                    'goal': 'minimize'
                },
                'early_terminate':{
                    'type': 'hyperband',
                    'min_iter': 5
                },  
                'parameters': {
                    'learning_rate':{
                        'values': [0.01, 0.0001]
                        }
                }
            }
        self.sweep_id = wandb.sweep(self.sweep_config)
        """
    def log_normal_pdf(self,sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        vals = -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi)
        return tf.reduce_sum(vals, axis=raxis)


    def compute_loss(self,model, x):
        # Output from encoder
        mean, logvar = model.encode(x)    
        # The reparameterization trick
        z = model.reparameterize(mean, logvar)    
        # We assume that p(x|z) is multivariate Bernoulli, ie. the final dense layer 
        # has a sigmoid activation function, see page. 11
        # in Kingma, D. P., & Welling, M. (2013).
        x_logit = model.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, 
                                                            labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])    
        # Assume normaility of p(z)
        logpz = self.log_normal_pdf(z, 0., 0.)    
        # Assume normality of q(z|x)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        # -tf.reduce_mean(decoder + sampler - encoder)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)
    
    
    @tf.function
    def train_step(self, model, x,optimizer):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(model, x)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    def run(self, epochs=1):
        train_dataset, _, val_dataset = self.data.create_batches_image()
        #config_defaults = {
        #    "learning_rate": 0.01
        #}
        #wandb.init(config=config_defaults, project="Train VAE")
        #wandb.config.epochs = epochs
        #self.optimizer = tf.keras.optimizers.Adam(learning_rate=wandb.config.learning_rate)
        optimizer = tf.keras.optimizers.Adam(0.0001)
        

        #wandb_callback = WandbCallback()
        for epoch in range(epochs):
            for train_x in train_dataset:
                self.train_step(self.model, train_x,optimizer)
                
            loss = tf.keras.metrics.Mean()
            for val_x in val_dataset:
                loss(self.compute_loss(self.model, val_x))
                
            variational_lower_bound = -loss.result()

    
            print(f'Epoch: {epoch}, Test set variational lower bound: {variational_lower_bound}')
        """
        for epoch in range(epochs):
            train_loss = []
            val_loss = []

            # Training
            for train_x in train_dataset:
                self.train_step(self.model, train_x)
                #train_loss.append(float(loss_value))
            
            # Validation
            loss = tf.keras.metrics.Mean()
            for val_x in val_dataset:
                loss_value = loss(self.compute_loss(self.model, val_x))
                val_loss.append(float(loss_value))

            # Calculate and log metrics
            mean_train_loss = np.mean(train_loss)
            mean_val_loss = np.mean(val_loss)

            variational_lower_bound = -mean_val_loss  # Assuming you want to log the validation loss
            """
        """
            wandb.log({
                'epochs': epoch,
                'loss': mean_train_loss,
                'val_loss': mean_val_loss,
            })

            # Manually call WandbCallback methods
            wandb_callback.on_epoch_end(epoch, {
                'loss': mean_train_loss,
                'val_loss': mean_val_loss,
            })
            """

        # Save model manually if needed
        #self.model.save_weights('your_model_weights.h5')
        #print(f'Epoch: {epoch}, Test set variational lower bound:{variational_lower_bound}')
    
    def train(self, epochs = 2):
        wandb.agent(self.sweep_id, function=lambda: self.run(epochs), count = 1)
            
    def latent_space(self):
        train_dataset,_,_ = self.data.create_batches_image()
        for train_x in train_dataset:
                return self.model.encode(train_x)
   
    def encode_decode(self, image):
        encode = self.model.encoder.predict(image, verbose = 0)
        decode = self.model.decoder.predict(encode, verbose = 0)
        return decode
    
        
    def plot(self):
        train_dataset,_,_ = self.data.create_batches_image()
        sample = None
        for x in train_dataset:
            sample = x
            break
        plt.figure(figsize=(20, 4))
        mean, logvar = self.model.encode(sample)
        z = self.model.reparameterize(mean, logvar)
        predictions = self.model.sample(z)
        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0])
            plt.axis('off')

        #plt.savefig('./vae-img/image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()
        
    def plot2(self,n):
        train_dataset,_,_ = self.data.create_batches_image()
        plt.figure(figsize=(20, 4))
        for i, images in enumerate(train_dataset.take(n)):  # Take 1 batch for visualization
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(images[i].numpy())
            plt.title("original")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
            ax = plt.subplot(2, n, i + 1 + n)
            mean, logvar = self.model.encode(images)
            z = self.model.reparameterize(mean, logvar)
            predictions = self.model.sample(z)
            
            
            plt.imshow(predictions[i][:, :, 0])
            plt.title("reconstructed")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

class AE(tf.keras.Model):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = self.encoder_build()
        self.decoder = self.decoder_build()
        
    def encoder_build(self):
        encoder = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),strides=2, activation='relu',padding='same', input_shape=(96, 96, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),strides=2, activation='relu',padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1),strides=1, activation='relu',padding='same'),
            tf.keras.layers.BatchNormalization(),
        ])
        #encoder.summary()
        return encoder
    
    def decoder_build(self):
        decoder = tf.keras.models.Sequential([
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3),strides=2, activation='relu',padding='same', input_shape=(24, 24, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3),strides=2, activation='relu',padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=1,strides=1, activation='relu'),
        ])
        #decoder.summary()
        return decoder
    
    def encode(self, x):
        return self.encoder(x)
    
class TrainAE():
    def __init__(self):
        self.data = DataLoader()
        self.model = AE()
        """
        self.sweep_config = {
                'method': 'grid',
                'metric': {
                    'name': 'val_loss',
                    'goal': 'minimize'
                },
                'early_terminate':{
                    'type': 'hyperband',
                    'min_iter': 5
                },  
                'parameters': {
                    'learning_rate':{
                        'values': [0.01, 0.0001]
                        }
                }
            }
        self.sweep_id = wandb.sweep(self.sweep_config)
        """
        
    def run(self, epochs = 1):
        """
        config_defaults = {
            "learning_rate": 0.01
        }
        wandb.init(config=config_defaults,project="Train AE")
        wandb.config.epochs = epochs
        """
        train_dataset,val_dataset,_ = self.data.create_batches_image_image()
        autoencoder = tf.keras.models.Sequential([self.model.encoder, self.model.decoder], name='autoencoder')
        autoencoder.compile(loss='mse', optimizer='adam')
        #autoencoder.summary()
        autoencoder.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
        """
        autoencoder.fit(train_dataset, validation_data=val_dataset, epochs=wandb.config.epochs,
                         callbacks=[WandbCallback(input_type="image")
                        ])
        wandb.finish()
        """
        

    def train(self, epochs = 2):
        wandb.agent(self.sweep_id, function=lambda: self.run(epochs))    
        
    def latent_space(self):
        train_dataset,_,_ = self.data.create_batches_image()
        for train_x in train_dataset:
            return self.model.encoder(train_x)
        
        
    def plot_latent_space(self,n):
        train_dataset,_,_ = self.data.create_batches_image()
        plt.figure(figsize=(20, 4))
        for i, images in enumerate(train_dataset.take(n)):  # Take 1 batch for visualization
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(images[i].numpy())
            plt.title("original")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
            ax = plt.subplot(2, n, i + 1 + n)
            image = self.latent_space()
            plt.imshow(image[i][:, :, 0],cmap='gray')
            plt.title("reconstructed")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
    
    def encode_decode(self, image):
        encode = self.model.encoder.predict(image, verbose = 0)
        decode = self.model.decoder.predict(encode, verbose = 0)
        return decode

    def plot(self,n):
        train_dataset,_,_ = self.data.create_batches_image()
        plt.figure(figsize=(20, 4))
        for i, images in enumerate(train_dataset.take(n)):  # Take 1 batch for visualization
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(images[i].numpy())
            plt.title("original")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
            ax = plt.subplot(2, n, i + 1 + n)
            image = self.encode_decode(images)
            plt.imshow(image[i][:, :, 0])
            plt.title("reconstructed")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
        
class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = self.model_build()
        
    def model_build(self):
        input_img = Input(shape=(24,24,1))
                
        x = Conv2D(16, (3, 3), padding='valid')(input_img)
        x = BatchNormalization()(x)  # Add BatchNormalization after Conv2D
        x = Activation('relu')(x)

        x = Conv2D(32, (3, 3), padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(64, (3, 3), padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
                
        x = Conv2D(64, (3, 3), padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(rate=0.2)(x)
        y = Dense(2, activation='softmax')(x)
        model = Model(inputs=input_img, outputs=y)
        return model
    
    def step(self):
        return 

class TrainCNN():
    def __init__(self, image_model = "resize", image_type = "resize"):
        self.data = DataLoader()
        self.model = CNN()
        self.image_model = image_model
        self.image_type = image_type

        """
        self.sweep_config = {
                'method': 'grid',
                'metric': {
                    'name': 'val_loss',
                    'goal': 'minimize'
                },
                'early_terminate':{
                    'type': 'hyperband',
                    'min_iter': 5
                },  
                'parameters': {
                    'learning_rate':{
                        'values': [0.01, 0.0001]
                        }
                }
            }
        self.sweep_id = wandb.sweep(self.sweep_config)
        """
    def run(self, epochs = 1):
        """
        config_defaults = {
            "learning_rate": 0.01
        }
        wandb.init(config=config_defaults,project="Train CNN")
        wandb.config.epochs = epochs
        """
        
        train_dataset,_,val_dataset = self.get_data()
        #sgd_opt = SGD(learning_rate=wandb.config.learning_rate, momentum=0.9, nesterov=True)
        
        sgd_opt = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        model = self.model.model_build()
        model.compile(optimizer=sgd_opt,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        hist = model.fit(train_dataset, validation_data=val_dataset, epochs = epochs)
                       
        #hist = model.fit(train_dataset, validation_data=val_dataset, epochs=wandb.config.epochs,
        #                 callbacks=[WandbCallback(input_type="image")])
        #wandb.finish()
        
    def train(self, epochs = 2):
        wandb.agent(self.sweep_id, function=lambda: self.run(epochs))
    
    def get_data(self):
        train_dataset_raw,test_dataset_raw,val_dataset_raw = self.data.create_batches_image_label()
        if self.image_type == "resize":
            def preprocess_data(image, label):
                resized_image = tf.image.resize(image, (24, 24))
                # Convert the image to grayscale (1 channel)
                grayscale_image = tf.image.rgb_to_grayscale(resized_image)
                return grayscale_image, label
            # Apply the preprocessing function to the dataset
            train_dataset = train_dataset_raw.map(preprocess_data)
            test_dataset = test_dataset_raw.map(preprocess_data)
            val_dataset = val_dataset_raw.map(preprocess_data)
        elif self.image_type == "AE":
            def preprocess_data_AE(image, label):
                return self.image_model.model.encode(image),label
            train_dataset = train_dataset_raw.map(preprocess_data_AE)
            test_dataset = test_dataset_raw.map(preprocess_data_AE)
            val_dataset = val_dataset_raw.map(preprocess_data_AE)
        elif self.image_type == "VAE_encode":
            def preprocess_data_VAE_encode(image, label):
                return self.image_model.model.encode(image),label
            train_dataset = train_dataset_raw.map(preprocess_data_VAE_encode)
            test_dataset = test_dataset_raw.map(preprocess_data_VAE_encode)
            val_dataset = val_dataset_raw.map(preprocess_data_VAE_encode)
        elif self.image_type == "VAE_decode":
            def preprocess_data_VAE_decode(image, label):
                encoded = self.image_model.model.encode(image)
                mean, logvar = self.image_model.model.latent_space(encoded)
                z = tf.exp(logvar * 0.5) + mean
                z_prime = self.image_model.model.decode_first_layer(z)
                return z_prime,label
            train_dataset = train_dataset_raw.map(preprocess_data_VAE_decode)
            test_dataset = test_dataset_raw.map(preprocess_data_VAE_decode)
            val_dataset = val_dataset_raw.map(preprocess_data_VAE_decode)
        else:
            print("WRONG INPUT TYPE FOR image_type")
            #quit()
        return  train_dataset, test_dataset ,val_dataset 