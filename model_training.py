# importing requisite libraries
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os 

# importing Keras libraries
from keras.models import Sequential
from keras.layers import Activation, Dropout, UpSampling2D
from keras.layers import Deconvolution2D, Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

# data exploration
# loading train and label images
train_images = os.listdir("/Users/abhilashalodha/Documents/LaneDetection/Images")
label_images = os.listdir("/Users/abhilashalodha/Documents/LaneDetection/Labels")
    
# converting image data to array
train_images = np.array(train_images)
labels = np.array(label_images)

# normalizing labels
labels = labels / 255

# shuffling images and their corresponding labels
train_images, labels = shuffle(train_images, labels)

# splitting data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_images, labels, test_size=0.1)

# hyperparameters for model optimisation
batch_size = 50
epochs = 50
pool_size = (2, 2)
input_shape = X_train.shape[1:]

# model definition
model = Sequential()

# normalizing incoming inputs
model.add(BatchNormalization(input_shape=input_shape))
model.add(Convolution2D(60, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', name = 'Conv1'))
model.add(Convolution2D(50, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', name = 'Conv2'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Convolution2D(40, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', name = 'Conv3'))
model.add(Dropout(0.2))
model.add(Convolution2D(30, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', name = 'Conv4'))
model.add(Dropout(0.2))
model.add(Convolution2D(20, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', name = 'Conv5'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Convolution2D(10, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', name = 'Conv6'))
model.add(Dropout(0.2))
model.add(Convolution2D(5, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', name = 'Conv7'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(UpSampling2D(size=pool_size))
model.add(Deconvolution2D(10, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', 
                          output_shape = model.layers[8].output_shape, name = 'Deconv1'))
model.add(Dropout(0.2))
model.add(Deconvolution2D(20, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', 
                          output_shape = model.layers[7].output_shape, name = 'Deconv2'))
model.add(Dropout(0.2))
model.add(UpSampling2D(size=pool_size))
model.add(Deconvolution2D(30, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', 
                          output_shape = model.layers[5].output_shape, name = 'Deconv3'))
model.add(Dropout(0.2))
model.add(Deconvolution2D(40, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', 
                          output_shape = model.layers[4].output_shape, name = 'Deconv4'))
model.add(Dropout(0.2))
model.add(Deconvolution2D(50, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', 
                          output_shape = model.layers[3].output_shape, name = 'Deconv5'))
model.add(Dropout(0.2))
model.add(UpSampling2D(size=pool_size))
model.add(Deconvolution2D(60, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', 
                          output_shape = model.layers[1].output_shape, name = 'Deconv6'))

# final layer - only including one channel so 1 filter
model.add(Deconvolution2D(1, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', 
                          output_shape = model.layers[0].output_shape, name = 'Final'))


datagen = ImageDataGenerator()
datagen.fit(X_train)

# compiling and training the model
model.compile(optimizer='Adam', loss='mean_squared_error')
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), samples_per_epoch = len(X_train),
                    nb_epoch=epochs, verbose=1, validation_data=(X_val, y_val))

# save model architecture and weights
model_json = model.to_json()
with open("full_CNN_model3.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights('full_CNN_model3.h5')

# Show summary of model
model.summary()

