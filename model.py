import csv
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
##
import tensorflow as tf
tf.python.control_flow_ops = tf
from keras.models import Sequential
from keras.layers import Cropping2D, Lambda, Dropout
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras import optimizers



# Read the driving log file
def get_samples():
    lines = []
    with open('data/driving_log.csv') as csvfile:
        reader=csv.reader(csvfile)
        for line in reader:
            lines.append(line)        
    # pop out the first row in the csv file
    #lines.pop(0)
    # split the samples into training set and validation set
    shuffle(lines)
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    return train_samples, validation_samples


# Data Processing 
def preprocess(image):
    #crop
    image_crop=image[45:140, :,:]
    #translate
    #dx=np.random.uniform(-1, 1)
    #dy=np.random.uniform(-2, 2)
    #M=np.float32([[1,0,dx],[0,1,dy]])
    #image_crop=cv2.warpAffine(image_crop, M, (image_crop.shape[0], image_crop.shape[1]))
    #resize
    image_resize=cv2.resize(image_crop, (200, 66), cv2.INTER_AREA)
    # change color
    image_color= cv2.cvtColor(image_resize, cv2.COLOR_BGR2YUV)
    return image_color


# Get the images and steering values
def batch_generator(lines, batch_size):
    num_samples = len(lines)
    while 1: 
        for offset in range(0, num_samples, batch_size):
            shuffle(lines)
            batch_samples = lines[offset: offset+batch_size]
            center_images = []
            left_images = []
            right_images = []
            cam_images = []
            steering_angles = []
            for line in batch_samples:
                # get the file path
                center_source_path = line[0]
                left_source_path = line[1]
                right_source_path = line[2]
                steering_angle = float(line[3])
                # read the images
                center_filename = center_source_path.split('\\')[-1]
                left_filename = left_source_path.split('\\')[-1]
                right_filename = right_source_path.split('\\')[-1]
                center_current_path = 'data/IMG/'+ center_filename
                left_current_path = 'data/IMG/'+ left_filename
                right_current_path = 'data/IMG/'+ right_filename
                center_image = cv2.imread(center_current_path)
                left_image = cv2.imread(left_current_path)
                right_image = cv2.imread(right_current_path)
                # preprocess the image data
                center_image = preprocess(center_image)
                left_image = preprocess(left_image)
                right_image = preprocess(right_image)
                #center_images.append(center_image)
                #left_images.append(left_image)
                #right_images.append(right_image)
                cam_images.extend([center_image, left_image, right_image])
                # get the steering values
                correction = 0.25 
                steering_angles.extend([steering_angle, steering_angle+correction, steering_angle-correction])
                #steering_angles.append(steering_angle)
            # convert to numpy array    
            X_train = np.array(cam_images)
            y_train = np.array(steering_angles)
            yield shuffle(X_train, y_train)

    
#Main

# Build the model
model=Sequential()
# lambda layer: normalize image data
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66,200,3)))
# Conv layer: 5x5
model.add(Convolution2D(24, 5, 5, activation="elu", border_mode='valid', subsample=(2, 2)))
# Conv layer: 5x5
model.add(Convolution2D(36, 5, 5, activation="elu", border_mode='valid', subsample=(2, 2)))
# Conv layer: 5x5
model.add(Convolution2D(48, 5, 5, activation="elu", border_mode='valid', subsample=(2, 2)))
# Conv layer: 3x3
model.add(Convolution2D(64, 3, 3, activation="elu", border_mode='valid'))
# Conv layer: 3x3
model.add(Convolution2D(64, 3, 3, activation="elu", border_mode='valid'))
# Dropout
model.add(Dropout(0.5))
# Flatten
model.add(Flatten())
# Dense
model.add(Dense(100, activation="elu"))
# Dense
model.add(Dense(50, activation="elu"))
# Dense
model.add(Dense(10, activation="elu"))

# Dense
model.add(Dense(1))


# Train the model
batch_size=128
epoch=10
# Read image data
samples_train, samples_valid = get_samples()
# Generator
train_generator=batch_generator(samples_train, batch_size)
validation_generator = batch_generator(samples_valid, batch_size)
# Define the loss function and optimizer
model.compile(loss='mse', optimizer=optimizers.Adam(lr=1e-4))
model.fit_generator(train_generator, 
                    samples_per_epoch = 3*len(samples_train), 
                    validation_data= validation_generator,
                    nb_val_samples = 3*len(samples_valid),
                    nb_epoch=epoch)


# Save the model
model.save('model.h5')

    
