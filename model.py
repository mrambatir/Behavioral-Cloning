import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import math
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D



# DATA PROCESSING*********************************************************************
    # reading and storing the driving log file 
df = pd.read_csv('./data/driving_log.csv')

    # Steering data is heavily centered at 0 (see histogram in write-up report); not good for training 
    # Drop 80% of Steering angles with value = 0
df = df.drop(df[df['steering'] == 0].sample(frac=0.8).index)

    # Image paths for center, left and right images are concatenated
images_path = pd.concat([df['center'], df['left'], df['right']])
images_path = np.array(images_path.values.tolist())

    # steering correction of +/- 0.25 is set for left and right camera images as compared to center
correction = 0.25
steering_c = df['steering']
steering_l = df['steering'] + correction
steering_r = df['steering'] - correction

    # measurements for center, left and right images are concatenated
measurements = pd.concat([steering_c, steering_l, steering_r])
measurements = np.array(measurements.values.tolist())


    # the next function generates X and y data
    # First few lines are to get the path of the image 
    # read the image using cv2 and converting to RGB format
    # appending image and measurement to X and y separately
    # Flippling the image (left to right) and the relevant measurement changes sign; appending to the arrays
    # Flip left to right is the data augmentation step in addition to using all camera images
    

def generate_data(images_path, measurements, batch_size=64):
    images_path, measurements = shuffle(images_path, measurements)
    X,y = ([],[])
    while 1: 
                   
        for i in range(len(measurements)):
            source_path = images_path[i]
            filename = source_path.split('/')[-1]
            current_path = './data/IMG/' + filename
            img = cv2.imread(current_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
           
            
            measurement = measurements[i]
            X.append(img)
            y.append(measurement)
            if len(X) == batch_size:
                yield (np.array(X), np.array(y))
                X, y = ([],[])
                images_path, measurements = shuffle(images_path, measurements)
            img = cv2.flip(img,1)
            measurement = measurements[i]*-1.0
            X.append(img)
            y.append(measurement)
            if len(X) == batch_size:
                yield (np.array(X), np.array(y))
                X, y = ([],[])
                images_path, measurements = shuffle(images_path, measurements)

'''
For huge data sets, we can use a python generator (as here) to generate data for training rather than storing the entire 
training data in memory.Particularly useful when the training data is very huge and/or if there are multiple augmentation techniques. I used only image flip here 
'''

                
# splitting into training and validation sets
images_path_train, images_path_valid, measurements_train, measurements_valid = train_test_split(images_path, measurements, test_size=0.2)

# Generating training and validation using "generate data" function
train_generator = generate_data(images_path_train, measurements_train, batch_size=64)
validation_generator = generate_data(images_path_valid, measurements_valid, batch_size=64)


# Model Architecture and training using Keras and generator**************************************************************

model = Sequential()
    # Image normailzation with range and mean
model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=(160,320,3)))
    # Image cropping with the region of interest
model.add(Cropping2D(cropping=((70,25), (0,0))))

    # Five convolution layers are added with filter sizes, strides and elu activation (as per the write-up)
    # A fixed rate 'r' is defined as drop-out rate for all
r = 0.12

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='elu'))
model.add(Dropout(r))

model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu'))
model.add(Dropout(r))

model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='elu'))
model.add(Dropout(r))
    
    # Last two 3x3 convolution layers (output depth 64, and 64)
model.add(Convolution2D(64, 3, 3, activation='elu'))
model.add(Dropout(r))
model.add(Convolution2D(64, 3, 3, activation='elu'))
model.add(Dropout(r))    
    # Add a flatten layer
model.add(Flatten())

    # Add three FC layers (depth 100, 50, 10), elu activation (and dropouts)
model.add(Dense(100))
model.add(Activation('elu'))
model.add(Dropout(r))

model.add(Dense(50))
model.add(Activation('elu'))
model.add(Dropout(r))

model.add(Dense(10))
model.add(Activation('elu'))
    #No drop-out fot the last FC layer

    # Output layer, steering angle
model.add(Dense(1))

    # Compile and train the model, shuffle the data with 20% of total data to validation set 

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, samples_per_epoch = len(measurements_train), validation_data=validation_generator, nb_val_samples=len(measurements_valid), nb_epoch=10) 

model.save('model.h5')
