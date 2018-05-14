import os
import csv
import cv2
import numpy as np 
import sklearn
from random import shuffle
from numpy import zeros, newaxis
from sklearn.model_selection import train_test_split

test_split = 0.25
samples = [] 
images = []
angles = []
cam_delta_left = 0.08 # 0.15 # 0.25 0.08
cam_delta_right = 0.08
nb_epoch = 2
batch_size = 32
model_name = "7-san-test.h5"


with open('./data/project_dataset_log.csv') as csvfile :
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
shuffle(samples)
        
## Prepare training / validation samples
train_samples, validation_samples = train_test_split(samples, test_size=test_split)


def load_image(image_path):
    """preproccesing training data to keep only S channel in HSV color space, and resize to 16X32"""
    img = cv2.imread(image_path)
    return img

def gray_conversion(x):
    import tensorflow as tf
    return tf.image.rgb_to_grayscale(x)

def get_current_path(path):
    path_elem = './data/sample_training_data/'+line[0]
    image_path = './'+'/'.join(path_elem.split('/')[-4:])
    return image_path


def load_in_memory():
    for line in samples :
        # center,left,right,steering,throttle,brake,speed
        center_image = load_image(get_current_path(line[0]))
        center_angle = float(line[3])
        images.append(center_image)
        angles.append(center_angle)
        
        ## Augment dataset by adding flipped image and its measurement
        ## Center & flipped
        center_image_flipped = np.fliplr(center_image)    
        center_angle_flipped = center_angle*(-1)
        images.append(center_image_flipped)
        angles.append(center_angle_flipped)
        
        ## Left & flipped
        '''
        left_image = load_image(get_current_path(line[1]))
        left_angle = center_angle + cam_delta_left
        images.append(left_image)
        angles.append(left_angle)        
        
        left_image_flipped = np.fliplr(left_image)    
        left_angle_flipped = left_angle*(-1)
        images.append(left_image_flipped)
        angles.append(left_angle_flipped)

        ## Right & flipped
        right_image = load_image(get_current_path(line[2]))
        right_angle = center_angle - cam_delta_right
        images.append(right_image)
        angles.append(right_angle)

        right_image_flipped = np.fliplr(right_image)    
        right_angle_flipped = right_angle*(-1)
        images.append(right_image_flipped)
        angles.append(right_angle_flipped)
        '''

    X_train = np.array(images)
    y_train = np.array(angles)
    
    return X_train,y_train


## Data Generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for line in batch_samples:
                center_image = load_image(get_current_path(line[0]))
                center_angle = float(line[3])
                images.append(center_image)
                angles.append(center_angle)

                ## Augment dataset by adding flipped image and its measurement
                ## Center & flipped
                #center_image_flipped = np.fliplr(center_image)    
                #center_angle_flipped = center_angle*(-1)
                #images.append(center_image_flipped)
                #angles.append(center_angle_flipped)

                ## Left & flipped
                left_image = load_image(get_current_path(line[1]))
                left_angle = center_angle + cam_delta
                images.append(left_image)
                angles.append(left_angle)        

                #left_image_flipped = np.fliplr(left_image)    
                #left_angle_flipped = left_angle*(-1)
                #images.append(left_image_flipped)
                #angles.append(left_angle_flipped)

                ## Right & flipped
                right_image = load_image(get_current_path(line[2]))
                right_angle = center_angle - cam_delta
                images.append(right_image)
                angles.append(right_angle)

                #right_image_flipped = np.fliplr(right_image)    
                #right_angle_flipped = right_angle*(-1)
                #images.append(right_image_flipped)
                #angles.append(right_angle_flipped)
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Activation
from keras.layers.convolutional import Conv2D, Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2, activity_l2


model = Sequential()
model.add(Lambda(lambda x : x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6, 5, 5, border_mode="valid", input_shape=(160,320,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, border_mode="valid", activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, border_mode="valid", activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, border_mode="valid", activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(120))
model.add(Dropout(0.2))
model.add(Dense(84))
model.add(Dense(1))
model.summary()


'''
model = Sequential()
model.add(Lambda(lambda x : (x/255.) - 0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))
#model.add(Lambda(gray_conversion))
#model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(160,320,3)))
#, output_shape=(160,320,1)))
#model.add(Lambda(lambda x: to_gray(x), input_shape=(160,320,1))
#model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
#model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(160,320,1)))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dropout(0.1))
model.add(Dense(120))
model.add(Dropout(0.05))
model.add(Dense(84))
#model.add(Dropout(0.25))
model.add(Dense(1))
model.summary()
'''



'''
# The nVidia CNN Architecture
model = Sequential()
# Crop
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160, 320, 3)))
# Normalize
model.add(Lambda(lambda x: x / 255.0 - 0.5))
# Convolution
model.add(Convolution2D(24,5,5, activation='relu'))
model.add(Convolution2D(36,5,5, activation='relu'))
model.add(Convolution2D(48,5,5, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
# Flatten
model.add(Flatten())
# Dense
model.add(Dropout(0.1))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dropout(0.1))
model.add(Dense(10))
model.add(Dense(1))
model.summary()
'''

model.compile(loss='mse', optimizer='adam')
#model.compile(optimizer=Adam(lr=1e-4), loss='mse')

X_train,y_train = load_in_memory()
model.fit(X_train, y_train, validation_split=test_split, shuffle=True, nb_epoch=nb_epoch)

'''
model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples), validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=nb_epoch)
'''
model.save('./models/{}'.format(model_name))

print("model ./models/{} save done".format(model_name))
