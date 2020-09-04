# import the libraries
import keras
from keras.layers import Input , Lambda , Dense , Flatten
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

IMAGE_SIZE = [224 , 224]

train_path = "Datasets/Train"
test_path = "Datasets/Test"

# preprocessing layer using the VGGnet
vgg = VGG16(input_shape=IMAGE_SIZE + [3] , weights = 'imagenet' , include_top = False)


# here not to train the ecisting weights

for layer in vgg.layers:
    layer.trainable = False
    
    
folders = glob("Datasets/Train/*")  # to check the no of folders inside the datasets


# to add the extra layers to make the output layer accurate

x = Flatten()(vgg.output)
prediction = Dense(len(folders) , activation = 'softmax')(x)

# to create a model
model = Model(inputs = vgg.input , outputs= prediction)
model.summary()

 
# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics = ['accuracy'])


# ot check the no of images in the train and test datasets

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale= 1./255)

training_set =train_datagen.flow_from_directory('Datasets/Train',
                                                target_size=(224,224),
                                                batch_size=32,
                                                class_mode='categorical')
test_set =train_datagen.flow_from_directory('Datasets/Test',
                                                target_size=(224,224),
                                                batch_size=32,
                                                class_mode='categorical')




# fit the model

r = model.fit_generator(
        train_path,
        validation_data=test_set,
        epochs = 5,
        steps_per_epoch=len(training_set),
        validation_steps=len(test_set)
        )




