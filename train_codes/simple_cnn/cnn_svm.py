import os
from sklearn import svm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

import matplotlib.pyplot as plt

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.keras.preprocessing.image_dataset import image_dataset_from_directory

tf.disable_v2_behavior()
# import tensorflow as tf

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)
# tf.random.set_seed(seed)

rootPath = 'dataset/collars'
# rootPath = 'dataset/collars_crop'
img_size = (125, 150)
# img_size = (20, 30)

imageGenerator = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[.2, .2],
    horizontal_flip=True,
    validation_split=0.2
)

trainGen = imageGenerator.flow_from_directory(os.path.join(rootPath, 'train'), class_mode='categorical',
                                              target_size=img_size, shuffle=True, subset='training',
                                              color_mode='rgba')

testGenerator = ImageDataGenerator(rescale=1. / 255)

testGen = imageGenerator.flow_from_directory(os.path.join(rootPath, 'train'), class_mode='categorical',
                                             target_size=img_size, shuffle=True, subset='validation',
                                             color_mode='rgba')

x, y = trainGen.next()
print(x[0].shape)
plt.imshow(x[0])
plt.show()

model = Sequential()

model.add(layers.InputLayer(input_shape=(img_size + (4,))))
# model.add(layers.Conv2D(16, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(3, activation='sigmoid'))

sv = svm.SVC(kernel='rbf', C=1)
result = sv.fit(layers.Flatten())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

epochs = 10
history = model.fit_generator(trainGen,
                              epochs=epochs,
                              steps_per_epoch=trainGen.samples / epochs * 100,
                              validation_data=testGen,
                              validation_steps=testGen.samples / epochs * 100)

print("\n Accuracy: %.4f" % (model.evaluate(testGen)[1]))

model.save('collars_model.h5')
# model.save('collars_crop_model.h5')
