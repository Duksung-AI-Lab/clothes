import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

np.random.seed(3)
tf.compat.v1.set_random_seed(3)

data_path = 'dataset/collar/collars_500x600(2368)_jpg/'  # 경로 지정

image_size = (150, 150)

train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True,
                                   width_shift_range=0.1, height_shift_range=0.1,
                                   fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(os.path.join(data_path, 'train'), shuffle=True,
                                                    target_size=image_size, class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1. / 255, fill_mode='nearest')
val_generator = val_datagen.flow_from_directory(os.path.join(data_path, 'val'), shuffle=True,
                                                target_size=image_size, class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1. / 255, fill_mode='nearest')
test_generator = val_datagen.flow_from_directory(os.path.join(data_path, 'test'), shuffle=True,
                                                 target_size=image_size, class_mode='binary')

transfer_model = VGG16(weights='imagenet', include_top=False, input_shape=image_size + (3,))
transfer_model.trainable = False
model = models.Sequential()
model.add(transfer_model)
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

epochs = 10
history = model.fit_generator(train_generator,
                              epochs=epochs,
                              steps_per_epoch=train_generator.samples // epochs * 5,
                              validation_data=val_generator,
                              validation_steps=val_generator.samples // epochs)

print("\n Test Accuracy: %.4f" % (model.evaluate_generator(test_generator)[1]))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, acc, marker='.', c="red", label='Train_acc')
plt.plot(x_len, val_acc, marker='.', c="blue", label='Test_acc')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss/acc')
plt.show()
