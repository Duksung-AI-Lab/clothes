import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, optimizers, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import densenet

tf.disable_v2_behavior()

np.random.seed(0)
tf.set_random_seed(0)

data_path = 'dataset/collar/collars_500x600/'

image_size = (125, 150)

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   horizontal_flip=True,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   rotation_range=5,
                                   zoom_range=1.2,
                                   fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(os.path.join(data_path, 'train'), shuffle=True,
                                                    target_size=image_size, class_mode='categorical')

val_datagen = ImageDataGenerator(rescale=1. / 255, fill_mode='nearest')
val_generator = val_datagen.flow_from_directory(os.path.join(data_path, 'val'), shuffle=True,
                                                target_size=image_size, class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1. / 255, fill_mode='nearest')
test_generator = val_datagen.flow_from_directory(os.path.join(data_path, 'test'), shuffle=True,
                                                 target_size=image_size, class_mode='categorical')

transfer_model = densenet.DenseNet121(weights='imagenet', include_top=False, input_shape=image_size + (3,))

transfer_model.trainable = False
model = models.Sequential()
model.add(transfer_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(),
              metrics=['acc'])

epochs = 20
history = model.fit_generator(train_generator,
                              epochs=epochs,
                              steps_per_epoch=train_generator.samples // epochs * 5,
                              validation_data=val_generator,
                              validation_steps=val_generator.samples // epochs)

print("\n Test Accuracy: %.4f" % (model.evaluate_generator(test_generator)[1]))

acc = history.history['acc']
val_acc = history.history['val_acc']
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
