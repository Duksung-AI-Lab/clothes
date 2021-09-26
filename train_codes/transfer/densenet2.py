import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, optimizers, Model, Input, losses
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import densenet

tf.disable_v2_behavior()

np.random.seed(0)
tf.set_random_seed(0)

data_path = 'dataset/collar/collars_500x600/'
image_size = (150, 150)

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

inputs = Input(shape=(150, 150, 3))
# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning, as you will
# learn in a few paragraphs.
x = transfer_model(inputs, training=False)
# Convert features of shape `base_model.output_shape[1:]` to vectors
x = GlobalAveragePooling2D()(x)
# A Dense classifier with a single unit (binary classification)
outputs = Dense(3)(x)
model = Model(inputs, outputs)

# model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
#               metrics=["accuracy"])

model.compile(optimizer=optimizers.Adam(),
              loss=losses.BinaryCrossentropy(from_logits=True),
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
