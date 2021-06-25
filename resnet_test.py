import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

rootPath = 'dataset/collars'
img_size = (125, 150)

imageGenerator = ImageDataGenerator(
    rescale=1. / 255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[.2, .2],
    horizontal_flip=True,
)

trainGen = imageGenerator.flow_from_directory(os.path.join(rootPath, 'train'), class_mode='categorical',
                                              target_size=img_size, shuffle=True,
                                              color_mode='rgba')

valGenerator = ImageDataGenerator(rescale=1. / 255)

valGen = valGenerator.flow_from_directory(os.path.join(rootPath, 'val'), class_mode='categorical',
                                          target_size=img_size, shuffle=True,
                                          color_mode='rgba')

testGenerator = ImageDataGenerator(rescale=1. / 255)

testGen = testGenerator.flow_from_directory(os.path.join(rootPath, 'test'), class_mode='categorical',
                                            target_size=img_size, shuffle=True,
                                            color_mode='rgba')

inputs = keras.Input(shape=(125, 150, 4))
x = inputs
#x = layers.MaxPooling2D(2)(x)
_x = layers.Conv2D(128, 3, activation='relu', padding="same")(x)
_x = layers.Conv2D(128, 3, activation='relu', padding="same")(_x)
x = _x
_x = layers.Conv2D(128, 3, activation='relu', padding="same")(x)
_x = layers.Conv2D(128, 3, activation='relu', padding="same")(_x)
x = x + _x
x = layers.MaxPooling2D(2)(x)
_x = layers.Conv2D(128, 3, activation='relu', padding="same")(x)
_x = layers.Conv2D(128, 3, activation='relu', padding="same")(_x)
x = x + _x
x = layers.MaxPooling2D(2)(x)
_x = layers.Conv2D(128, 3, activation='relu', padding="same")(x)
_x = layers.Conv2D(128, 3, activation='relu', padding="same")(_x)
x = x + _x
x = layers.MaxPooling2D(2)(x)

x = layers.Flatten()(x)
x = layers.Dense(128)(x)
x = layers.Dense(128)(x)
x = layers.Dense(3, activation='softmax')(x)
outputs = x

model = keras.Model(inputs, outputs)
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
epochs = 30
history = model.fit_generator(trainGen,
                              epochs=epochs,
                              steps_per_epoch=trainGen.samples / epochs * 5,
                              validation_data=valGen,
                              validation_steps=valGen.samples / epochs)

print("\n Valid Accuracy: %.4f" % (model.evaluate_generator(valGen)[1]))
print("\n Test Accuracy: %.4f" % (model.evaluate_generator(testGen)[1]))

# graph
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'go', label='Training Loss')
plt.plot(epochs, val_loss, 'g', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

model.save('collars_model_resnet.h5')