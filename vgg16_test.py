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

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

inputs = keras.Input(shape=(img_size + (4,)))
x = inputs
x = layers.Conv2D(64, 3, activation='relu', padding="same")(x)
x = layers.Conv2D(64, 3, activation='relu', padding="same")(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(128, 3, activation='relu', padding="same")(x)
x = layers.Conv2D(128, 3, activation='relu', padding="same")(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(256, 3, activation='relu', padding="same")(x)
x = layers.Conv2D(256, 3, activation='relu', padding="same")(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(512, 3, activation='relu', padding="same")(x)
x = layers.Conv2D(512, 3, activation='relu', padding="same")(x)
#x = layers.MaxPooling2D(2)(x)

x = layers.Flatten()(x)
x = layers.Dense(512)(x)
x = layers.Dense(512)(x)
x = layers.Dense(3, activation='softmax')(x)
outputs = x

model = keras.Model(inputs, outputs)
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
epochs = 15
history = model.fit_generator(trainGen,
                              epochs=epochs,
                              steps_per_epoch=trainGen.samples / epochs * 5,
                              validation_data=valGen,
                              validation_steps=valGen.samples / epochs)

print("\n Valid Accuracy: %.4f" % (model.evaluate_generator(valGen)[1]))
print("\n Test Accuracy: %.4f" % (model.evaluate_generator(testGen)[1]))

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['acc'])
# history = model.fit(x_train, y_train, epochs=5)
# model.evaluate(x_test,  y_test, verbose=2)

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

model.save('collars_model_vgg16_2.h5')