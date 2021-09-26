import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

import tensorflow.keras.models as models
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dropout

import matplotlib.pyplot as plt

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
# import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # 텐서플로가 두 번째 GPU만 사용하도록 제한
    try:
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    except RuntimeError as e:
        # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
        print(e)

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)
# tf.random.set_seed(seed)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

rootPath = 'dataset/collars_v2_crop_2class'
img_size = (120, 90)

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

# x, y = trainGen.next()
# print(x[0].shape)
# plt.imshow(x[0])
# plt.show()

model = models.Sequential()

model.add(layers.InputLayer(input_shape=(img_size + (4,))))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

epochs = 15
# history = model.fit_generator(trainGen,
#                               epochs=epochs,
#                               steps_per_epoch=trainGen.samples / epochs * 5,
#                               validation_data=valGen,
#                               validation_steps=valGen.samples / epochs,
#                               callbacks=[early_stopping])

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

model.save('models/collars_model.h5')
