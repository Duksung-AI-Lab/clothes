import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, optimizers
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import densenet
from keras.callbacks import EarlyStopping

tf.disable_v2_behavior()

np.random.seed(0)
tf.set_random_seed(0)

# # GPU
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     # 텐서플로가 세 번째 GPU만 사용하도록 제한
#     try:
#         tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
#     except RuntimeError as e:
#         # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
#         print(e)

# 이미지 불러오기
data_path = 'pattern_5class'

image_size = (224, 224)

train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True,
                                   width_shift_range=0.1, height_shift_range=0.1,
                                   fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(os.path.join(data_path, 'train'), shuffle=True,
                                                    target_size=image_size, class_mode='categorical')

val_datagen = ImageDataGenerator(rescale=1. / 255, fill_mode='nearest')
val_generator = val_datagen.flow_from_directory(os.path.join(data_path, 'val'), shuffle=True,
                                                target_size=image_size, class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1. / 255, fill_mode='nearest')
test_generator = val_datagen.flow_from_directory(os.path.join(data_path, 'test'), shuffle=True,
                                                 target_size=image_size, class_mode='categorical')

# 모델 생성
transfer_model = densenet.DenseNet121(weights='imagenet', include_top=False, input_shape=image_size + (3,))
transfer_model.trainable = False
transfer_model.summary()

model = models.Sequential()
model.add(transfer_model)
model.add(GlobalAveragePooling2D())
# model.add(Dropout(0.2))  # Regularize with dropout
model.add(Dense(5, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0002),
              metrics=['acc'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10)
epochs = 20
history = model.fit_generator(train_generator,
                              epochs=epochs,
                              steps_per_epoch=train_generator.samples // epochs * 5,
                              validation_data=val_generator,
                              validation_steps=train_generator.samples // epochs,
                              callbacks = [early_stopping])

# 그래프
acc = history.history['acc']
val_acc = history.history['val_acc']
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, y_loss, 'go', label='Training Loss')
plt.plot(epochs, y_vloss, 'g', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# fine tuning
transfer_model.trainable = True

# for layer in transfer_model.layers[:53]:
#     layer.trainable = False

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-5), metrics=['acc'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
epochs = 20
history = model.fit_generator(train_generator,
                              epochs=epochs,
                              steps_per_epoch=train_generator.samples // epochs * 5,
                              validation_data=val_generator,
                              validation_steps=train_generator.samples // epochs,
                              callbacks = [early_stopping])

# 모델 저장
model.save('model/pattern_5class.h5')

# 테스트 결과
print("\n Valid Accuracy: %.4f" % (model.evaluate_generator(val_generator)[1]))
print("\n Test Accuracy: %.4f" % (model.evaluate_generator(test_generator)[1]))

# 그래프
acc = history.history['acc']
val_acc = history.history['val_acc']
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, y_loss, 'go', label='Training Loss')
plt.plot(epochs, y_vloss, 'g', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
