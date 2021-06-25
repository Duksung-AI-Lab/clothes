import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import os

# 랜덤시드 고정시키기
#np.random.seed(4)

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

train_generator = train_datagen.flow_from_directory(
        '/gdrive/My Drive/Colab Notebooks/dataset/shirts_collar/train_shirts',
        shuffle=True,
        target_size=(32, 20),
        batch_size=5,
        class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        '/gdrive/My Drive/Colab Notebooks/dataset/shirts_collar/test',
        shuffle = True,
        target_size=(32, 20),
        batch_size=5,
        class_mode='categorical')

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

val_generator = val_datagen.flow_from_directory(
        '/gdrive/My Drive/Colab Notebooks/dataset/shirts_collar/train_shirts',
        shuffle=True,
        target_size=(32, 20),
        batch_size=5,
        class_mode='categorical')

# predict_datagen = ImageDataGenerator(rescale=1./255)

# predict_generator = test_datagen.flow_from_directory(
#         '/gdrive/My Drive/Colab Notebooks/dataset/shirts_collar/predict',
#         target_size=(32, 20),
#         batch_size=3,
#         class_mode='categorical')

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(32,20,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))#
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))#
model.add(Dense(128, activation='relu'))#
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='hinge', optimizer='adam', metrics=['accuracy'])

# MODEL_DIR = '/gdrive/My Drive/Colab Notebooks/models/'
# if not os.path.exists(MODEL_DIR):
#   os.mkdir(MODEL_DIR)

# modelpath="/gdrive/My Drive/Colab Notebooks/models/{epoch:02d}-{val_loss:.4f}.hdf5"

# checkpointer = ModelCheckpoint(filepath=modelpath,monitor='val_loss',verbose=1)

# history=model.fit_generator(
#         train_generator,
#         steps_per_epoch=train_generator.samples/3,
#         epochs=30,
#         validation_data=val_generator,
#         validation_steps=3,
#         callbacks=[checkpointer])

history=model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples/5,
        epochs=30,
        validation_data=val_generator,
        validation_steps=10)

print("-- Evaluate --")
scores = model.evaluate_generator(test_generator, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

model.save('/gdrive/My Drive/Colab Notebooks/dataset/shirts_collar/collar_type_ff.h5')

# print("-- Predict --")
# output = model.predict_generator(predict_generator, steps=5)
# np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
# print(test_generator.class_indices)
# print(output)

print(history.history)

#훈련 과정 시각화 (정확도)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Full Image Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

#훈련 과정 시각화 (손실)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()