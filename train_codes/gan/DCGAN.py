from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU, UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

np.random.seed(3)
tf.random.set_seed(3)

# 생성자 모델을 만듭니다.
dropout = 0.4
depth = 64 + 64 + 64 + 64
dim = 8
generator = Sequential()
generator.add(Dense(dim * dim * depth, input_dim=100))
generator.add(BatchNormalization(momentum=0.9))
generator.add(LeakyReLU(alpha=0.2))
generator.add(Reshape((dim, dim, depth)))
generator.add(Dropout(dropout))
generator.add(UpSampling2D())
generator.add(Conv2DTranspose(int(depth / 2), 5, padding='same'))
generator.add(BatchNormalization(momentum=0.9))
generator.add(LeakyReLU(alpha=0.2))
generator.add(UpSampling2D())
generator.add(Conv2DTranspose(int(depth / 4), 5, padding='same'))
generator.add(BatchNormalization(momentum=0.9))
generator.add(LeakyReLU(alpha=0.2))
generator.add(Conv2DTranspose(int(depth / 8), 5, padding='same'))
generator.add(BatchNormalization(momentum=0.9))
generator.add(LeakyReLU(alpha=0.2))
generator.add(Conv2DTranspose(3, 5, padding='same'))
generator.add(Activation('tanh'))
generator.summary()

# 판별자 모델을 만듭니다.
depth = 64
dropout = 0.4
discriminator = Sequential()
discriminator.add(Conv2D(depth * 1, 5, strides=2, input_shape=(32, 32, 3), padding='same'))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(dropout))
discriminator.add(Conv2D(depth * 2, 5, strides=2, padding='same'))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(dropout))
discriminator.add(Conv2D(depth * 4, 5, strides=2, padding='same'))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(dropout))
discriminator.add(Conv2D(depth * 8, 5, strides=1, padding='same'))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(dropout))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.summary()
discriminator.compile(loss='binary_crossentropy', optimizer="adam")
discriminator.trainable = False

# 생성자와 판별자 모델을 연결시키는 gan 모델을 만듭니다.
ginput = Input(shape=(100,))
dis_output = discriminator(generator(ginput))
gan = Model(ginput, dis_output)
gan.compile(loss='binary_crossentropy', optimizer="adam")
gan.summary()


# 신경망을 실행시키는 함수를 만듭니다.
def gan_train(epoch, batch_size, saving_interval):
    d_loss_memo = []
    g_loss_memo = []

    # 데이터 불러오기
    ### 경로지정
    X_train = []
    images = os.listdir("/content/gdrive/MyDrive/Study/professional_dataset/collar_32X32")
    for path in images:
        img = Image.open("/content/gdrive/MyDrive/Study/professional_dataset/collar_32X32/" + path)
        img = np.asarray(img)
        X_train.append(img)

    X_train = np.asarray(X_train)
    X_train = (X_train / 255 - 0.5) * 2
    X_train = np.clip(X_train, -1, 1)

    # half_batch = batch_size // 2

    for i in range(epoch):
        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        # Train Discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        # Sample noise and generate a half batch of new images
        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(noise)

        # Train the discriminator (real classified as ones and generated as zeros)
        d_loss_real = discriminator.train_on_batch(imgs, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Print progress
        print('epoch:%d' % i, ' d_loss:%.4f' % d_loss, ' g_loss:%.4f' % g_loss)

        d_loss_memo.append(d_loss)
        g_loss_memo.append(g_loss)

        # 이부분은 중간 과정을 이미지로 저장해 주는 부분입니다.
        if i % saving_interval == 0:
            # r, c = 5, 5
            noise = np.random.normal(0, 1, (25, 100))
            gen_imgs = generator.predict(noise)

            for k in range(25):
                plt.subplot(5, 5, k + 1, xticks=[], yticks=[])
                plt.imshow(((gen_imgs[k] + 1) * 127).astype(np.uint8))
                # print((gen_imgs[k] + 1)* 127)

            ### 경로지정
            plt.savefig("/content/gdrive/MyDrive/Study/professional_dataset/collar_GAN/gan_collar_w_%d.png" % i)
            plt.tight_layout()
            plt.show()

    # plotting the metrics
    plt.plot(d_loss_memo)
    plt.plot(g_loss_memo)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Discriminator', 'Adversarial'], loc='center right')
    plt.show()


gan_train(12001, 32, 200)  # 12000번 반복되고(+1을 해 주는 것에 주의), 배치 사이즈는 32,  200번 마다 결과가 저장되게 하였습니다.