import warnings
import cv2, os
import matplotlib.pyplot as plt
from keras.models import load_model

warnings.simplefilter(action='ignore', category=FutureWarning)

model = load_model('collars_model.h5')
# model = load_model('collars_crop_model.h5')
path = 'dataset/collars/test/'
# path = 'dataset/collars_crop/test/'


nrows = 4
ncols = 6
k = 0
for n in range(4):
    i = 0
    plt.figure(figsize=(12, 8))

    img_num = 0
    while img_num < 24:
        img = cv2.imread(path + 'splash_SHIRT_' + str(2848 + k) + '.jpg.png', cv2.IMREAD_UNCHANGED)
        k += 1

        if img is None:
            # bk = True
            continue

        i += 1
        img_num += 1
        recolor_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        plt.subplot(nrows, ncols, i)
        test_num = cv2.resize(recolor_img, (125, 150))
        test_num = test_num.astype('float32') / 255.
        # test_num = cv2.cvtColor(test_num, cv2.COLOR_RGBA2RGB)
        plt.imshow(cv2.cvtColor(test_num, cv2.COLOR_RGBA2RGB), interpolation='nearest')

        test_num = test_num.reshape((1, 125, 150, 4))

        cls_index = ['etc', 'straight', 'wide']
        result_classes = model.predict_classes(test_num)
        result = model.predict(test_num)

        plt.title(cls_index[result_classes[0]])

        print('예측:', cls_index[result_classes[0]])
        print(max(result[0]))

    plt.tight_layout()
    plt.show()
