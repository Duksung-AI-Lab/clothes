import warnings
import cv2, os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.simplefilter(action='ignore', category=FutureWarning)

# 모델 불러오기
model = tf.keras.models.load_model('../model/패턴분류모델.h5')

# # 테스트할 데이터셋 불러오기
# rootPath = '../DataSet/pattern_5class'
#
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# test_generator = test_datagen.flow_from_directory(
#        os.path.join(rootPath, 'test'),
#        batch_size=10,
#        target_size=(224, 224),
#        class_mode='categorical',
#        shuffle=True)

# # 테스트 결과
# print("Test Accuracy: %.4f" % (model.evaluate_generator(test_generator)[1]))

# 클래스별로 모델 예측하기
path = '../DataSet/pattern_5class/test/'  # 예측할 데이터셋 폴더 경로 지정

nrows = 4
ncols = 6


def predict(classes):
    global total_cor, correct, all
    total_cor = 0  # 정답을 맞췄을 경우 예측 확률값의 합
    correct = 0  # 예측 결과 정답을 맞춘 개수
    all = 0  # 예측한 전체 이미지 개수

    file_list = os.listdir(path + classes + '/')
    for n in range(4):
        bk = False
        plt.figure(figsize=(12, 8))
        for k in range(1, 25):
            try:
                img = cv2.imread(path + classes + '/' + file_list[24 * n + k - 1])

                if img is None:
                    print(file_list)
                    print('file none')
                    bk = True
                    break

                recolor_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.subplot(nrows, ncols, k)
                test_num = cv2.resize(recolor_img, (224, 224))
                test_num = test_num.astype('float32') / 255.

                plt.imshow(test_num, interpolation='nearest')

                test_num = test_num.reshape((1, 224, 224, 3))

                cls_index =  ['check', 'dot', 'floral', 'solid', 'stripe']
                result_classes = model.predict_classes(test_num)
                result = model.predict(test_num)
                print(result_classes)

                if cls_index[result_classes[0]] == classes:
                    fd = {'color': 'black'}
                    answer = 'O'
                    total_cor += max(result[0])
                    correct += 1
                else:
                    fd = {'color': 'red'}
                    answer = 'X'

                plt.title("{} {:2.0f}%".format(cls_index[result_classes[0]], 100*max(result[0])), fontdict=fd)

                print('예측:', cls_index[result_classes[0]], answer)
                print(max(result[0]))

                all += 1
            except Exception as e :
                print(str(e))
        plt.tight_layout()
        plt.show()
        if bk:
            return

predict('stripe')  # 예측하고자 하는 클래스 이름 지정

cor_percent = correct / all * 100  # 정답을 맞출 확률
average = total_cor / correct  # 정답을 맞춘 예측 확률들의 평균 값
print("correct {}%".format(cor_percent))
print("average prediction : {}%".format(average))

