import os
import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

# Root directory of the project
ROOT_DIR = os.path.abspath("../../../")

# Import Mask RCNN
sys.path.append("../../../")  ### 경로 확인

# Path to weights file
WEIGHTS_PATH = os.path.join(ROOT_DIR, "samples/clothes/model/mask_rcnn_collar_0030.h5")  ### 학습된 카라 인식 모델 경로 넣어주기

# Directory to save logs and model checkpoints, if not provided
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")  ### 경로 확인

# GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


############################################################
#  Configurations
############################################################

class ClothesConfig(Config):
    """Configuration for training on Clothes Dataset(Top, Botton category segmentation custom dataset)
    Derives from the base Config class and overrides values specific
    to the Clothes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "clothes"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + category(collar)

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 200

    USE_MINI_MASK = True


class InferenceConfig(ClothesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


############################################################
#  Predict
############################################################

def predict():
    ### 학습된 CNN모델 경로지정
    collar_model = load_model(os.path.join(ROOT_DIR, "samples/clothes/model/densenet121_ft_1.h5"))
    pattern_model = load_model(os.path.join(ROOT_DIR, "samples/clothes/model/densenet121_ft_5c_1.h5"))

    for i in range(len(img_file_name_list)):
        try:
            image_file_name = img_file_name_list[i]
            crop_img = crop_img_list[i]
            crop_img = cv.resize(crop_img, (224, 224))
            crop_img = crop_img.astype('float32') / 255.
            crop_img = crop_img.reshape((1, 224, 224, 3))
            ### 원본 이미지 불러오기 경로지정
            original_image = cv.imread(
                os.path.join(ROOT_DIR, "samples/clothes/DataSet/패턴_통합_DataSet/") + image_file_name, cv.IMREAD_COLOR)
            image = cv.resize(original_image, (224, 224))
            image = image.astype('float32') / 255.
            image = image.reshape((1, 224, 224, 3))

            print("Running on {}".format(image_file_name))

            # collar 예측
            collar_cls_index = ['Band', 'ButtonDown', 'Notched', 'Regular']
            collar_result_classes = collar_model.predict_classes(crop_img)
            collar_result = collar_model.predict(crop_img)
            print('예측:', collar_cls_index[collar_result_classes[0]], 100 * max(collar_result[0]))

            # pattern 예측
            pattern_cls_index = ['check', 'dot', 'floral', 'solid', 'stripe']
            pattern_result_classes = pattern_model.predict_classes(image)
            pattern_result = pattern_model.predict(image)
            print('예측:', pattern_cls_index[pattern_result_classes[0]], 100 * max(pattern_result[0]))

            # Save output
            ### 결과 저장 경로 지정
            if max(collar_result[0]) * 100 >= 70 and max(pattern_result[0]) * 100 >= 70:
                test_file_name = os.path.join(ROOT_DIR, "samples/clothes/Predict/패턴통합이용/") + \
                                 collar_cls_index[collar_result_classes[0]] + "_" + \
                                 pattern_cls_index[pattern_result_classes[0]] + "/{}".format(image_file_name)
                print(test_file_name)
                cv.imwrite(test_file_name, original_image)
            else:
                test_file_name = os.path.join(ROOT_DIR, "samples/clothes/Predict/패턴통합이용/") + "Etc/{}".format(
                    image_file_name)
                cv.imwrite(test_file_name, original_image)
        except Exception as e:
            print("image didn't be saved")
            print(str(e))


############################################################
#  model Detect
############################################################
###  test
def detect(model):
    ### 경로 지정 (원하는 이미지가 있는 디렉토리)
    test_img_dir = os.path.join(ROOT_DIR, "samples/clothes/DataSet/패턴_통합_DataSet")
    images = os.listdir(test_img_dir)

    for image_file_name in images:
        print("Running on {}".format(image_file_name))

        # Read image
        image = cv.imread(test_img_dir + '/' + image_file_name, cv.IMREAD_COLOR)
        # cv.imshow(image_file_name, image)

        # Detect objects
        try:
            r = model.detect([image], verbose=1)[0]

            # detect highest score bbox & crop image to the bbox & resize = (224, 224)
            if r['rois'].shape[0] > 0:
                id = np.argmax(r['scores'])
                print(r['rois'])
                big_box = r['rois'][id]
                x1, y1, x2, y2 = big_box
                x1 = x1 - 10 if x1 - 10 > 0 else 0
                y1 = y1 - 10 if y1 - 10 > 0 else 0
                x2 = x2 + 10 if x2 + 10 < image.shape[0] else image.shape[0]
                y2 = y2 + 10 if y2 + 10 < image.shape[1] else image.shape[1]

                width = x2 - x1
                height = y2 - y1
                if width > height:
                    dif = width - height
                    crop_img = image[x1:x2,
                               int(y1 - dif / 2) if int(y1 - dif / 2) >= 0 else 0:int(y2 + dif / 2) if (y2 + dif / 2) <=
                                                                                                       image.shape[
                                                                                                           1] else
                               image.shape[1]]
                else:
                    dif = height - width
                    crop_img = image[
                               int(x1 - dif / 2) if int(x1 - dif / 2) >= 0 else 0: int(x2 + dif / 2) if int(
                                   x2 + dif / 2) <=
                                                                                                        image.shape[
                                                                                                            0] else
                               image.shape[0], y1:y2]
                print("bbox 가로 세로:", crop_img.shape)
                crop_img = cv.resize(crop_img, dsize=(224, 224), interpolation=cv.INTER_CUBIC)
                print("resize후 가로 세로:", crop_img.shape)

                # Save output
                img_file_name_list.append(image_file_name)
                crop_img_list.append(crop_img)
                print("detect success")
            else:
                print("detect fail")

        except Exception as e:
            print("detect fail")
            print(str(e))



############################################################
#  main
############################################################

if __name__ == "__main__":
    # Configurations
    config = InferenceConfig()
    config.display()

    # Create model
    model = MaskRCNN(mode="inference", config=config,
                     model_dir=DEFAULT_LOGS_DIR)
    # Load weights
    # model.load_weights(WEIGHTS_PATH, by_name=True)
    model.load_weights(WEIGHTS_PATH, by_name=True)

    # detect collar and crop in bbox
    # detect success image add to img_file_name_list, crop_img_list
    img_file_name_list = []
    crop_img_list = []
    detect(model)

    # classification with CNN
    predict()