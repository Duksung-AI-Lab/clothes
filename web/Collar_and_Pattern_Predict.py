import os
import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.models import load_model
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

# Root directory of the project
ROOT_DIR = os.path.abspath("/content/gdrive/MyDrive/Study/MaskRCNN")  ### 경로 확인

# Import Mask RCNN
sys.path.append("/content/gdrive/MyDrive/Study/MaskRCNN")  ### 경로 확인

# Path to weights file
WEIGHTS_PATH = "D:/anaconda/envs/venv_flask/clothes/web/static/models/MaskRCNN_collar_detect.h5"  ### 경로 확인

# Directory to save logs and model checkpoints, if not provided
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")  ### 경로 확인


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
graph = tf.get_default_graph()

set_session(session)
############################################################
#  Configurations
############################################################
<<<<<<< HEAD
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
=======

# class ClothesConfig(Config):
#     """Configuration for training on Clothes Dataset(Top, Botton category segmentation custom dataset)
#         Derives from the base Config class and overrides values specific
#         to the Clothes dataset.
#         """
#     # Give the configuration a recognizable name
#     NAME = "clothes"
#
#     # We use a GPU with 12GB memory, which can fit two images.
#     # Adjust down if you use a smaller GPU.
#     IMAGES_PER_GPU = 2
#
#     # Uncomment to train on 8 GPUs (default is 1)
#     GPU_COUNT = 1
#
#     # Number of classes (including background)
#     NUM_CLASSES = 1 + 1  # Background + category(collar)
#
#     # Number of training steps per epoch
#     STEPS_PER_EPOCH = 200
#
#     USE_MINI_MASK = True
#
#
# class InferenceConfig(ClothesConfig):
#     # Set batch size to 1 since we'll be running inference on
#     # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1
>>>>>>> origin/main

############################################################
#  Load Model
############################################################
# Load collar_model
### 경로 확인
collar_model = load_model("D:/anaconda/envs/venv_flask/clothes/web/static/models/collars_4class.h5")
# Load pattern_model
### 경로 확인
pattern_model = load_model("D:/anaconda/envs/venv_flask/clothes/web/static/models/pattern_5class.h5")
# Load Mark-RCNN collar model
# Configurations
config = InferenceConfig()
# Create MaskRCNN model
model = MaskRCNN(mode="inference", config=config, model_dir=DEFAULT_LOGS_DIR)
# Load weights
model.load_weights(WEIGHTS_PATH, by_name=True)
collar_model._make_predict_function()
pattern_model._make_predict_function()
model.keras_model._make_predict_function()



############################################################
#  Predict
############################################################


def collar_predict():
    """detection collars with Mask R-CNN
        not service on browser"""
<<<<<<< HEAD
    global model, collar_model, session, graph
    # Read image
    image = cv.imread('user_img.jpg', cv.IMREAD_COLOR)
    #     # Detect objects

    with graph.as_default():
        set_session(session)
        r = model.detect([image], verbose=1)[0]
    #     # detect highest score bbox & crop image to the bbox & resize = (224, 224)
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
                                                                                               image.shape[1] else
                       image.shape[1]]
        else:
            dif = height - width
            crop_img = image[
                       int(x1 - dif / 2) if int(x1 - dif / 2) >= 0 else 0: int(x2 + dif / 2) if int(x2 + dif / 2) <=
                                                                                                image.shape[0] else
                       image.shape[0], y1:y2]
            # print("bbox 가로 세로:", crop_img.shape)
            crop_img = cv.resize(crop_img, dsize=(224, 224), interpolation=cv.INTER_CUBIC)
            # print("resize후 가로 세로:", crop_img.shape)
            # print("detect success")
    else:
        # print("detect fail")
        return "fail"

    crop_img = cv.resize(crop_img, (224, 224))
    crop_img = crop_img.astype('float32') / 255.
    crop_img = crop_img.reshape((1, 224, 224, 3))

    # Predict collar
    collar_cls_index = ['Band', 'ButtonDown', 'Notched', 'Regular']

    with graph.as_default():
        set_session(session)
        collar_result_classes = collar_model.predict_classes(crop_img)
    # collar_result = collar_model.predict(crop_img)
    # print('예측:', collar_cls_index[collar_result_classes[0]], 100 * max(collar_result[0]))

    return collar_cls_index[collar_result_classes[0]]
'''
=======
    # # Configurations
    # config = InferenceConfig()
    # # config.display()

    # # Create MaskRCNN model
    # model = MaskRCNN(mode="inference", config=config, model_dir=DEFAULT_LOGS_DIR)
    # # Load weights
    # model.load_weights(WEIGHTS_PATH, by_name=True)
    #
    # try:
    #     # Read image
    #     image = cv.imread('user_img.jpg', cv.IMREAD_COLOR)
    #
    #     # Detect objects
    #     r = model.detect([image], verbose=1)[0]
    #
    #     # detect highest score bbox & crop image to the bbox & resize = (224, 224)
    #     if r['rois'].shape[0] > 0:
    #         id = np.argmax(r['scores'])
    #         print(r['rois'])
    #         big_box = r['rois'][id]
    #         x1, y1, x2, y2 = big_box
    #         x1 = x1 - 10 if x1 - 10 > 0 else 0
    #         y1 = y1 - 10 if y1 - 10 > 0 else 0
    #         x2 = x2 + 10 if x2 + 10 < image.shape[0] else image.shape[0]
    #         y2 = y2 + 10 if y2 + 10 < image.shape[1] else image.shape[1]
    #
    #         width = x2 - x1
    #         height = y2 - y1
    #         if width > height:
    #             dif = width - height
    #             crop_img = image[x1:x2,
    #                        int(y1 - dif / 2) if int(y1 - dif / 2) >= 0 else 0:int(y2 + dif / 2) if (y2 + dif / 2) <=
    #                                                                                                image.shape[1] else
    #                        image.shape[1]]
    #         else:
    #             dif = height - width
    #             crop_img = image[
    #                        int(x1 - dif / 2) if int(x1 - dif / 2) >= 0 else 0: int(x2 + dif / 2) if int(x2 + dif / 2) <=
    #                                                                                                 image.shape[0] else
    #                        image.shape[0], y1:y2]
    #             # print("bbox 가로 세로:", crop_img.shape)
    #             crop_img = cv.resize(crop_img, dsize=(224, 224), interpolation=cv.INTER_CUBIC)
    #             # print("resize후 가로 세로:", crop_img.shape)
    #             # print("detect success")
    #     else:
    #         # print("detect fail")
    #         return "fail"
    #
    #     # Load collar_model
    #     ### 경로 확인
    #     collar_model = load_model("models/collars_4class.h5")
    #
    #     crop_img = cv.resize(crop_img, (224, 224))
    #     crop_img = crop_img.astype('float32') / 255.
    #     crop_img = crop_img.reshape((1, 224, 224, 3))
    #
    #     # Predict collar
    #     collar_cls_index = ['Band', 'ButtonDown', 'Notched', 'Regular']
    #     collar_result_classes = collar_model.predict_classes(crop_img)
    #     # collar_result = collar_model.predict(crop_img)
    #     # print('예측:', collar_cls_index[collar_result_classes[0]], 100 * max(collar_result[0]))
    #
    # except Exception as e:
    #     # print("detect fail")
    #     print(str(e))
    #     return "fail"

>>>>>>> origin/main
    """ Detection and prediction without Mask R-CNN """
    # Load collar_model
    ### 경로 확인
    collar_model = load_model("models/collars_4class.h5")

    try:
        # Read image
        image = cv.imread('user_img.jpg', cv.IMREAD_COLOR)

        # Crop image
        h, w = image.shape[0] // 3, image.shape[1] // 2
        fh, fw = h * 5 // 12, h // 2
        crop_img = image[fh: fh + h, w - fw: w + fw]

        crop_img = cv.resize(crop_img, (224, 224))
        crop_img = crop_img.astype('float32') / 255.
        crop_img = crop_img.reshape((1, 224, 224, 3))

        # Predict collar
        collar_cls_index = ['Band', 'ButtonDown', 'Notched', 'Regular']
        collar_result_classes = collar_model.predict_classes(crop_img)

    except Exception as e:
        print(str(e))
        return "fail"

    return collar_cls_index[collar_result_classes[0]]

'''
def pattern_predict():

    global pattern_model, session, graph
    # Read image
    original_image = cv.imread('user_img.jpg', cv.IMREAD_COLOR)
    image = cv.resize(original_image, (224, 224))
    image = image.astype('float32') / 255.
    image = image.reshape((1, 224, 224, 3))

    # Predict pattern
    pattern_cls_index = ['check', 'dot', 'floral', 'solid', 'stripe']

<<<<<<< HEAD
    with graph.as_default():
        set_session(session)
        pattern_result_classes = pattern_model.predict_classes(image)
    # pattern_result = model.predict(image)
    # print('예측:', pattern_cls_index[pattern_result_classes[0]], 100 * max(pattern_result[0]))
=======
    except Exception as e:
        print(str(e))
        return "fail"
>>>>>>> origin/main

    return pattern_cls_index[pattern_result_classes[0]]
