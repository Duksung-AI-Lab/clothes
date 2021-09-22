import cv2
from tensorflow.keras.models import load_model



def model_predict():
    model = load_model('models/pattern_5class.h5')

    original_image = cv2.imread('user_img.jpg', cv2.IMREAD_COLOR)
    image = cv2.resize(original_image, (224, 224))
    image = image.astype('float32') / 255.
    image = image.reshape((1, 224, 224, 3))

    pattern_cls_index = ['check', 'dot', 'floral', 'solid', 'stripe']
    pattern_result_classes = model.predict_classes(image)
    # pattern_result = model.predict(image)
    # print('예측:', pattern_cls_index[pattern_result_classes[0]], 100 * max(pattern_result[0]))

    return pattern_cls_index[pattern_result_classes[0]]
