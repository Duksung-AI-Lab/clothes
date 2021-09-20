import glob
import os, sys
import random

import cv2

real_path = os.path.dirname(os.path.realpath(__file__))
sub_path = os.path.split(real_path)[0]
os.chdir(sub_path)

from flask import Flask, escape, request, Response, g, make_response
from flask.templating import render_template

from tensorflow.keras.models import load_model

class_file_list, search_file_list = None, None
os.chdir('web/static')


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

    os.remove('user_img.jpg')

    return pattern_cls_index[pattern_result_classes[0]]


app = Flask(__name__)
app.debug = True


# Main page
@app.route('/')
def clothes():
    return render_template('clothes.html')


@app.route('/', methods=['GET', 'POST'])
def post():
    if request.method == 'POST':
        global class_file_list, search_file_list

        if "classform" in request.form:
            # Reference Image
            collar = str(request.form['collars'])
            pattern = str(request.form['pattern'])

            result_dir_path = os.path.join('images/result_img', collar + '_' + pattern + '/*')
            class_file_list = glob.glob(result_dir_path)

            random.shuffle(class_file_list)
            class_file_list = class_file_list[:10]

            print(result_dir_path, class_file_list)

        else:
            # User Image
            user_img = request.files['user_img']
            user_img.save('user_img.jpg')

            pattern = model_predict()

            user_dir_path = os.path.join('images/result_img', 'Band' + '_' + pattern + '/*')
            search_file_list = glob.glob(user_dir_path)

            random.shuffle(search_file_list)
            search_file_list = search_file_list[:10]

        return render_template('clothes.html', class_res=class_file_list, search_res=search_file_list)


if __name__ == "__main__":
    app.run()
