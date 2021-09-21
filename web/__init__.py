import glob
import os, sys
import random

import cv2

real_path = os.path.dirname(os.path.realpath(__file__))
sub_path = os.path.split(real_path)[0]
os.chdir(sub_path)

from flask import Flask, escape, request, Response, g, make_response
from flask.templating import render_template

import patterns
# import Collar_and_Pattern_Predict

class_file_list, search_file_list, refer_img = None, None, None
os.chdir('web/static')

app = Flask(__name__)
app.debug = True


# Main page
@app.route('/')
def clothes():
    return render_template('clothes.html')


@app.route('/', methods=['GET', 'POST'])
def post():
    if request.method == 'POST':
        global class_file_list, search_file_list, refer_img

        if "classform" in request.form:
            # Reference Image
            collar = str(request.form['collars'])
            pattern = str(request.form['pattern'])

            result_dir_path = os.path.join('images/result_img', collar + '_' + pattern + '/*')
            class_file_list = glob.glob(result_dir_path)

            random.shuffle(class_file_list)
            class_file_list = class_file_list[:10]
            class_file_list.append(collar)
            class_file_list.append(pattern)

        else:
            # User Image
            user_img = request.files['user_img']
            user_img.save('user_img.jpg')
            refer_img = 'user_img.jpg'

            collar = 'Regular'
            pattern = patterns.model_predict()

            # collar = Collar_and_Pattern_Predict.collar_predict()
            # pattern = Collar_and_Pattern_Predict.pattern_predict()

            user_dir_path = os.path.join('images/result_img', collar + '_' + pattern + '/*')
            search_file_list = glob.glob(user_dir_path)

            random.shuffle(search_file_list)
            search_file_list = search_file_list[:10]
            search_file_list.append(pattern)

        return render_template('clothes.html', class_res=class_file_list, search_res=search_file_list,
                               refer_img=refer_img)


if __name__ == "__main__":
    app.run()
