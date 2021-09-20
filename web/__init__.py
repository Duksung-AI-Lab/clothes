import glob
import os, sys
import random

import numpy as np

real_path = os.path.dirname(os.path.realpath(__file__))
sub_path = os.path.split(real_path)[0]
os.chdir(sub_path)

from flask import Flask, escape, request, Response, g, make_response, url_for
from flask.templating import render_template
from werkzeug.utils import secure_filename, redirect

app = Flask(__name__)
app.debug = True


def root_path():
    # root 경로 유지
    real_path = os.path.dirname(os.path.realpath(__file__))
    sub_path = "\\".join(real_path.split("\\")[:-1])
    return os.chdir(sub_path)


# Main page
@app.route('/')
def clothes():
    return render_template('clothes.html')


# class_file_list, search_file_list = None, None
os.chdir('web/static/')


@app.route('/', methods=['GET', 'POST'])
def post():

        # root_path()

        # global class_file_list, search_file_list
        if "classform" in request.form:
            src = 'class'
            if request.method == 'POST':
                # Reference Image
                collar = str(request.form['collars'])
                pattern = str(request.form['pattern'])

            else:
                collar = request.args.get('collars')
                pattern = request.args.get('pattern')

            # result_dir_path = os.path.join('images/result_img', str(collar) + '_' + str(pattern) + '/*')
            # class_file_list = glob.glob(result_dir_path)
            #
            # random.shuffle(class_file_list)
            # class_file_list = class_file_list[:10]
            print(collar, pattern)
            return redirect(url_for('post_class', type=collar+'_'+pattern))


        else:
            src = 'search'

            # User Image
            user_img = request.files['user_img']
            collar = 'Notched'
            pattern = 'solid'

            # search_dir_path = os.path.join('images/result_img', str(collar) + '_' + str(pattern) + '/*')
            # search_file_list = glob.glob(search_dir_path)
            #
            # random.shuffle(search_file_list)
            # search_file_list = search_file_list[:10]

            return redirect(url_for('post_search', type=collar+'_'+pattern))


@app.route('/class/<type>')
def post_class(type):
    result_dir_path = os.path.join('images/result_img', type + '/*')
    class_file_list = glob.glob(result_dir_path)

    random.shuffle(class_file_list)
    class_file_list = class_file_list[:10]

    return render_template('class.html', class_res=class_file_list)


@app.route('/search/<type>')
def post_search(type):
    search_dir_path = os.path.join('images/result_img', type + '/*')
    search_file_list = glob.glob(search_dir_path)

    random.shuffle(search_file_list)
    search_file_list = search_file_list[:10]

    return render_template('search.html', search_res=search_file_list)


if __name__ == "__main__":
    app.run()
