import glob
import os, sys
import random

real_path = os.path.dirname(os.path.realpath(__file__))
sub_path = os.path.split(real_path)[0]
os.chdir(sub_path)

from flask import Flask, escape, request, Response, g, make_response
from flask.templating import render_template
from werkzeug.utils import secure_filename

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


class_file_list, search_file_list = None, None
os.chdir('web/static')

@app.route('/', methods=['GET', 'POST'])
def post():
    if request.method == 'POST':
        # root_path()
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
            user_dir_path = './images/result_img/Band_dot/*'
            search_file_list = glob.glob(user_dir_path)

            random.shuffle(search_file_list)
            search_file_list = search_file_list[:10]

        return render_template('clothes.html', class_res=class_file_list, search_res=search_file_list)


if __name__ == "__main__":
    app.run()
