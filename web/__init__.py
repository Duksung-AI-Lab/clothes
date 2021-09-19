import os, sys

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


result_img_path, user_img_path = None, None


@app.route('/', methods=['GET', 'POST'])
def post():
    if request.method == 'POST':
        # root_path()
        global result_img_path, user_img_path
        if "classform" in request.form:
            # Reference Image
            collar = request.form['collars']
            pattern = request.form['pattern']
            result_dir_path = './images/result_img/' + str(collar) + '_' + str(pattern)
            result_img_path = result_dir_path + '/' + 'result3.jpg'

        else:
            # User Image
            user_img = request.files['user_img']
            user_img_path = './images/result_img/Band_dot/406.jpg'

        return render_template('clothes.html', class_res=result_img_path, search_res=user_img_path)


if __name__ == "__main__":
    app.run()
