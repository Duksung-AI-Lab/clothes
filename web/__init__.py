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


# @app.route('.')
# def get():
#     return render_template('clothes.html')


@app.route('/', methods=['GET', 'POST'])
def post():
    if request.method == 'POST':
        # root_path()

        # Reference Image
        collar = request.form['collars_img']
        pattern = request.form['pattern_img']
        result_img_path = '/Users/kr/Downloads/ICT 결과물을 위한 전달자료/카라&패턴 예측/모델이 분류한 ' \
                         'DataSet/카라_통합_DataSet_이용/' + 'Band_check/141.jpg'
                         # + str(collar) + '_' + str(pattern)


        # # User Image (target image)
        # user_img = request.files['user_img']
        # user_img.save('./flask_deep/static/images/'+str(user_img.filename))
        # user_img_path = '/images/'+str(user_img.filename)

        # # Neural Style Transfer
        # transfer_img = neural_style_transfer.main(refer_img_path, user_img_path)
        # transfer_img_path = '/images/' + str(transfer_img.split('/')[-1])

    return render_template('clothes.html',
                           image_file=result_img_path)



### Object Detection ###
# @app.route('/object_detection_get')
# def object_detection_get():
#     return render_template('object_detection_get.html')
#
#
# @app.route('/object_detection_post', methods=['GET', 'POST'])
# def object_detection_post():
#     if request.method == 'POST':
#
#         root_path()
#         # User Image
#         user_img = request.files['object_img']
#         user_img.save('./flask_deep/static/images/' + str(user_img.filename))
#         user_img_path = '../static/images/' + str(user_img.filename)
#
#         user_img_type = str(user_img.filename).split('.')[1]
#         # img file
#         if user_img_type in ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']:
#             print('Type is Image')
#             # Object Detection by darkflow with YOLO V2
#             detected_img = processing_images.main(user_img_path)
#             detected_img_path = '../static/images/' + str(detected_img.split('/')[-1])
#
#         # video file
#         elif user_img_type in ['avi', 'AVI', 'mp4', 'MP4', 'MPEG', 'mkv', 'MKV']:
#             print('Type is Video')
#
#     return render_template('object_detection_post.html', detected_img=detected_img_path)

if __name__ == "__main__":
    app.run()
