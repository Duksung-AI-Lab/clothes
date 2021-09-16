import os
import cv2

dir_path = 'dataset/collar/collars_500x600(2368)/'  # 경로 지정
save_dir = dir_path[:-1] + '_jpg/'  # 경로 자동 생성

# 디렉토리 없으면 생성
if not os.path.exists(save_dir):
    set_list = ['train', 'test', 'val']
    cls_list = ['straight', 'wide']
    for s in set_list:
        for c in cls_list:
            os.makedirs(save_dir + s + '/' + c)

for (path, dirs, files) in os.walk(dir_path):
    for file_name in files:

        if not file_name.endswith('.png'):
            continue

        img_path = path + '/' + file_name
        save_path = save_dir + path[len(dir_path):] + '/' + file_name[:-4] + '.jpg'

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print("file none")
            exit(1)

        # b_channel, g_channel, r_channel, a_channel = cv2.split(img)

        # mask_img = cv2.merge([b_channel, g_channel, r_channel])
        print(img_path, img.shape, save_path)
        cv2.imwrite(save_path, img)
