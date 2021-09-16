import os
import cv2
import numpy as np

dir_path = 'dataset/collars_500x600/'  # 경로 지정
save_dir = dir_path[:-1] + '_crop/'  # 경로 자동 생성

# 디렉토리 없으면 생성
if not os.path.exists(save_dir):
    set_list = ['train', 'test', 'val']
    cls_list = ['straight', 'wide', 'etc']
    for s in set_list:
        for c in cls_list:
            os.makedirs(save_dir + s + '/' + c)

for (path, dirs, files) in os.walk(dir_path):
    for file_name in files:

        if not file_name.endswith('.png'):
            continue

        img_path = path + '/' + file_name
        save_path = save_dir + path[len(dir_path):] + '/' + file_name

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print("file none")
            exit(1)

        b_channel, g_channel, r_channel, a_channel = cv2.split(img)

        # 최대,최소 좌표 검색
        pointList = np.argwhere(np.array(a_channel) > 0)
        y_max = np.max(np.array(pointList).T[0])
        y_min = np.min(np.array(pointList).T[0])
        x_max = np.max(np.array(pointList).T[1])
        x_min = np.min(np.array(pointList).T[1])
        # print(pointList)

        # # 채널 값 출력
        # for i in a_channel:
        #     for j in i:
        #         print("%3d" % j, end=" ")
        #     print()

        crop = img[y_min: y_max + 1, x_min: x_max + 1]
        print(img_path, crop.shape, save_path)
        cv2.imwrite(save_path, crop)
