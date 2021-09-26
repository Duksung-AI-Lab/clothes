# clothes
find &amp; classification

## image_process directory
사용했던 데이터 전처리 코드

## predict directory
생성한 모델들로 이미지 예측코드

## train_codes directory
학습을 위해 사용한 코드
- simple cnn : 일반 cnn으로 학습
- transfer : 전이학습 이용
- gan : GAN 모델로 이미지 생성 학습

## web directory
웹 개발을 위해 사용한 코드 및 데이터

***
***

## collar_cnn.py
셔츠 카라유형을 straight point, wide collar, etc로 분류하는 CNN 코드

## crop.py
크롤링한 셔츠 이미지를 셔츠 카라부분만 잘라서 이미지로 저장하는 코드

## dataset
collar_v2_crop : 더 명확한 기준을 가진 셔츠 카라 이미지들로 라벨링한 것(카라부분만 잘려진 이미지)\
mask_collars_crop_200x200 : 이미지 크기들이 조금씩 달라서 200x200으로 맞춘 마스크 이미지들(카라부분만 잘려진 마스크 이미지)

## vgg16_test.py
구조가 간단한 모델\
필터 사이즈가 3x3으로 고정, 16개의 층으로 구성

## resnet_test.py
skip connection, bottleneck을 이용한 모델\
VGGNet보다 깊은 층으로 설계\
특성맵들끼리 더하는 방식

## CNN-SENet.ipynb
어떤 모델에도 적용할 수 있는 SE Block을 활용한 모델\
특성맵들을 1x1 사이즈로 squeeze, 상대적 중요도를 알아내는 excitation을 동작
convolution을 통해 생성된 특성을 채널당 중요도를 고려해서 재보정하는 것

## Black_White_Image.ipynb
데이터로 쓸 이미지들의 크기들이 조금씩 달라서 정해진 크기로 만들어주는 코드\
200x200로 셔츠 카라부분만 crop해서 생성

## DenseNet(titu).ipynb
특성맵끼리 concatenation을 시킨 모델\
bottleneck 이용\
ResNet보다 적은 파라미터 

## u-net_segmentation.ipynb
적은 데이터를 가지고도 정확한 Segmentation\
U자형태의 모델, Contracting path+Expansive path\
End-to-End 구조
