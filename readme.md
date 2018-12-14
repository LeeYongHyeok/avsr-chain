Audio-Visual Speech Recognition using chain model in kaldi toolkit 
==================================================================
# 1. DB 

## 1.1 Audio
CHiME-1 or CHiME-2(small corpus)의 실생활 잡음 데이터를 사용한다.
http://spandh.dcs.shef.ac.uk/chime_challenge/chime2013/chime2_task1.html#data

## 1.2 Video
Audio DB는 원래 Audio Visual Database인 GRID DB에서 clean audio를 챌린지에 맞게 수정한 데이터이므로,
Video DB는 GRID DB를 사용한다.
이때, 21번화자와 s8의 일부는 DB에 문제가 있으므로 사용하지 않는다.
http://spandh.dcs.shef.ac.uk/gridcorpus/

## 1.3 Audio-Video
Audio와 Video는 기본적으로 싱크가 맞지 않기 때문에 싱크를 맞춰야 한다.
이를 맞추기 위하여 MATLAB에서 Audio와 Video의 Audio의 cross correlation이 maximum 되는 구간을 계산하여 delay time 정보를 구하고 (cut.m)
이를 csv 형식으로 저장하여 ubuntu환경에서 ffmpeg를 이용하여 영상을 trim 한다. (cutVideo.py)

# 2. Video Feature Preparation

## 2.1 Lip Extraction
LipNet의 script/extract_mouth_batch.py 를 사용하며, 이때 predict는 shape_predictor_68_face_landmarks.dat를 사용한다.
![image](https://user-images.githubusercontent.com/39906282/49982241-f24f5080-ff9e-11e8-8211-6f9c6d0964e3.png)

## 2.2 CNN-VAE feature
### 2.2.1 Training
CNN기반 Variance AutoEncoder를 적절한 training을 하기 위하여 PyTorch를 기반으로 스크립트를 작성하였으며, Encoder로 ReLU함수를 activation function으로 하는 Convolution 4 Layer를 사용하였고 그 후 Fully Connected 3 Layer, Decoder또한 Encoder와 비슷하게 4 Layer로 구성하였다.
이후 Epoch 50회 Training하였다. (CNN-VAE.py)
![image](https://user-images.githubusercontent.com/39906282/49982487-1e1f0600-ffa0-11e8-947b-6bc685e6cf49.png)

### 2.2.2 Feature Extraction
2.1에서 추출한 Lip Image를 2.2.1에서 Training 시킨 모델의 입력으로 하고 출력을 bottleneck에서 구하면 feature로 사용할 수 있다. (feat_extract_CNN_VAE.py)

## 2.2 Modifying features to fit the Kaldi type
frame단위로 추출한 features를 utterance 단위로 합치고, 이를 kaldi에서 인식할 수 있도로 구분자와 utterance_ID를 삽입하여 형태를 맞춰준다. (writeFeatures.py)

## 2.3 Replace existing DCT feature structures
화자/발화.ark 파일로 구성되어 있는 features를 train/ test/ devel/ 의 형태로 분배한다. (sortFeatures.py)

## 3 Video Feature Augmentation
chain model (kaldi/nnet3)는 decoding 속도를 빠르게 하기 위해 tri-state HMM을 one-state HMM 구조로 줄였고 이 때문에 발생하는 성능 하락을 막기 위해 0.9, 1.1배속과 볼륨 조절을 하여 data augmentation을 한다. Video data 또한 Audio data의 frame을 맞춰야 하기 때문에, data augmentation을 해야한다.
이를 맞추기 위하여 Differential Digital Analyser(DDA)를 구현하여 서로 다른 feature frame을 맞춰준다. (dda_video_feat.sh)
![image](https://user-images.githubusercontent.com/39906282/49982931-ecf40500-ffa2-11e8-8f2c-e8afa9b32d14.png)






