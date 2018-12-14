Audio-Visual Speech Recognition using chain model in kaldi toolkit 
==================================================================
1. DB 준비
----------
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



1. Make sure you have the audio data of CHiME-1 (track 1) or CHiME-2.
http://spandh.dcs.shef.ac.uk/chime_challenge/chime2013/chime2_task1.html#data

2. Adjust the paths in path.sh according to your environment.

3. Run the script ./run.sh
