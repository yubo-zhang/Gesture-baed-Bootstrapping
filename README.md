# Gesture-baed-Bootstrapping
The GBN folder contains the codes and caffe model for taining person specific hand detector in paper "Gesture-based bootstrapping for egosentric hand segmentation", the paper is available on arxiv: http://arxiv.org/abs/1612.02889.

Yubo Zhang, Vishnu Naresh Boddeti, Kris M. Kitani, “Gesture-based Bootstrapping for Egocentric Hand Segmentation” Submitted to Computer Vision and Pattern Recognition CVPR2017

The process takes an input video [in the form of MP4, AVI, MOV, WMV] and outputs a hand detecor for the user in the video. 

Before run the code, please download the folder and install caffe contained in the folder. The Libraries we use include opencv3.0, Numpy, Scipy, PIL, please make sure they are installed properly on your computer.

To begin the process, please run excute.sh, the steps are explained in the file. 
===================
Files necessary to run the process are contained in the folder core_files
The intermedia and final results are stored in folder result
The hand detector trained with different iterations can be found in path ./caffe/models/appearance
