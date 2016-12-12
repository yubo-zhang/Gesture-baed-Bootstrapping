import sys
caffe_root = '../caffe'
sys.path.insert(0, caffe_root+ '/python')
import caffe
import surgery
import test_var_mean

import numpy as np
import os

import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))
#init
caffe.set_device(3)
caffe.set_mode_gpu()


layer_var = 'score1'
image = 'data'
layer_mean = 'prob1'

cur_path = os.getcwd()
a = cur_path.split('core_files')
gt_dir = os.path.join(a[0] + 'results/gesture/Masks')
if not os.path.exists(gt_dir):
	os.makedirs(gt_dir)
var_dir = os.path.join(a[0] + 'results/gesture/Variance')
if not os.path.exists(var_dir):
	os.makedirs(var_dir)

test_num = len([name for name in os.listdir('../results/gesture/result_bg_op') if os.path.isfile(os.path.join('../results/gesture/result_bg_op/',name))]) #number of images to test
sample_num = 100
model = '../caffe/models/gesture/VGG_VOC2012_fine_tune.caffemodel'
test_net = caffe.Net('../caffe/models/gesture/test.prototxt',model,caffe.TEST)
test_var_mean.F1_measure(test_net,gt_dir,var_dir,test_num,layer_mean,layer_var,image,sample_num)
