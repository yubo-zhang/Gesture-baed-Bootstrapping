from __future__ import division
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import os
from datetime import datetime
import sys
import scipy


caffe_root = '/home/yuboz/caffe1/caffe'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + '/python')
import caffe


def F1_measure(test_net, gt_dir,var_dir, dataset, layer_mean, layer_var, data,sample_num):
    result = do_measure(test_net, gt_dir,var_dir, dataset, layer_mean,layer_var, data,sample_num)
    return result

def do_measure(net, gt_dir,var_dir, dataset, layer_mean,layer_var, data,sample_num):
    index = 1

    for idx in range(dataset):
        output_matrix_m = np.empty((sample_num,380,1030))
        output_matrix_v = np.empty((sample_num,48,129))
        for times in range(sample_num):
            net.forward()
            output_mean = net.blobs[layer_mean].data[0].astype(np.float64)
            output_var =  net.blobs[layer_var].data[0].astype(np.float64)
            output_v = output_var[1] # prob of being a foreground
            output_m = output_mean[1]
            
            #vmax = np.amax(output_realsize);
            #vmin = np.amin(output_realsize);
            #show1 = np.array((output_realsize-vmin)/(vmax-vmin)*250,dtype = np.uint8)
            #im = Image.fromarray(show1)
            #im.save('out.jpg')
            #break;
            #np.savetxt(os.path.join(gt_dir,'check.txt'),output_v, delimiter = ',',newline = '\n')
            output_matrix_m[times] = output_m
            output_matrix_v[times] = output_v
        total_im_m = np.sum(output_matrix_m, axis = 0)
        total_im_v = np.sum(output_matrix_v, axis = 0)
        mean = total_im_m/sample_num
        mean[mean<0.5] = 0
        mean[mean>=0.5] = 1
        v_mean = total_im_v/sample_num

        big_mean = np.tile(v_mean,[sample_num,1,1])
        dif = big_mean-output_matrix_v
        variance1 = np.sum(np.square(dif),axis = 0)/(sample_num-1)
        variance = scipy.misc.imresize(variance1, (380,1030),interp='bilinear')
        var_max = np.amax(variance)
        var_min = np.amin(variance)
        variance_norm1 = (variance-var_min)/(var_max-var_min)+0.8
        one = np.ones((380,1030),dtype = np.float64)
        variance_norm = np.divide(one, variance_norm1)
        if var_dir:
            np.savetxt(os.path.join(var_dir, str(idx+1) + '.txt'),variance_norm,delimiter = ',',newline = '\n')
        if gt_dir:
            mask = np.ndarray.astype(mean, np.uint8)
            im = Image.fromarray(mask)
            im.save(os.path.join(gt_dir, str(idx+1) + '.png'))
