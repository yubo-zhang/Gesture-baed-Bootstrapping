from __future__ import division
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import os
from datetime import datetime
import sys
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as inv_tool
import tables as tb
from sklearn.covariance import GraphLassoCV, ledoit_wolf
from sklearn.externals.joblib import Parallel, delayed
from scipy.linalg import svd,diagsvd,eigvals,inv,norm
import math
caffe_root = '/home/yuboz/caffe_ts/caffe'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + '/python')
import caffe

def F1_measure(test_net,save_dir,txt_dir,dataset,layer,image ,gt):
	result = do_measure(test_net,save_dir,txt_dir,dataset,layer,image ,gt)

def do_measure(net,save_dir,txt_dir,dataset,layer,image ,gt):
	if save_dir:
		if not (os.path.exists(save_dir)):
			os.mkdir(save_dir)
	if txt_dir:
		txt_file = open(txt_dir,'a')
	total = 0
	for idx in range(dataset):
		net.forward()
		output_acc1 = net.blobs[layer].data[0].astype(np.float64)
		output_acc = output_acc1[0,:,:]
		output_acc_show = np.zeros((380,1030),dtype = np.uint8)
		output_acc_show[output_acc>=0.5] = 1
		#print output_acc
		output_data= net.blobs[image].data[0].astype(np.float64)
		gt_output= net.blobs[gt].data[0].astype(np.float64)
		data_output = np.ndarray.astype(output_data,np.uint8)

		show = np.zeros((380,1030,3),np.uint8)
		for i in range(380):
			for j in range(1030):
				if data_output[2][i][j]+output_acc_show[i][j]*100>255:
					show[i,j,0] = 255
				else:
					show[i,j,0] = data_output[2][i][j]+output_acc_show[i][j]*100
		show[:,:,1] = data_output[1]
		show[:,:,2] = data_output[0]
		t_l  = 0
		t = 0
		l = 0
		l = gt_output.sum()
		t = (output_acc>0.5).sum()
		t_l = ((gt_output > 0) & (output_acc > 0.5)).sum()
		if l == 0:
			print "there is no label ==1 in image"
			continue
		if t == 0:
			print "there is no detection ==1 in image"
			continue
		r = float(t_l)/float(l)
		p = float(t_l)/float(t)
		if r+p == 0:
			print "recall+precision ==0"
			continue
		F1 = 2*r*p/(r+p)
		total+=F1
		sh = Image.fromarray(show)
		sh.save(os.path.join(save_dir,str(idx) + '.png'))
	txt_file.write(str(total/(idx+1)))
	txt_file.write("\n")
	txt_file.close()

