import numpy as np
import tables as tb
from scipy.sparse import csr_matrix
import h5py
import sys
caffe_root = '/home/yuboz/Bootstrapping/GBN/caffe'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + '/python')
import caffe

class EuclideanLossLayer(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need three inputs to compute distance.")

    def reshape(self, bottom, top):
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        if bottom[0].count != bottom[2].count:
            raise Exception("Inputs and weights must have the same dimension.")
        # difference is shape of inputs
        self.sig = np.zeros_like(bottom[0].data, dtype=np.float64)
        self.diff = np.zeros_like(bottom[1].data, dtype=np.float64)
        self.mask = np.zeros_like(bottom[1].data, dtype=np.float64)
        self.tran = np.zeros_like(bottom[0].data, dtype=np.int8)
        self.seed = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.mask[...] = bottom[1].data
        self.mask[self.mask==1] = 3.5
        self.mask[self.mask==0] = 0.6

        top[0].reshape(1)

    def forward(self, bottom, top):
        self.sig[...] = 1/(1+np.exp(-1*0.6*bottom[0].data))
        self.diff[...] = self.sig[...] - bottom[1].data
	a = bottom[2].data


        
        top[0].data[...] = np.sum(self.diff**2 * self.mask * bottom[2].data) / bottom[0].num / 2.

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign *0.6*self.sig[...]*(1-self.sig[...])*self.diff *bottom[2].data* self.mask / bottom[i].num

