import sys
caffe_root = '../caffe'
sys.path.insert(0, caffe_root+ '/python')
import caffe
import surgery
import eul_test

import numpy as np
import os

import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))
#init
caffe.set_device(1)
caffe.set_mode_gpu()

argument = sys.argv
weights = '../caffe/models/appearance/VGG_VOC2012ext.caffemodel'
solve = '../caffe/models/appearance/VGG_VOC2012ext_solver.prototxt'
solver = caffe.SGDSolver(solve)
solver.net.copy_from(weights)
iteration = int(sys.argv[1])
solver.step(iteration)


