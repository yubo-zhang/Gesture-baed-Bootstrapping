DATA=/home/yuboz/data/input_data/videos/GOPR0502/input_result/rgb
rm -rf $DATA/img_train_lmdb
/home/yuboz/caffe4/caffe/build/tools/convert_imageset \
/home/yuboz/data/input_data/videos/GOPR0502/input_result/rgb/ $DATA/index.txt  $DATA/img_train_lmdb
