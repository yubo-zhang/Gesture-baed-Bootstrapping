DATA=/home/yuboz/data/input_data/videos/GOPR0502/input_result/2_mask
rm -rf $DATA/img_train_lmdb
/home/yuboz/caffe4/caffe/build/tools/convert_imageset --gray \
/home/yuboz/data/input_data/videos/GOPR0502/input_result/2_mask/ $DATA/index.txt  $DATA/img_train_lmdb
