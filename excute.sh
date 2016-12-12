#generate images and necessary index
cd core_files
matlab -nosplash -nodesktop -r get_im ../GOPR0783.mp4
mkdir ../results/gesture
mkdir build
cd build
#generate input for gesture network
g++ ../bgsegm.hpp ../src/bgfg_gmg.cpp ../generate_input.cpp -o gesture_input.out `pkg-config --cflags --libs opencv`
cd ..
./build/gesture_input.out ../results/index/gesture_index.txt
#run gesture network
python ./solve_gesture.py
#data augmentation for training detector
matlab -nosplash -nodesktop -r data_aug
cd build
g++ ../png2h5.cpp -o png2h5.out `pkg-config --cflags --libs opencv` 
cd ..
./build/png2h5.out
mkdir ../results/appearance
#train apppearance network, change iter_num below for training with different iterations
python ./solve_appearance.py [iter_num]
