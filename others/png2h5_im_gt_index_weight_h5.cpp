// the file is used to augment data for hand detection network
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "stdlib.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/video.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/objdetect.hpp"
#include <sys/stat.h>
#include "H5Cpp.h"
#include <cstddef> 


#ifndef H5_NO_NAMESPACE
using namespace H5;
#endif

using namespace std;
using namespace cv;

const H5std_string  DATASET_NAME( "data" );
const H5std_string  LABEL_NAME( "label" );
const H5std_string  WEIGHT_NAME( "mask" );

//convert int to string part

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}
//end

//step 1 brightness
float im_save[3][380][1030];
float label_save[380][1030];
float weight_save[380][1030];

int main(int argc, char **argv){
    char* video_name = argv[1];
    string video_name_str(video_name);
    string image_fd = "/home/yuboz/data/input_data/videos/cali_test/"+video_name_str+"/input_result/augmentation1/RGBimages";
    string fd = image_fd+"/";

    string save_fd = "/home/yuboz/data/input_data/videos/cali_test/"+video_name_str+"/input_result/augmentation1/H5_file_mask10";
    mkdir(save_fd.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    string save = save_fd+"/";
    string gt_fd1  = "/home/yuboz/data/input_data/videos/cali_test/"+video_name_str+"/input_result/augmentation1/Masks";
    string gt_fd = gt_fd1+"/";

    string weight_fd1 = "/home/yuboz/data/input_data/videos/cali_test/"+video_name_str+"/input_result/augmentation1/weight";
    string weight_fd = weight_fd1+"/";
    Mat im, im_gt,im_gt1,im_weight,im_weight1;
    string im_path,im_gt_path,im_out,weight_path;
    
    FloatType datatype(PredType::IEEE_F32BE);
    hsize_t  dimsf[4];    
    hsize_t dimsf_label[4];
    hsize_t dimsf_weight[4];
    int rank_data = 4;
    int rank_label = 4;
    int rank_weight = 4;
    H5File file;
    dimsf_label[0] = 1;
    dimsf_label[1] = 1;
    dimsf_label[2] = 380;
    dimsf_label[3] = 1030;
    dimsf[0] = 1;
    dimsf[1] = 3;
    dimsf[2] = 380;
    dimsf[3] = 1030;
    dimsf_weight[0] = 1;
    dimsf_weight[1] = 1;
    dimsf_weight[2] = 380;
    dimsf_weight[3] = 1030;   
    DataSpace dataspace1( rank_label, dimsf_label);
    DataSpace dataspace( rank_data, dimsf);
    DataSpace dataspace2( rank_weight, dimsf_weight);
    DataSet dataset,dataset1,dataset2;
    int num = 1;
    while(num<100000){
        im_path = fd+patch::to_string(num)+".png";
        im = imread(im_path,CV_LOAD_IMAGE_COLOR);
        if (im.data==NULL){
          cout<<"no data~!"<<endl;
          break;
        }
        im_gt_path = gt_fd+patch::to_string(num)+".png";
        
        im_gt1 = imread(im_gt_path,CV_LOAD_IMAGE_COLOR);
        cvtColor(im_gt1, im_gt, COLOR_BGR2GRAY);

        im_out = save+patch::to_string(num)+".h5";

        weight_path = weight_fd+patch::to_string(num)+".txt";
        std::ifstream file1(weight_path.c_str());

        double number;
        int row = 0;
        for(std::string line;getline(file1,line);row+=1){
            int col = 0;
            std::istringstream ss(line);
            while(ss>>number){
                weight_save[row][col] = number;
                if (!isdigit(ss.peek())){
                    ss.ignore();
                }
                col+=1;
            }
        }
        for (int i=0;i<380;i++){
            for(int j = 0;j<1030;j++){
              label_save[i][j]=(float)(im_gt.at<uchar>(i,j));
            }
        }
        for (int i=0;i<380;i++){
            for(int j = 0;j<1030;j++){
              im_save[0][i][j]=im.at<Vec3b>(i,j)[0];
              im_save[1][i][j]=im.at<Vec3b>(i,j)[1];
              im_save[2][i][j]=im.at<Vec3b>(i,j)[2];
            }
          } 
        H5File file(im_out,H5F_ACC_TRUNC);
        dataset= file.createDataSet( DATASET_NAME, datatype, dataspace );
        dataset1 = file.createDataSet( LABEL_NAME, datatype, dataspace1 );
        dataset2 = file.createDataSet( WEIGHT_NAME, datatype, dataspace2 );
        dataset1.write(label_save, PredType::NATIVE_FLOAT);
        dataset.write(im_save, PredType::NATIVE_FLOAT);
        dataset2.write(weight_save, PredType::NATIVE_FLOAT);
        num+=1;
        cout<<num<<endl;
        file.close();
    }

    return 0;
}
