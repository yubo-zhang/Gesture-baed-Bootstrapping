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
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/optflow.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/tracking.hpp"
#include "opencv2/ximgproc/seeds.hpp"
#include <opencv2/ximgproc.hpp>
//#include <direct.h>
#include "opencv2/video/background_segm.hpp"
#include "./bgsegm.hpp"
//#include "./optflow/include/optflow.hpp"
//#include"opencv2/bgsegm.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/video.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/video/tracking.hpp"
#include <sys/stat.h>
#include "H5Cpp.h"
#include <cstddef> 

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

#ifndef H5_NO_NAMESPACE
using namespace H5;
#endif

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

Mat frame;
//Mat fgMaskGMG;
Mat image;
Ptr<cv::bgsegm::BackgroundSubtractorGMG> GMG;
Ptr<cv::DenseOpticalFlow> DF;
int keyboard;
void processVideo(char* fd_list,char* gt_list);
bool is_opt_flow;
bool is_bg_seg;
bool is_gray;
const H5std_string  DATASET_NAME( "data" );
const H5std_string  LABEL_NAME( "label" );

// calculate input for optical flow
Mat drawOptFlowMap_x(const Mat& flow) //, Mat& cflowmap, int step,double, const Scalar& color)
{
  int thre = 10;
  int step = 1;
    //double max_x = 0;
  float rad_x =0;
  Mat cflow_x;
    /*
    for(int y = 0; y < flow.rows; y += step){
        for(int x = 0; x < flow.cols; x += step)
        {
          const Point2f& fxy = flow.at<Point2f>(y, x);
             rad_x = sqrt(fxy.x*fxy.x);
             if (rad_x>max_x){
               max_x = rad_x;
              }
         }
    }*/
         cflow_x = Mat::zeros(flow.rows,flow.cols,CV_32FC(1));   
         for(int y = 0; y < flow.rows; y += step){
          for(int x = 0; x < flow.cols; x += step)
          {

            const Point2f& fxy = flow.at<Point2f>(y, x);
            rad_x = sqrt(fxy.x*fxy.x);
             //if (max_x!=0){
            cflow_x.at<float>(Point(x,y)) = (float)(rad_x);
            //}else{
            // cflow_x.at<float>(Point(x,y)) = 0;
            // }
          }

        }
        return cflow_x;
      }

Mat drawOptFlowMap_y(const Mat& flow) //, Mat& cflowmap, int step,double, const Scalar& color)
{
  int thre = 10;
  int step = 1;
    //double max_y = 0;
  float rad_y = 0;
  Mat cflow_y;
   /*
   for(int y = 0; y < flow.rows; y += step){
        for(int x = 0; x < flow.cols; x += step)
        {
          const Point2f& fxy = flow.at<Point2f>(y, x);
             rad_y = sqrt(fxy.y*fxy.y);
             if (rad_y>max_y){
               max_y = rad_y;
              }
         }
     }
     */
     cflow_y = Mat::zeros(flow.rows,flow.cols,CV_32FC(1));
     for(int y = 0; y < flow.rows; y += step){
      for(int x = 0; x < flow.cols; x += step)
      {

        const Point2f& fxy = flow.at<Point2f>(y, x);
        rad_y = sqrt(fxy.y*fxy.y);
             //if(max_y!=0){
        cflow_y.at<float>(Point(x,y))=(float)(rad_y);
             //}else{
             //cflow_y.at<uchar>(Point(x,y)) = 0;
             //}

      }
    }
    return cflow_y;
  }
//end

//float im_to_save1[3][380][1030];
float im_to_save[2][380][1030];

  void processVideo(char* fd_list,char* gt_list) {

  //
  // The GMG part!!
  //
    char filename[256];

// the optical flow part

    Mat frame,cflow,uflow;
    Mat gt_frame;
    Mat gray,gray1;
    DF= cv::createOptFlow_DualTVL1();
    //Mat fgMaskGMG;
    Mat gray_input;
    Mat prevgray;

    Mat bg_show, dx_show,dy_show,gray_show;
    Mat bg_show1, dx_show1,dy_show1,gray_show1;

	// loop all the folders
    ifstream im_folder_txt;
    ifstream gt_folder_txt;
    ifstream im_txt;
    im_folder_txt.open(fd_list,ios::in);
    gt_folder_txt.open(gt_list,ios::in);
    string fd;
    string gt;
    string im_path;
    string gt_path;
    string hdf5_name;


    getline(gt_folder_txt,gt);
    size_t found = gt.find_first_of("+");
    im_path = gt.substr(0,found);
    size_t found2 = found+1;
    found = gt.find_first_of("+",found2);
    gt_path = gt.substr(found2,found-found2);
    found2 = found+1;
    found = gt.find_first_of("\\",found2);
    hdf5_name = gt.substr(found2,found-found2);



    int cha;
    int rank_data;
    int rank_label;
    FloatType datatype(PredType::IEEE_F32BE);
    
    hsize_t  dimsf[4];    
    hsize_t dimsf_label[4];
    rank_data = 4;
    rank_label = 4;
    int st_row = 380;
    int st_col = 1030;
    while(getline(im_folder_txt,fd)){
      char fd_to_read1[] = "../videos/cali_train/";
      char middle[30];
      char index[] = "index.txt";
      char im_list[200];
      
      //int eol = fd.find_first_of("\\");
      cout<<fd<<endl;
      //strncpy(middle,fd.c_str(),eol);
      strcpy(middle,fd.c_str());
      strcat(fd_to_read1,middle);
      
      char filename1[100];
      strcpy(filename1,fd_to_read1);
      char resultfd1[]="/input_result/result_bg";
      strcat(filename1,resultfd1);
      mkdir(filename1, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

      char filename2[100];
      strcpy(filename2,fd_to_read1);
      char resultfd2[]="/input_result/result_op";
      strcat(filename2,resultfd2);
      mkdir(filename2, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

      char filename3[100];
      strcpy(filename3,fd_to_read1);
      char resultfd3[]="/input_result/result_gray";
      strcat(filename3,resultfd3);
      mkdir(filename3, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

      char filename4[100];
      strcpy(filename4,fd_to_read1);
      char resultfd4[]="/input_result/result_bg_op";
      strcat(filename4,resultfd4);
      mkdir(filename4, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
     
      char filename4_1[100];
      strcpy(filename4_1,fd_to_read1);
      char resultfd4_1[]="/input_result/result_bg_op1";
      strcat(filename4_1,resultfd4_1);
      mkdir(filename4_1, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

     
      char filename5[100];
      strcpy(filename5,fd_to_read1);
      char resultfd5[]="/input_result/result_bg_gray";
      strcat(filename5,resultfd5);
      mkdir(filename5, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

      char filename6[100];
      strcpy(filename6,fd_to_read1);
      char resultfd6[]="/input_result/result_op_gray";
      strcat(filename6,resultfd6);
      mkdir(filename6, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

      char filename7[100];
      strcpy(filename7,fd_to_read1);
      char resultfd7[]="/input_result/result_bg_op_gray";
      strcat(filename7,resultfd7);
      mkdir(filename7, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

      char filename8[100];
      strcpy(filename8,fd_to_read1);
      char resultfd8[]="/input_result/result_bg_x";
      strcat(filename8,resultfd8);
      mkdir(filename8, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

      char filename9[100];
      strcpy(filename9,fd_to_read1);
      char resultfd9[]="/input_result/result_bg_x_gray";
      strcat(filename9,resultfd9);
      mkdir(filename9, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

      strcpy(im_list,fd_to_read1);
      char back[] = "/input_result/2/%d.png";
      strcat(im_list,back);
     
      string filename1_str(filename1);
      string filename2_str(filename2);
      string filename3_str(filename3);
      string filename4_str(filename4);

      string filename4_1_str(filename4_1);

      string filename5_str(filename5);
      string filename6_str(filename6);
      string filename7_str(filename7);
      string filename8_str(filename8);
      string filename9_str(filename9);

      ofstream file_to_save;
      //float* im_to_save;
      //float*label_to_save;

      int num = 0;
      while(num< 1000){
        sprintf(filename,im_list,num);
        cout<< filename<<endl;
        frame = imread(filename,CV_LOAD_IMAGE_COLOR);
        int row = frame.rows;
        int col = frame.cols;
        int row_up = (row-st_row)*3/4;
        int row_down = (row-st_row)/4;
        int col_up = (col-st_col)*3/4;
        int col_down = (col-st_col)/4;

        if (frame.data==NULL){
          cout<<"no data~!"<<endl;
          break;
        }

//std::string imname41 = filename4_1_str+"/"+hdf5_name+".h5";
//std::string imname41_1 = filename4_1_str+"/"+hdf5_name+"_1"+".h5";
//std::string imname41_2 = filename4_1_str+"/"+hdf5_name+"_2"+".h5";

        std::string imname1 = filename1_str+"/"+hdf5_name+".h5";
        std::string imname2 = filename2_str+"/"+hdf5_name+".h5";
        std::string imname3 = filename3_str+"/"+hdf5_name+".h5";
        std::string imname4 = filename4_str+"/"+hdf5_name+".h5";
        std::string imname5 = filename5_str+"/"+hdf5_name+".h5";
        std::string imname6 = filename6_str+"/"+hdf5_name+".h5";
        std::string imname7 = filename7_str+"/"+hdf5_name+".h5";
        std::string imname8 = filename8_str+"/"+hdf5_name+".h5";
        std::string imname9 = filename9_str+"/"+hdf5_name+".h5";

        std::string imname1_1 = filename1_str+"/"+hdf5_name+"_1"+".h5";
        std::string imname2_1 = filename2_str+"/"+hdf5_name+"_1"+".h5";
        std::string imname3_1 = filename3_str+"/"+hdf5_name+"_1"+".h5";
        std::string imname4_1 = filename4_str+"/"+hdf5_name+"_1"+".h5";
        std::string imname5_1 = filename5_str+"/"+hdf5_name+"_1"+".h5";
        std::string imname6_1 = filename6_str+"/"+hdf5_name+"_1"+".h5";
        std::string imname7_1 = filename7_str+"/"+hdf5_name+"_1"+".h5";
        std::string imname8_1 = filename8_str+"/"+hdf5_name+"_1"+".h5";
        std::string imname9_1 = filename9_str+"/"+hdf5_name+"_1"+".h5";

        std::string imname1_2 = filename1_str+"/"+hdf5_name+"_2"+".h5";
        std::string imname2_2 = filename2_str+"/"+hdf5_name+"_2"+".h5";
        std::string imname3_2 = filename3_str+"/"+hdf5_name+"_2"+".h5";
        std::string imname4_2 = filename4_str+"/"+hdf5_name+"_2"+".h5";
        std::string imname5_2 = filename5_str+"/"+hdf5_name+"_2"+".h5";
        std::string imname6_2 = filename6_str+"/"+hdf5_name+"_2"+".h5";
        std::string imname7_2 = filename7_str+"/"+hdf5_name+"_2"+".h5";
        std::string imname8_2 = filename8_str+"/"+hdf5_name+"_2"+".h5";
        std::string imname9_2 = filename9_str+"/"+hdf5_name+"_2"+".h5";

        Mat fgMaskGMG = Mat(row,col,CV_32FC1);
        GMG->apply(frame, fgMaskGMG);
        bg_show = fgMaskGMG.clone();
        bg_show*=200;
        bg_show.convertTo(bg_show1,CV_8UC1);
        //imshow("background", bg_show1);
        //keyboard = waitKey(5);

         // the gray part!!
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        // the optical flow part!!
        if( prevgray.empty()||gray.rows!=prevgray.rows||gray.cols!=prevgray.cols){
          gray.copyTo(prevgray);
          //cout<<prevgray.rows<<endl;
          //cout<<prevgray.cols<<endl;
          //cout<<gray.rows<<endl;
          //cout<<gray.cols<<endl;
        }

         DF->calc(prevgray,gray,uflow);
         //cvtColor(prevgray, cflow, COLOR_GRAY2BGR);
         Mat cflow_x;
         Mat cflow_y;
        cflow_x = drawOptFlowMap_x(uflow);//, cflow, a16, 1.5, Scalar(0, 255, 0));
        cflow_y = drawOptFlowMap_y(uflow);

        dx_show = cflow_x.clone();
        dx_show*=100;
        dx_show.convertTo(dx_show1,CV_8UC1);
        //imshow("dx", dx_show1);
        //keyboard = waitKey(5);

        dy_show = cflow_y.clone();
        dy_show*=100;
        dy_show.convertTo(dy_show1,CV_8UC1);
       // imshow("dy", dy_show1);
        //keyboard = waitKey( 5);



        gray1 = gray.clone();
        gray1.convertTo(gray1,CV_32FC1);


        gray_show = gray1.clone();
        gray_show.convertTo(gray_show1,CV_8UC1);
        //imshow("gray", gray_show1);
        //keyboard = waitKey(5);
       
        //cout<< imname1<<endl;

        string filename_str(filename);
        if (!(filename_str.compare(im_path))){
          Mat gt_frame,gt_frame1;// = Mat(row,col,CV_32FC1);
          gt_frame = imread(gt_path,CV_LOAD_IMAGE_COLOR);
          cvtColor(gt_frame, gt_frame1, COLOR_BGR2GRAY);

          //label_to_save = (float*)(gt_frame1.data);
          //im_to_save = (float*)(fgMaskGMG.data);
          //im_to_save = gt_frame.ptr<float>(0);
         // label_to_save = gt_frame.ptr<float>(0);

         int corner_row1 = (rand()%(row_up-row_down))+row_down;
         int corner_col1 = (rand()%(col_up-col_down))+col_down;
         int corner_row2 = (rand()%(row_up-row_down))+row_down;
         int corner_col2 = (rand()%(col_up-col_down))+col_down;
         int corner_row3 = (rand()%(row_up-row_down))+row_down;
         int corner_col3 = (rand()%(col_up-col_down))+col_down;

          //H5File file(imname1 , H5F_ACC_TRUNC);
          float label_to_save1[st_row][st_col];
          float label_to_save2[st_row][st_col];
          float label_to_save3[st_row][st_col];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              label_to_save1[i][j]=(float)(gt_frame1.at<uchar>(i+corner_row1,j+corner_col1));
              label_to_save2[i][j]=(float)(gt_frame1.at<uchar>(i+corner_row2,j+corner_col2));
              label_to_save3[i][j]=(float)(gt_frame1.at<uchar>(i+corner_row3,j+corner_col3));
            }
          }
          dimsf_label[0] = 1;
          dimsf_label[1] = 1;
          dimsf_label[2] = st_row;
          dimsf_label[3] = st_col;
          DataSpace dataspace1( rank_label, dimsf_label);
          //cout<< corner_row1<<endl;
          //cout<< corner_col1<<endl;
          //cout<< corner_row2<<endl;
          //cout<< corner_col2<<endl;
          //cout<< corner_row3<<endl;
          //cout<< corner_col3<<endl;
          
          /*
          //for different test kind
          //bg
          //corp1
          DataSet dataset1 = file.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          dataset1.write(label_to_save1, PredType::NATIVE_FLOAT);
          float im_to_save[st_row][st_col];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save[i][j]=fgMaskGMG.at<float>(i+corner_row1,j+corner_col1);
            }
          }          
          dimsf[0] = 1;
          dimsf[1] = 1;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          DataSpace dataspace( rank_data, dimsf);        
          DataSet dataset= file.createDataSet( DATASET_NAME, datatype, dataspace );
          dataset.write(im_to_save, PredType::NATIVE_FLOAT);

          //crop2
          H5File file1(imname1_1 , H5F_ACC_TRUNC);
          dataset1 = file1.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          dataset1.write(label_to_save2, PredType::NATIVE_FLOAT);
          //float im_to_save[st_row][st_col];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save[i][j]=fgMaskGMG.at<float>(i+corner_row2,j+corner_col2);
            }
          }          
          dimsf[0] = 1;
          dimsf[1] = 1;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          //DataSpace dataspace( rank_data, dimsf);        
          dataset= file1.createDataSet( DATASET_NAME, datatype, dataspace );
          dataset.write(im_to_save, PredType::NATIVE_FLOAT);

          //crop3
          H5File file2(imname1_2 , H5F_ACC_TRUNC);
          dataset1 = file2.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          dataset1.write(label_to_save3, PredType::NATIVE_FLOAT);
          //float im_to_save[st_row][st_col];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save[i][j]=fgMaskGMG.at<float>(i+corner_row3,j+corner_col3);
            }
          }          
          dimsf[0] = 1;
          dimsf[1] = 1;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          //DataSpace dataspace( rank_data, dimsf);        
          dataset= file2.createDataSet( DATASET_NAME, datatype, dataspace );
          dataset.write(im_to_save, PredType::NATIVE_FLOAT);
          
           
          //op
          //crop1
          H5File file3(imname2 , H5F_ACC_TRUNC);
          DataSet dataset1 = file3.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          dataset1.write(label_to_save1, PredType::NATIVE_FLOAT);
          //cha=2;
          //float im_to_save[st_row][st_col][cha];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save[0][i][j]=80*(cflow_x.at<float>(i+corner_row1,j+corner_col1));
              im_to_save[1][i][j]=80*(cflow_y.at<float>(i+corner_row1,j+corner_col1));
            }
          } 
          dimsf[0] = 1;
          dimsf[1] = 2;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          DataSpace dataspace( rank_data, dimsf);        
          DataSet dataset= file3.createDataSet( DATASET_NAME, datatype, dataspace );
          dataset.write(im_to_save, PredType::NATIVE_FLOAT);

          //crop2
          H5File file4(imname2_1 , H5F_ACC_TRUNC);
          dataset1 = file4.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          dataset1.write(label_to_save2, PredType::NATIVE_FLOAT);
          //cha=2;
          //float im_to_save[st_row][st_col][cha];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save[0][i][j]=80*(cflow_x.at<float>(i+corner_row2,j+corner_col2));
              im_to_save[1][i][j]=80*(cflow_y.at<float>(i+corner_row2,j+corner_col2));
            }
          } 
          dimsf[0] = 1;
          dimsf[1] = 2;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          //DataSpace dataspace( rank_data, dimsf);        
          dataset= file4.createDataSet( DATASET_NAME, datatype, dataspace );
          dataset.write(im_to_save, PredType::NATIVE_FLOAT);

          //crop3
          H5File file5(imname2_2 , H5F_ACC_TRUNC);
          dataset1 = file5.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          dataset1.write(label_to_save3, PredType::NATIVE_FLOAT);
          //cha=2;
          //float im_to_save[st_row][st_col][cha];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save[0][i][j]=80*(cflow_x.at<float>(i+corner_row3,j+corner_col3));
              im_to_save[1][i][j]=80*(cflow_y.at<float>(i+corner_row3,j+corner_col3));
            }
          } 
          dimsf[0] = 1;
          dimsf[1] = 2;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          //DataSpace dataspace( rank_data, dimsf);        
          dataset= file5.createDataSet( DATASET_NAME, datatype, dataspace );
          dataset.write(im_to_save, PredType::NATIVE_FLOAT);
          
          
          //gray
          //crop1
          H5File file6(imname3 , H5F_ACC_TRUNC);
          DataSet dataset1 = file6.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          dataset1.write(label_to_save1, PredType::NATIVE_FLOAT);
          //float im_to_save[st_row][st_col];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save[i][j]=gray1.at<float>(i+corner_row1,j+corner_col1);
            }
          }          
          dimsf[0] = 1;
          dimsf[1] = 1;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          DataSpace dataspace( rank_data, dimsf);        
          DataSet dataset= file6.createDataSet( DATASET_NAME, datatype, dataspace );
          dataset.write(im_to_save, PredType::NATIVE_FLOAT);
          //crop2
          H5File file7(imname3_1 , H5F_ACC_TRUNC);
          dataset1 = file7.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          dataset1.write(label_to_save2, PredType::NATIVE_FLOAT);
          //float im_to_save[st_row][st_col];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save[i][j]=gray1.at<float>(i+corner_row2,j+corner_col2);
            }
          }          
          dimsf[0] = 1;
          dimsf[1] = 1;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          //DataSpace dataspace( rank_data, dimsf);        
          dataset= file7.createDataSet( DATASET_NAME, datatype, dataspace );
          dataset.write(im_to_save, PredType::NATIVE_FLOAT);

          //crop3
          H5File file8(imname3_2 , H5F_ACC_TRUNC);
          dataset1 = file8.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          dataset1.write(label_to_save3, PredType::NATIVE_FLOAT);
          //float im_to_save[st_row][st_col];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save[i][j]=gray1.at<float>(i+corner_row3,j+corner_col3);
            }
          }          
          dimsf[0] = 1;
          dimsf[1] = 1;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          //DataSpace dataspace( rank_data, dimsf);        
          dataset= file8.createDataSet( DATASET_NAME, datatype, dataspace );
          dataset.write(im_to_save, PredType::NATIVE_FLOAT);
           /*
                    //bg_op
          //crop1
          H5File file9(imname41 , H5F_ACC_TRUNC);
          DataSet dataset1 = file9.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          
          dataset1.write(label_to_save1, PredType::NATIVE_FLOAT);
          cha=3;
          
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save1[0][i][j]=250*(fgMaskGMG.at<float>(i+corner_row1,j+corner_col1));
              //cout<<"bg"<<endl;
              im_to_save1[1][i][j]=80*(cflow_x.at<float>(i+corner_row1,j+corner_col1));
              //cout<<"x"<<endl;
              im_to_save1[2][i][j]=80*(cflow_y.at<float>(i+corner_row1,j+corner_col1));
              //cout<<"y"<<endl;
            }
          } 
          dimsf[0] = 1;
          dimsf[1] = 3;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          DataSpace dataspace( rank_data, dimsf);      
          DataSet dataset= file9.createDataSet( DATASET_NAME, datatype, dataspace ); 
          dataset.write(im_to_save1, PredType::NATIVE_FLOAT);
          
          //crop2
          H5File file10(imname41_1 , H5F_ACC_TRUNC);
          dataset1 = file10.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          dataset1.write(label_to_save2, PredType::NATIVE_FLOAT);
          cha=3;
          //float im_to_save[st_row][st_col][cha];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save1[0][i][j]=250*(fgMaskGMG.at<float>(i+corner_row2,j+corner_col2));
              im_to_save1[1][i][j]=80*(cflow_x.at<float>(i+corner_row2,j+corner_col2));
              im_to_save1[2][i][j]=80*(cflow_y.at<float>(i+corner_row2,j+corner_col2));
            }
          } 
          dimsf[0] = 1;
          dimsf[1] = 3;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          //DataSpace dataspace( rank_data, dimsf);        
          dataset= file10.createDataSet( DATASET_NAME, datatype, dataspace );
          dataset.write(im_to_save1, PredType::NATIVE_FLOAT);

          //crop3
          H5File file11(imname41_2 , H5F_ACC_TRUNC);
          dataset1 = file11.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          dataset1.write(label_to_save3, PredType::NATIVE_FLOAT);
          cha=3;
          //loat im_to_save[st_row][st_col][cha];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save1[0][i][j]=250*(fgMaskGMG.at<float>(i+corner_row3,j+corner_col3));
              im_to_save1[1][i][j]=80*(cflow_x.at<float>(i+corner_row3,j+corner_col3));
              im_to_save1[2][i][j]=80*(cflow_y.at<float>(i+corner_row3,j+corner_col3));
            }
          } 
          dimsf[0] = 1;
          dimsf[1] = 3;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          //DataSpace dataspace( rank_data, dimsf);        
          dataset= file11.createDataSet( DATASET_NAME, datatype, dataspace );
          dataset.write(im_to_save1, PredType::NATIVE_FLOAT);
          
         
          //bg_gray
          //crop1
          H5File file12(imname5 , H5F_ACC_TRUNC);
          DataSet dataset1 = file12.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          dataset1.write(label_to_save1, PredType::NATIVE_FLOAT);
          //cha=2;
          //float im_to_save[st_row][col][cha];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save[0][i][j]=250*(fgMaskGMG.at<float>(i+corner_row1,j+corner_col1));
              im_to_save[1][i][j]=gray1.at<float>(i+corner_row1,j+corner_col1);
            }
          } 
          dimsf[0] = 1;
          dimsf[1] = 2;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          DataSpace dataspace( rank_data, dimsf);        
          DataSet dataset= file12.createDataSet( DATASET_NAME, datatype, dataspace );
          dataset.write(im_to_save, PredType::NATIVE_FLOAT);

          //crop2
          H5File file13(imname5_1 , H5F_ACC_TRUNC);
          dataset1 = file13.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          dataset1.write(label_to_save2, PredType::NATIVE_FLOAT);
          //cha=2;
          //float im_to_save[st_row][col][cha];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save[0][i][j]=250*(fgMaskGMG.at<float>(i+corner_row2,j+corner_col2));
              im_to_save[1][i][j]=gray1.at<float>(i+corner_row2,j+corner_col2);
            }
          } 
          dimsf[0] = 1;
          dimsf[1] = 2;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          //DataSpace dataspace( rank_data, dimsf);        
          dataset= file13.createDataSet( DATASET_NAME, datatype, dataspace );
          dataset.write(im_to_save, PredType::NATIVE_FLOAT);

          //crop3
          H5File file14(imname5_2 , H5F_ACC_TRUNC);
          dataset1 = file14.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          dataset1.write(label_to_save3, PredType::NATIVE_FLOAT);
          //cha=2;
          //float im_to_save[st_row][col][cha];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save[0][i][j]=250*(fgMaskGMG.at<float>(i+corner_row3,j+corner_col3));
              im_to_save[1][i][j]=gray1.at<float>(i+corner_row3,j+corner_col3);
            }
          } 
          dimsf[0] = 1;
          dimsf[1] = 2;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          //DataSpace dataspace( rank_data, dimsf);        
          dataset= file14.createDataSet( DATASET_NAME, datatype, dataspace );
          dataset.write(im_to_save, PredType::NATIVE_FLOAT);
/*
          //op_gray
          //crop1
          H5File file15(imname6 , H5F_ACC_TRUNC);
          DataSet dataset1 = file15.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          dataset1.write(label_to_save1, PredType::NATIVE_FLOAT);
          //cha=3;
          //float im_to_save[st_row][st_col][cha];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save[0][i][j]=cflow_x.at<float>(i+corner_row1,j+corner_col1);
              im_to_save[1][i][j]=cflow_y.at<float>(i+corner_row1,j+corner_col1);
              im_to_save[2][i][j]=gray1.at<float>(i+corner_row1,j+corner_col1);
            }
          } 
          dimsf[0] = 1;
          dimsf[1] = 3;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          DataSpace dataspace( rank_data, dimsf);        
          DataSet dataset= file15.createDataSet( DATASET_NAME, datatype, dataspace );
          dataset.write(im_to_save, PredType::NATIVE_FLOAT);

          //crop2
          H5File file16(imname6_1 , H5F_ACC_TRUNC);
          dataset1 = file16.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          dataset1.write(label_to_save2, PredType::NATIVE_FLOAT);
          //cha=3;
          //float im_to_save[st_row][st_col][cha];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save[0][i][j]=cflow_x.at<float>(i+corner_row2,j+corner_col2);
              im_to_save[1][i][j]=cflow_y.at<float>(i+corner_row2,j+corner_col2);
              im_to_save[2][i][j]=gray1.at<float>(i+corner_row2,j+corner_col2);
            }
          } 
          dimsf[0] = 1;
          dimsf[1] = 3;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          //DataSpace dataspace( rank_data, dimsf);        
          dataset= file16.createDataSet( DATASET_NAME, datatype, dataspace );
          dataset.write(im_to_save, PredType::NATIVE_FLOAT);

          //crop3
          H5File file17(imname6_2 , H5F_ACC_TRUNC);
          dataset1 = file17.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          dataset1.write(label_to_save3, PredType::NATIVE_FLOAT);
          //cha=3;
          //float im_to_save[st_row][st_col][cha];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save[0][i][j]=cflow_x.at<float>(i+corner_row3,j+corner_col3);
              im_to_save[1][i][j]=cflow_y.at<float>(i+corner_row3,j+corner_col3);
              im_to_save[2][i][j]=gray1.at<float>(i+corner_row3,j+corner_col3);
            }
          } 
          dimsf[0] = 1;
          dimsf[1] = 3;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          //DataSpace dataspace( rank_data, dimsf);        
          dataset= file17.createDataSet( DATASET_NAME, datatype, dataspace );
          dataset.write(im_to_save, PredType::NATIVE_FLOAT);
           
          //bg_op_gray
          //crop1
          H5File file18(imname7 , H5F_ACC_TRUNC);
          DataSet dataset1 = file18.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          dataset1.write(label_to_save1, PredType::NATIVE_FLOAT);
          //cha=4;
          //float im_to_save[st_row][st_col][cha];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save[0][i][j]=fgMaskGMG.at<float>(i+corner_row1,j+corner_col1);
              im_to_save[1][i][j]=cflow_x.at<float>(i+corner_row1,j+corner_col1);
              im_to_save[2][i][j]=cflow_y.at<float>(i+corner_row1,j+corner_col1);
              im_to_save[3][i][j]=gray1.at<float>(i+corner_row1,j+corner_col1);
            }
          } 
          dimsf[0] = 1;
          dimsf[1] = 4;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          DataSpace dataspace( rank_data, dimsf);        
          DataSet dataset= file18.createDataSet( DATASET_NAME, datatype, dataspace );
          dataset.write(im_to_save, PredType::NATIVE_FLOAT);

          //crop2
          H5File file19(imname7_1 , H5F_ACC_TRUNC);
          dataset1 = file19.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          dataset1.write(label_to_save2, PredType::NATIVE_FLOAT);
          //cha=4;
          //float im_to_save[st_row][st_col][cha];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save[0][i][j]=fgMaskGMG.at<float>(i+corner_row2,j+corner_col2);
              im_to_save[1][i][j]=cflow_x.at<float>(i+corner_row2,j+corner_col2);
              im_to_save[2][i][j]=cflow_y.at<float>(i+corner_row2,j+corner_col2);
              im_to_save[3][i][j]=gray1.at<float>(i+corner_row2,j+corner_col2);
            }
          } 
          dimsf[0] = 1;
          dimsf[1] = 4;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          //DataSpace dataspace( rank_data, dimsf);        
          dataset= file19.createDataSet( DATASET_NAME, datatype, dataspace );
          dataset.write(im_to_save, PredType::NATIVE_FLOAT);

          //crop3
          H5File file20(imname7_2 , H5F_ACC_TRUNC);
          dataset1 = file20.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          dataset1.write(label_to_save3, PredType::NATIVE_FLOAT);
          //cha=4;
          //float im_to_save[st_row][st_col][cha];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save[0][i][j]=fgMaskGMG.at<float>(i+corner_row3,j+corner_col3);
              im_to_save[1][i][j]=cflow_x.at<float>(i+corner_row3,j+corner_col3);
              im_to_save[2][i][j]=cflow_y.at<float>(i+corner_row3,j+corner_col3);
              im_to_save[3][i][j]=gray1.at<float>(i+corner_row3,j+corner_col3);
            }
          } 
          dimsf[0] = 1;
          dimsf[1] = 4;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          //DataSpace dataspace( rank_data, dimsf);        
          dataset= file20.createDataSet( DATASET_NAME, datatype, dataspace );
          dataset.write(im_to_save, PredType::NATIVE_FLOAT);

          //bg_x
          //crop1
          H5File file21(imname8 , H5F_ACC_TRUNC);
          DataSet dataset1 = file21.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          dataset1.write(label_to_save1, PredType::NATIVE_FLOAT);
          //cha=2;
          //float im_to_save[st_row][col][cha];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save1[0][i][j]=fgMaskGMG.at<float>(i+corner_row1,j+corner_col1);
              im_to_save1[1][i][j]=cflow_x.at<float>(i+corner_row1,j+corner_col1);
            }
          } 
          dimsf[0] = 1;
          dimsf[1] = 2;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          DataSpace dataspace( rank_data, dimsf);        
          DataSet dataset= file21.createDataSet( DATASET_NAME, datatype, dataspace );
          dataset.write(im_to_save1, PredType::NATIVE_FLOAT);

          //crop2
          H5File file22(imname8_1 , H5F_ACC_TRUNC);
          dataset1 = file22.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          dataset1.write(label_to_save2, PredType::NATIVE_FLOAT);
          //cha=2;
          //float im_to_save[st_row][col][cha];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save1[0][i][j]=fgMaskGMG.at<float>(i+corner_row2,j+corner_col2);
              im_to_save1[0][i][j]=cflow_x.at<float>(i+corner_row2,j+corner_col2);
            }
          } 
          dimsf[0] = 1;
          dimsf[1] = 2;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          //DataSpace dataspace( rank_data, dimsf);        
          dataset= file22.createDataSet( DATASET_NAME, datatype, dataspace );
          dataset.write(im_to_save1, PredType::NATIVE_FLOAT);

          //crop3
          H5File file23(imname8_2 , H5F_ACC_TRUNC);
          dataset1 = file23.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          dataset1.write(label_to_save3, PredType::NATIVE_FLOAT);
          cha=2;
          float im_to_save[st_row][col][cha];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save1[0][i][j]=fgMaskGMG.at<float>(i+corner_row3,j+corner_col3);
              im_to_save1[1][i][j]=cflow_x.at<float>(i+corner_row3,j+corner_col3);
            }
          } 
          dimsf[0] = 1;
          dimsf[1] = 2;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          //DataSpace dataspace( rank_data, dimsf);        
          dataset= file23.createDataSet( DATASET_NAME, datatype, dataspace );
          dataset.write(im_to_save1, PredType::NATIVE_FLOAT);
          */
          //bg_x_gray
          //crop1
          H5File file24(imname9 , H5F_ACC_TRUNC);
          DataSet dataset1 = file24.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          dataset1.write(label_to_save1, PredType::NATIVE_FLOAT);
          //cha=3;
          //float im_to_save[st_row][st_col][cha];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save[0][i][j]=fgMaskGMG.at<float>(i+corner_row1,j+corner_col1);
              im_to_save[1][i][j]=cflow_x.at<float>(i+corner_row1,j+corner_col1);
              im_to_save[2][i][j]=gray1.at<float>(i+corner_row1,j+corner_col1);
            }
          } 
          dimsf[0] = 1;
          dimsf[1] = 3;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          DataSpace dataspace( rank_data, dimsf);        
          DataSet dataset= file24.createDataSet( DATASET_NAME, datatype, dataspace );
          dataset.write(im_to_save, PredType::NATIVE_FLOAT);

          //crop2
          H5File file25(imname9_1 , H5F_ACC_TRUNC);
          dataset1 = file25.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          dataset1.write(label_to_save2, PredType::NATIVE_FLOAT);
          //cha=3;
          //float im_to_save[st_row][st_col][cha];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save[0][i][j]=fgMaskGMG.at<float>(i+corner_row2,j+corner_col2);
              im_to_save[1][i][j]=cflow_x.at<float>(i+corner_row2,j+corner_col2);
              im_to_save[2][i][j]=gray1.at<float>(i+corner_row2,j+corner_col2);
            }
          } 
          dimsf[0] = 1;
          dimsf[1] = 3;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          //DataSpace dataspace( rank_data, dimsf);        
          dataset= file25.createDataSet( DATASET_NAME, datatype, dataspace );
          dataset.write(im_to_save, PredType::NATIVE_FLOAT);

          //crop3
          H5File file26(imname9_2 , H5F_ACC_TRUNC);
          dataset1 = file26.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          dataset1.write(label_to_save3, PredType::NATIVE_FLOAT);
          //cha=3;
          //float im_to_save[st_row][st_col][cha];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save[0][i][j]=fgMaskGMG.at<float>(i+corner_row3,j+corner_col3);
              im_to_save[1][i][j]=cflow_x.at<float>(i+corner_row3,j+corner_col3);
              im_to_save[2][i][j]=gray1.at<float>(i+corner_row3,j+corner_col3);
            }
          } 
          dimsf[0] = 1;
          dimsf[1] = 3;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          //DataSpace dataspace( rank_data, dimsf);        
          dataset= file26.createDataSet( DATASET_NAME, datatype, dataspace );
          dataset.write(im_to_save, PredType::NATIVE_FLOAT);
          */

          
          //end of test for different test kind
          getline(gt_folder_txt,gt);
          size_t found = gt.find_first_of("+");
          im_path = gt.substr(0,found);
          size_t found2 = found+1;
          found = gt.find_first_of("+",found2);
          gt_path = gt.substr(found2,found-found2);
          found2 = found+1;
          found = gt.find_first_of("\\",found2);
          hdf5_name = gt.substr(found2,found-found2);
          //cout<<im_path<<endl;
          //cout<<gt_path<<endl;
          //cout<<hdf5_name<<endl;
        }


/*
    //fgMaskGMG.convertTo(fgMaskGMG, CV_64FC3); 
    const char* filename4 = "/home/claire/stable/input_result1/result_bg";
    mkdir(filename4, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    const char* filename5 = "/home/claire/stable/input_result1/result_op";
    mkdir(filename5, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    const char* filename6 = "/home/claire/stable/input_result1/result_gray";
    mkdir(filename6, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    const char* filename7 = "/home/claire/stable/input_result1/result_bg_op";
    mkdir(filename7, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    const char* filename8 = "/home/claire/stable/input_result1/result_bg_gray";
    mkdir(filename8, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    const char* filename9 = "/home/claire/stable/input_result1/result_op_gray";
    mkdir(filename9, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    const char* filename10 = "/home/claire/stable/input_result1/result_bg_op_gray";
    mkdir(filename10, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    ofstream file_to_save;
    float* im_to_save;
   while(num< 1000){
    std::sprintf(filename,"/home/claire/stable/input_result1/2/%d.png",num);
    frame = imread(filename,CV_LOAD_IMAGE_COLOR);
    //frame.convertTo(frame, CV_64FC3);
    int row = frame.rows;
    int col = frame.cols;
    int cha;   
    if (frame.data==NULL){
     break;
     }
     std::string imname4 = "/home/claire/stable/input_result1/result_bg/"+patch::to_string(num)+".binary";
     std::string imname5 = "/home/claire/stable/input_result1/result_op/"+patch::to_string(num)+".binary";
     std::string imname6 = "/home/claire/stable/input_result1/result_gray/"+patch::to_string(num)+".binary";
     std::string imname7 = "/home/claire/stable/input_result1/result_bg_op/"+patch::to_string(num)+".binary";
     std::string imname8 = "/home/claire/stable/input_result1/result_bg_gray/"+patch::to_string(num)+".binary";
     std::string imname9 = "/home/claire/stable/input_result1/result_op_gray/"+patch::to_string(num)+".binary";
     std::string imname10 = "/home/claire/stable/input_result1/result_bg_op_gray/"+patch::to_string(num)+".binary";

     GMG->apply(frame, fgMaskGMG);

     bg_show = fgMaskGMG.clone();
     bg_show.convertTo(bg_show1,CV_8UC1);
     imshow("background", bg_show1);
     keyboard = waitKey( 5);
     //imwrite(imname,fgMaskGMG);
     //get the input from the keyboard
     
      
    // the gray part!!
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    


    // the optical flow part!!
    if( prevgray.empty()){
           gray.copyTo(prevgray);
       }
       DF->calc(prevgray,gray,uflow);
      //cvtColor(prevgray, cflow, COLOR_GRAY2BGR);
      Mat cflow_x;
      Mat cflow_y;
      cflow_x = drawOptFlowMap_x(uflow);//, cflow, a16, 1.5, Scalar(0, 255, 0));
      cflow_y = drawOptFlowMap_y(uflow);
       
      dx_show = cflow_x.clone();
      dx_show.convertTo(dx_show1,CV_8UC1);
      imshow("dx", dx_show1)
      keyboard = waitKey( 5);
       
      dy_show = cflow_y.clone();
      dy_show.convertTo(dy_show1,CV_8UC1);
      imshow("dy", dy_show1)
      keyboard = waitKey( 5);



      gray1 = gray.clone();
      gray1.convertTo(gray1,CV_32FC1);


      gray_show = gray1.clone();
      gray_show.convertTo(gray_show1,CV_8UC1);
      imshow("gray", gray_show1)
       keyboard = waitKey(5);

      // the tenth folder
      cha = 4;
      Mat output_mat10(row,col,CV_32FC4);
         for(int i = 0;i<row;i++){
          for(int j = 0; j<col;j++){
            output_mat10.at<Vec4f>(i+corner_row1,j+corner_col1)[0] = fgMaskGMG.at<float>(i+corner_row1,j+corner_col1);
            output_mat10.at<Vec4f>(i+corner_row1,j+corner_col1)[1] = cflow_x.at<float>(i+corner_row1,j+corner_col1);
            output_mat10.at<Vec4f>(i+corner_row1,j+corner_col1)[2] = cflow_y.at<float>(i+corner_row1,j+corner_col1);
            output_mat10.at<Vec4f>(i+corner_row1,j+corner_col1)[3] = gray1.at<float>(i+corner_row1,j+corner_col1);
          }
         }
        im_to_save = (float*)(output_mat10.data);
        const char* imname10_1;
        imname10_1=imname10.c_str();
        file_to_save.open(imname10_1,ios::out | ios::binary);
        //ifstream file_to_save(imname10,ios::in | ios:: binary);
        file_to_save.write((char*)&row, sizeof(int));
        file_to_save.write((char*)&col, sizeof(int));
        file_to_save.write((char*)&cha, sizeof(int));
        file_to_save.write((char*)& im_to_save, row*col*cha*sizeof(float));
        file_to_save.close();


      //the ninth folder
        cha = 3;
      Mat output_mat9(row,col,CV_32FC3);
         for(int i = 0;i<row;i++){
          for(int j = 0; j<col;j++){
            output_mat9.at<Vec3f>(i+corner_row1,j+corner_col1)[0] = cflow_x.at<float>(i+corner_row1,j+corner_col1);
            output_mat9.at<Vec3f>(i+corner_row1,j+corner_col1)[1] = cflow_y.at<float>(i+corner_row1,j+corner_col1);
            output_mat9.at<Vec3f>(i+corner_row1,j+corner_col1)[2] = gray1.at<float>(i+corner_row1,j+corner_col1);
          }
         }
        im_to_save = (float*)(output_mat9.data);
        const char* imname9_1;
        imname9_1=imname9.c_str();
        file_to_save.open(imname9_1,ios::out | ios::binary);
        //ifstream file_to_save(imname10,ios::in | ios:: binary);
        file_to_save.write((char*)&row, sizeof(int));
        file_to_save.write((char*)&col, sizeof(int));
        file_to_save.write((char*)&cha, sizeof(int));
        file_to_save.write((char*)& im_to_save, row*col*cha*sizeof(float));
        file_to_save.close();  

      // the eight folder
        cha = 2;
      Mat output_mat8(row,col,CV_32FC2);
         for(int i = 0;i<row;i++){
          for(int j = 0; j<col;j++){
            output_mat8.at<Vec2f>(i+corner_row1,j+corner_col1)[0] = fgMaskGMG.at<float>(i+corner_row1,j+corner_col1);
            output_mat8.at<Vec2f>(i+corner_row1,j+corner_col1)[1] = gray1.at<float>(i+corner_row1,j+corner_col1);
          }
         }
        im_to_save = (float*)(output_mat8.data);
        const char* imname8_1;
        imname8_1=imname8.c_str();
        file_to_save.open(imname8_1,ios::out | ios::binary);
        //ifstream file_to_save(imname10,ios::in | ios:: binary);
        file_to_save.write((char*)&row, sizeof(int));
        file_to_save.write((char*)&col, sizeof(int));
        file_to_save.write((char*)&cha, sizeof(int));
        file_to_save.write((char*)& im_to_save, row*col*cha*sizeof(float));
        file_to_save.close(); 

      // the seventh folder
        cha = 3;
      Mat output_mat7(row,col,CV_32FC3);
         for(int i = 0;i<row;i++){
          for(int j = 0; j<col;j++){
            output_mat7.at<Vec3f>(i+corner_row1,j+corner_col1)[0] = fgMaskGMG.at<float>(i+corner_row1,j+corner_col1);
            output_mat7.at<Vec3f>(i+corner_row1,j+corner_col1)[1] = cflow_x.at<float>(i+corner_row1,j+corner_col1);
            output_mat7.at<Vec3f>(i+corner_row1,j+corner_col1)[2] = cflow_y.at<float>(i+corner_row1,j+corner_col1);
          }
         }
        im_to_save = (float*)(output_mat7.data);
        const char* imname7_1;
        imname7_1=imname7.c_str();
        file_to_save.open(imname7_1,ios::out | ios::binary);
        //ifstream file_to_save(imname10,ios::in | ios:: binary);
        file_to_save.write((char*)&row, sizeof(int));
        file_to_save.write((char*)&col, sizeof(int));
        file_to_save.write((char*)&cha, sizeof(int));
        file_to_save.write((char*)& im_to_save, row*col*cha*sizeof(float));
        file_to_save.close(); 


        // the sixth folder
        cha = 1;
         im_to_save = (float*)(gray1.data);
        const char* imname6_1;
        imname6_1=imname6.c_str();
        file_to_save.open(imname6_1,ios::out | ios::binary);
        //ifstream file_to_save(imname10,ios::in | ios:: binary);
        file_to_save.write((char*)&row, sizeof(int));
        file_to_save.write((char*)&col, sizeof(int));
        file_to_save.write((char*)&cha, sizeof(int));
        file_to_save.write((char*)& im_to_save, row*col*cha*sizeof(float));
        file_to_save.close(); 

        // the fifth folder 
        cha = 2;
         Mat output_mat5(row,col,CV_32FC2);
         for(int i = 0;i<row;i++){
          for(int j = 0; j<col;j++){
            output_mat5.at<Vec2f>(i+corner_row1,j+corner_col1)[0] = cflow_x.at<float>(i+corner_row1,j+corner_col1);
            output_mat5.at<Vec2f>(i+corner_row1,j+corner_col1)[1] = cflow_y.at<float>(i+corner_row1,j+corner_col1);
          }
         }
        im_to_save = (float*)(output_mat5.data);
        const char* imname5_1;
        imname5_1=imname5.c_str();
        file_to_save.open(imname5_1,ios::out | ios::binary);
        //ifstream file_to_save(imname10,ios::in | ios:: binary);
        file_to_save.write((char*)&row, sizeof(int));
        file_to_save.write((char*)&col, sizeof(int));
        file_to_save.write((char*)&cha, sizeof(int));
        file_to_save.write((char*)& im_to_save, row*col*cha*sizeof(float));
        file_to_save.close();

        // the fourth folder
        cha=1;
         im_to_save = (float*)(fgMaskGMG.data);
        const char* imname4_1;
        imname4_1=imname4.c_str();
        file_to_save.open(imname4_1,ios::out | ios::binary);
        //ifstream file_to_save(imname10,ios::in | ios:: binary);
        file_to_save.write((char*)&row, sizeof(int));
        file_to_save.write((char*)&col, sizeof(int));
        file_to_save.write((char*)&cha, sizeof(int));
        file_to_save.write((char*)& im_to_save, row*col*cha*sizeof(float));
        file_to_save.close();

*/
        std::swap(prevgray, gray);
        num=num+1;
        cout<<"processing"<<endl;
        }
    }
  }



    int main(int argc, char **argv)
    {
      char* gt_list = argv[2];
      char* fd_list = argv[1];
      int iniframe = 6;
      double thre = 0.97;
      namedWindow("Frame");
      GMG =cv::bgsegm:: createBackgroundSubtractorGMG(iniframe,thre);
      GMG->setDefaultLearningRate(0.6);
      GMG->setSmoothingRadius(0);
      GMG->setBackgroundPrior(0.9);
      processVideo(fd_list,gt_list);
      destroyAllWindows();
      return 0;
    }




