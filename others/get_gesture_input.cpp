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

float im_to_save1[380][1030];
float im_to_save2[2][380][1030];
float im_to_save3[380][1030];
float im_to_save4[3][380][1030];
float im_to_save5[2][380][1030];
float im_to_save6[3][380][1030];
float im_to_save7[4][380][1030];
float im_to_save8[2][380][1030];
float im_to_save9[2][380][1030];

  void processVideo(char* fd_list,char* gt_list) {

  //
  // The GMG part!!
  //
    char filename[256];

// the optical flow part

    Mat frame,cflow,uflow;
    //Mat gt_frame;
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
    cout<<gt<<endl;
    im_path = gt.substr(0,found);
    size_t found2 = found+1;
    found = gt.find_first_of("+",found2);
    gt_path = gt.substr(found2,found-found2);
    found2 = found+1;
    found = gt.find_first_of("\n",found2);
    hdf5_name = gt.substr(found2,found-found2);
    cout<<"the file # is"<<hdf5_name<<endl;



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
      //cout<<fd<<endl;
      //strncpy(middle,fd.c_str(),eol);
      strcpy(middle,fd.c_str());
      //cout<<middle<<endl;
      strcat(fd_to_read1,middle);
      cout<<fd_to_read1<<endl;
      
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

      char filename4[100];
      strcpy(filename4,fd_to_read1);
      char resultfd4[]="/input_result/result_bg_op";
      strcat(filename4,resultfd4);
      mkdir(filename4, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
     
      char filename8[100];
      strcpy(filename8,fd_to_read1);
      char resultfd8[]="/input_result/result_bg_x";
      strcat(filename8,resultfd8);
      mkdir(filename8, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

      char filename9[100];
      strcpy(filename9,fd_to_read1);
      char resultfd9[]="/input_result/result_bg_y";
      strcat(filename9,resultfd9);
      mkdir(filename9, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
     
      char filename10[100];
      strcpy(filename10,fd_to_read1);
      char resultfd10[]="/input_result/RGBimages";
      strcat(filename10,resultfd10);
      mkdir(filename10, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

      char filename11[100];
      strcpy(filename11,fd_to_read1);
      char resultfd11[]="/input_result/inspect";
      strcat(filename11,resultfd11);
      mkdir(filename11, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

      char filename12[100];
      strcpy(filename12,fd_to_read1);
      char resultfd12[]="/input_result/inspect2";
      strcat(filename12,resultfd12);
      mkdir(filename12, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);


     //cout<<fd_to_read1<<endl;
      strcpy(im_list,fd_to_read1);
      char back[] = "/input_result/2/%d.png";
      strcat(im_list,back);
      //cout<< im_list<<endl;


      //char filename12[100];
      //strcpy(filename12,fd_to_read1);
      //char resultfd12[]="/input_result/Masks";
     // strcat(filename12,resultfd12);
      //mkdir(filename12, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

      string filename1_str(filename1);
      string filename2_str(filename2);
      string filename4_str(filename4);
      string filename8_str(filename8);
      string filename9_str(filename9);
      string filename10_str(filename10);
      string filename11_str(filename11);
      string filename12_str(filename12);

      ofstream file_to_save;
      
      int num = 1;
      Mat frame_crop;
      Mat gt_frame_crop;
      double max_bg,min_bg,max_x,min_x,max_y,min_y;
      float max_bg1,max_x1,max_y1;
      while(num< 1000000){
        sprintf(filename,im_list,num);
        cout<< filename<<endl;
        frame = imread(filename,CV_LOAD_IMAGE_COLOR);
        int row = frame.rows;
        int col = frame.cols;
        int row_crop = (row-st_row)*0.5;
        int col_crop = (col-st_col)*0.5;

        if (frame.data==NULL){
          cout<<"no data~!"<<endl;
          break;
        }

//std::string imname41 = filename4_1_str+"/"+hdf5_name+".h5";
//std::string imname41_1 = filename4_1_str+"/"+hdf5_name+"_1"+".h5";
//std::string imname41_2 = filename4_1_str+"/"+hdf5_name+"_2"+".h5";

        std::string imname1 = filename1_str+"/"+hdf5_name+".h5";
        std::string imname2 = filename2_str+"/"+hdf5_name+".h5";
        std::string imname4 = filename4_str+"/"+hdf5_name+".h5";
        std::string imname8 = filename8_str+"/"+hdf5_name+".h5";
        std::string imname9 = filename9_str+"/"+hdf5_name+".h5";
        std::string imname10 = filename10_str+"/"+hdf5_name+".png";
        std::string imname11 = filename11_str+"/"+hdf5_name+".png";
        std::string imname12 = filename12_str+"/"+hdf5_name+".png";

        Rect crop(col_crop,row_crop,st_col,st_row);
        frame_crop = frame(crop);
        imwrite(imname10,frame_crop);

        Mat fgMaskGMG ;//= Mat(row,col,CV_32FC1);
        GMG->apply(frame, fgMaskGMG);
        

        minMaxLoc(fgMaskGMG, &min_bg, &max_bg);
        max_bg1 = (float)max_bg;
     
        bg_show = fgMaskGMG.clone();
        bg_show*=200;
        bg_show.convertTo(bg_show1,CV_8UC1);
        imwrite(imname11,bg_show1);
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
        minMaxLoc(cflow_x, &min_x, &max_x);
        minMaxLoc(cflow_y, &min_y, &max_y);
        max_x1=(float)max_x;
        max_y1 = (float)max_y;


        dx_show = cflow_x.clone();
        dx_show*=100;
        dx_show.convertTo(dx_show1,CV_8UC1);
        imwrite(imname12,dx_show);

        gray1 = gray.clone();
        gray1.convertTo(gray1,CV_32FC1);


        string filename_str(filename);
        if (!(filename_str.compare(im_path))){
          cout<<"here"<<endl;
          Mat gt_frame,gt_frame1;// = Mat(row,col,CV_32FC1);
          gt_frame = imread(gt_path,CV_LOAD_IMAGE_COLOR);
          cvtColor(gt_frame, gt_frame1, COLOR_BGR2GRAY);
          gt_frame_crop = gt_frame1(crop);
          //imwrite(imname11,gt_frame_crop);


          //label_to_save = (float*)(gt_frame1.data);
          //im_to_save = (float*)(fgMaskGMG.data);
          //im_to_save = gt_frame.ptr<float>(0);
         // label_to_save = gt_frame.ptr<float>(0);

         //int row_crop = (rand()%(row_up-row_down))+row_down;
         //int col_crop = (rand()%(col_up-col_down))+col_down;
         //int corner_row2 = (rand()%(row_up-row_down))+row_down;
         //int corner_col2 = (rand()%(col_up-col_down))+col_down;
         //int corner_row3 = (rand()%(row_up-row_down))+row_down;
         //int corner_col3 = (rand()%(col_up-col_down))+col_down;


          float label_to_save1[st_row][st_col];
          //float label_to_save2[st_row][st_col];
          //float label_to_save3[st_row][st_col];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              label_to_save1[i][j]=(float)(gt_frame1.at<uchar>(i+row_crop,j+col_crop));
              //label_to_save2[i][j]=(float)(gt_frame1.at<uchar>(i+corner_row2,j+corner_col2));
              //label_to_save3[i][j]=(float)(gt_frame1.at<uchar>(i+corner_row3,j+corner_col3));
            }
          }
          dimsf_label[0] = 1;
          dimsf_label[1] = 1;
          dimsf_label[2] = st_row;
          dimsf_label[3] = st_col;
          DataSpace dataspace1( rank_label, dimsf_label);


          //cout<< row_crop<<endl;
          //cout<< col_crop<<endl;
          //cout<< corner_row2<<endl;
          //cout<< corner_col2<<endl;
          //cout<< corner_row3<<endl;
          //cout<< corner_col3<<endl;
          
 /*
          //for different test kind
          //bg
          //corp1
         H5File file(imname1 , H5F_ACC_TRUNC);
         DataSet dataset1 = file.createDataSet( LABEL_NAME, datatype,dataspace1); 
          dataset1.write(label_to_save1, PredType::NATIVE_FLOAT);
          //float im_to_save[st_row][st_col];
          
          if (max_bg1<=0){
            cout<<"no valid bg num"<<endl;
            max_bg1 = 1;
          }
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save1[i][j]=fgMaskGMG.at<float>(i+row_crop,j+col_crop)*255/max_bg1-120;
              if (im_to_save1[i][j]<-120 ||im_to_save1[i][j]>135 ){
                cout<<"wrong estimation number"<<endl;
              }
            }
          }          
          dimsf[0] = 1;
          dimsf[1] = 1;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          DataSpace dataspace( rank_data, dimsf);        
          DataSet dataset= file.createDataSet( DATASET_NAME, datatype, dataspace );
          dataset.write(im_to_save1, PredType::NATIVE_FLOAT);

        
           
          //op
          //crop1
          H5File file3(imname2 , H5F_ACC_TRUNC);
          DataSet dataset2 = file3.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          dataset2.write(label_to_save1, PredType::NATIVE_FLOAT);
          //cha=2;
          //float im_to_save[st_row][st_col][cha];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save2[0][i][j]=cflow_x.at<float>(i+row_crop,j+col_crop)*255/max_x1-120;
              im_to_save2[1][i][j]=cflow_y.at<float>(i+row_crop,j+col_crop)*255/max_y1-120;
            }
          } 
          dimsf[0] = 1;
          dimsf[1] = 2;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          DataSpace dataspace_op( rank_data, dimsf);        
          DataSet dataset_op= file3.createDataSet( DATASET_NAME, datatype, dataspace_op );
          dataset_op.write(im_to_save2, PredType::NATIVE_FLOAT);
*/
      //bg_op
          //crop1
          H5File file9(imname4, H5F_ACC_TRUNC);
          DataSet dataset3 = file9.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          
          dataset3.write(label_to_save1, PredType::NATIVE_FLOAT);
          cha=3;
          
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save4[0][i][j]=(fgMaskGMG.at<float>(i+row_crop,j+col_crop))*255/max_bg1-120;
              //cout<<"bg"<<endl;
              im_to_save4[1][i][j]=(cflow_x.at<float>(i+row_crop,j+col_crop))*255/max_x1-120;
              //cout<<"x"<<endl;
              im_to_save4[2][i][j]=(cflow_y.at<float>(i+row_crop,j+col_crop))*255/max_y1-120;
              //cout<<"y"<<endl;
            }
          } 
          dimsf[0] = 1;
          dimsf[1] = 3;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          DataSpace dataspace_bg_op( rank_data, dimsf);      
          DataSet dataset_bg_op= file9.createDataSet( DATASET_NAME, datatype, dataspace_bg_op ); 
          dataset_bg_op.write(im_to_save4, PredType::NATIVE_FLOAT);
          
/*
          //bg_x
          //crop1
          H5File file21(imname8 , H5F_ACC_TRUNC);
          DataSet dataset4 = file21.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          dataset4.write(label_to_save1, PredType::NATIVE_FLOAT);
          //cha=2;
          //float im_to_save[st_row][col][cha];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save8[0][i][j]=fgMaskGMG.at<float>(i+row_crop,j+col_crop)*255/max_bg1-120;
              im_to_save8[1][i][j]=cflow_x.at<float>(i+row_crop,j+col_crop)*255/max_x1-120;
            }
          } 
          dimsf[0] = 1;
          dimsf[1] = 2;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          DataSpace dataspace_bg_x( rank_data, dimsf);        
          DataSet dataset_bg_x= file21.createDataSet( DATASET_NAME, datatype, dataspace_bg_x );
          dataset_bg_x.write(im_to_save8, PredType::NATIVE_FLOAT);





          H5File file22(imname9 , H5F_ACC_TRUNC);
          DataSet dataset5 = file22.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          dataset5.write(label_to_save1, PredType::NATIVE_FLOAT);
          //cha=2;
          //float im_to_save[st_row][col][cha];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save9[0][i][j]=fgMaskGMG.at<float>(i+row_crop,j+col_crop)*255/max_bg1-120;
              im_to_save9[1][i][j]=cflow_y.at<float>(i+row_crop,j+col_crop)*255/max_x1-120;
            }
          } 
          dimsf[0] = 1;
          dimsf[1] = 2;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          DataSpace dataspace_bg_y( rank_data, dimsf);        
          DataSet dataset_bg_y= file22.createDataSet( DATASET_NAME, datatype, dataspace_bg_y );
          dataset_bg_y.write(im_to_save9, PredType::NATIVE_FLOAT);


          
          //bg_x_gray
          //crop1
          H5File file24(imname9 , H5F_ACC_TRUNC);
          //dataset1 = file24.createDataSet( LABEL_NAME, datatype, dataspace1 ); 
          //dataset1.write(label_to_save1, PredType::NATIVE_FLOAT);
          //cha=3;
          //float im_to_save[st_row][st_col][cha];
          for (int i=0;i<st_row;i++){
            for(int j = 0;j<st_col;j++){
              im_to_save9[0][i][j]=fgMaskGMG.at<float>(i+row_crop,j+col_crop)*255/max_bg1-120;
              im_to_save9[1][i][j]=cflow_x.at<float>(i+row_crop,j+col_crop)*255/max_x1-120;
              im_to_save9[2][i][j]=gray1.at<float>(i+row_crop,j+col_crop)-120;
            }
          } 
          dimsf[0] = 1;
          dimsf[1] = 3;
          dimsf[2] = st_row;
          dimsf[3] = st_col;
          DataSpace dataspace_bg_x_g( rank_data, dimsf);        
          DataSet dataset_bg_x_g= file24.createDataSet( DATASET_NAME, datatype, dataspace_bg_x_g );
          dataset_bg_x_g.write(im_to_save9, PredType::NATIVE_FLOAT);

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
      int iniframe = 1;
      double thre = 0.97;
      GMG =cv::bgsegm:: createBackgroundSubtractorGMG(iniframe,thre);
      GMG->setDefaultLearningRate(0.6);
      GMG->setSmoothingRadius(0);
      GMG->setBackgroundPrior(0.9);
      processVideo(fd_list,gt_list);
      return 0;
    }




