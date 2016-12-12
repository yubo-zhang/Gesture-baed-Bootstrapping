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
#include "opencv2/video/background_segm.hpp"
#include "./bgsegm.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/video.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/video/tracking.hpp"
#include <sys/stat.h>
#include "H5Cpp.h"
#include <cstddef> 

#ifndef H5_NO_NAMESPACE
using namespace H5;
#endif

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


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


// generating optical flow and background Bayesian estimation
void processVideo(char* gt_list);
const H5std_string  DATASET_NAME( "data" );
const H5std_string  LABEL_NAME( "label" );
Ptr<cv::bgsegm::BackgroundSubtractorGMG> GMG;
Ptr<cv::DenseOpticalFlow> DF;

// calculate input for optical flow
Mat drawOptFlowMap_x(const Mat& flow){
  int step = 1;
  float rad_x =0;
  Mat cflow_x;
  cflow_x = Mat::zeros(flow.rows,flow.cols,CV_32FC(1));   
  for(int y = 0; y < flow.rows; y += step){
     for(int x = 0; x < flow.cols; x += step){
         const Point2f& fxy = flow.at<Point2f>(y, x);
         rad_x = sqrt(fxy.x*fxy.x);
         cflow_x.at<float>(Point(x,y)) = (float)(rad_x);
         }
      }
   return cflow_x;
   }

Mat drawOptFlowMap_y(const Mat& flow){
  int step = 1;
  float rad_y = 0;
  Mat cflow_y;
  cflow_y = Mat::zeros(flow.rows,flow.cols,CV_32FC(1));
  for(int y = 0; y < flow.rows; y += step){
      for(int x = 0; x < flow.cols; x += step){
          const Point2f& fxy = flow.at<Point2f>(y, x);
          rad_y = sqrt(fxy.y*fxy.y);
          cflow_y.at<float>(Point(x,y))=(float)(rad_y);
      }
   }
   return cflow_y;
   }
//end

float im_to_save4[3][380][1030];

void processVideo(char* gt_list) {
    char filename[256];
    Mat frame,cflow,uflow;
    Mat gray,gray1;
    DF= cv::createOptFlow_DualTVL1();
    Mat gray_input;
    Mat prevgray;
    Mat bg_show, dx_show,dy_show,gray_show;
    Mat bg_show1, dx_show1,dy_show1,gray_show1;
    ifstream gt_folder_txt;
    ifstream im_txt;
    gt_folder_txt.open(gt_list,ios::in);
    string gt;
    string im_path;
    string gt_path;
    string hdf5_name;

    //read the files to get image information
    getline(gt_folder_txt,gt);
    size_t found = gt.find_first_of("+");
    cout<<gt<<endl;
    im_path = gt.substr(0,found);
    size_t found2 = found+1;
    found = gt.find_first_of("\n",found2);
    hdf5_name = gt.substr(found2,found-found2);
    cout<<"the file # is"<<hdf5_name<<endl;


    //prepare for writing into h5 files
    int cha;//channel number
    int rank_data;
    int rank_label;
    FloatType datatype(PredType::IEEE_F32BE);
    hsize_t  dimsf[4];    
    hsize_t dimsf_label[4];
    rank_data = 4;
    rank_label = 4;
    //standard output size
    int st_row = 380;
    int st_col = 1030;


      char fd_to_read1[] = "../results";
      char im_list[200];

      char filename4[100];
      strcpy(filename4,fd_to_read1);
      char resultfd4[]="/gesture/result_bg_op";
      strcat(filename4,resultfd4);
      mkdir(filename4, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
          
      char filename10[100];
      strcpy(filename10,fd_to_read1);
      char resultfd10[]="/gesture/RGBimages";
      strcat(filename10,resultfd10);
      mkdir(filename10, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

      char filename11[100];
      strcpy(filename11,fd_to_read1);
      char resultfd11[]="/gesture/inspect";
      strcat(filename11,resultfd11);
      mkdir(filename11, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

      char filename12[100];
      strcpy(filename12,fd_to_read1);
      char resultfd12[]="/gesture/inspect2";
      strcat(filename12,resultfd12);
      mkdir(filename12, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);


     //cout<<fd_to_read1<<endl;
      strcpy(im_list,fd_to_read1);
      char back[] = "/gesture/crop/%d.png";
      strcat(im_list,back);

      string filename4_str(filename4);
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

        std::string imname4 = filename4_str+"/"+hdf5_name+".h5";
        std::string imname10 = filename10_str+"/"+hdf5_name+".png";
        std::string imname11 = filename11_str+"/"+hdf5_name+".png";
        std::string imname12 = filename12_str+"/"+hdf5_name+".png";
        //save RGBimages
        Rect crop(col_crop,row_crop,st_col,st_row);
        frame_crop = frame(crop);
        imwrite(imname10,frame_crop);
        //Background subtraction part
        Mat fgMaskGMG ;
        GMG->apply(frame, fgMaskGMG);
        

        minMaxLoc(fgMaskGMG, &min_bg, &max_bg);
        max_bg1 = (float)max_bg;
     
        bg_show = fgMaskGMG.clone();
        bg_show*=200;
        bg_show.convertTo(bg_show1,CV_8UC1);
        imwrite(imname11,bg_show1);

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        // the optical flow part
        if( prevgray.empty()||gray.rows!=prevgray.rows||gray.cols!=prevgray.cols){
          gray.copyTo(prevgray);
        }

         DF->calc(prevgray,gray,uflow);
         Mat cflow_x;
         Mat cflow_y;
         cflow_x = drawOptFlowMap_x(uflow);
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
            //generate hdf5 files with 3 dimension: op-x, op-y, bg
            H5File file9(imname4, H5F_ACC_TRUNC);
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
          
          getline(gt_folder_txt,gt);
          size_t found = gt.find_first_of("+");
          cout<<gt<<endl;
          im_path = gt.substr(0,found);
          size_t found2 = found+1;
          found = gt.find_first_of("\n",found2);
          hdf5_name = gt.substr(found2,found-found2);
        }

        std::swap(prevgray, gray);
        num=num+1;
        cout<<"processing"<<endl;
    }
  }



int main(int argc, char **argv)
{
    int final_bd_up=0; 
    int final_bd_left=0;
    Mat frame,cflow,flow,uflow;

    //calibration parameter
    Mat cameraMatrix;
    Mat distCoeffs;
    Mat map1,map2;
    Size imageSize;
    double aa[3][3]={{7.1874833893829771e+02, 0., 6.3950000000000000e+02},{0,7.1874833893829771e+02,3.5950000000000000e+02},{0,0,1}};
    cameraMatrix = Mat(3, 3, CV_64F,aa);
    double bb[5][1]= {{-2.484045e-01},{1.0144540877e-01},{0},{0},{-2.614560230472e-02}};
    distCoeffs = Mat(5, 1, CV_64F,bb);

    Mat cur, cur_grey,cur_new;
    Mat prev, prev_grey;
    Mat first,first_grey,first_new;
    char folder[] = "../results/raw_images";
    std::string txtfile(folder);
    char txtfile1[200];
    char index_file[] = "/index.txt"; //the index of raw images
    cout<<"here"<<endl;
    strcpy(txtfile1,folder);
    //cout<<"here+1"<<endl;
    strcat(txtfile1,index_file);
    //cout<<txtfile1<<endl;
    char result_fd1[] = "../results/gesture/stabalize";
    char newfile1_to_read[200];
    strcat(newfile1_to_read,result_fd1);
    mkdir(result_fd1,S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    char result_fd2[] = "../results/gesture/crop";
    mkdir(result_fd2,S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    
    //read raw images for stabalization
    ifstream file1;
    file1.open(txtfile1,ios::in);
    std::string str,im_name;
    std::getline(file1, str);
    im_name =txtfile+'/'+str.substr(0,7); // the image should be *.png
    first = imread(im_name, CV_LOAD_IMAGE_COLOR);   // Read the file
    if(! first.data)                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
     
   int k=1; //index of output images
   //calibration
   imageSize = first.size();
        initUndistortRectifyMap(
                cameraMatrix, distCoeffs, Mat(),
                getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0), imageSize,
                CV_16SC2, map1, map2);
        remap(first, first_new, map1, map2, INTER_LINEAR);
   //crop no straight part
   int rows = imageSize.height;
   int cols = imageSize.width;
   int midrow = rows/2;
   int midcol = cols/2;
   int up,down,left,right;
   for(int i=0;i<rows;i++){
       if (first_new.at<Vec3b>(i,midcol)[0]!=0||first_new.at<Vec3b>(i,midcol)[1]!=0||first_new.at<Vec3b>(i,midcol)[2]!=0){
         up = i;
         break;
       }
   }
   for(int i=rows-1;i>=0;i--){
       if (first_new.at<Vec3b>(i,midcol)[0]!=0||first_new.at<Vec3b>(i,midcol)[1]!=0||first_new.at<Vec3b>(i,midcol)[2]!=0){
         down = i;
         break;
       }
   }
   for(int i=0;i<cols;i++){
       if (first_new.at<Vec3b>(midrow,i)[0]!=0||first_new.at<Vec3b>(midrow,i)[1]!=0||first_new.at<Vec3b>(midrow,i)[2]!=0){
         left = i;
         break;
       }
   }
   for(int i=cols-1;i>=0;i--){
       if (first_new.at<Vec3b>(midrow,i)[0]!=0||first_new.at<Vec3b>(midrow,i)[1]!=0||first_new.at<Vec3b>(midrow,i)[2]!=0){
         right = i;
         break;
       }
   }
   Rect crop(left,up,right-left+1,down-up+1);
   first_new = first_new(crop);
   Size calibrated_size = first_new.size();
   int cali_row = calibrated_size.height;
   int cali_col = calibrated_size.width;
   int final_bd_right = cali_col-1;
   int final_bd_down = cali_row-1;

   std::string imname_root(result_fd1);
   std::string imname = imname_root+'/'+patch::to_string(k)+".png";
   imwrite(imname,first_new);

   //crop parameter
   vector<Point2f> ori_points;
   ori_points.push_back(Point2f(0,0));
   ori_points.push_back(Point2f(cali_col-1,0));
   ori_points.push_back(Point2f(0,cali_row-1));
   ori_points.push_back(Point2f(cali_col-1,cali_row-1));
   vector<Point2f> trans_points;


    cvtColor(first_new, first_grey, COLOR_BGR2GRAY);
    Mat last_T(3,3,CV_32F);

    while(std::getline(file1, str)) {
      
      k++;

      std::size_t pos = str.find("\\");
      im_name =txtfile+'/'+str.substr(0,pos);
      cout<<im_name<<endl;
      cur = imread(im_name, CV_LOAD_IMAGE_COLOR);
      while(cur.data == NULL) {
            break;
            
        }
      //calibration
        remap(cur, cur_new, map1, map2, INTER_LINEAR);
        Rect crop(left,up,right-left+1,down-up+1);
        cur_new = cur_new(crop);
        //end

        cvtColor(cur_new, cur_grey, COLOR_BGR2GRAY);

        // perspective transformation
        Mat T = Mat::eye(3,3,CV_32F);
        const int warp_mode = MOTION_HOMOGRAPHY;
        // Specify the maximum number of iterations.
         int number_of_iterations = 4000;
        // Specify the threshold of the increment in the correlation coefficient between two iterations
        double termination_eps = 1e-10;
        TermCriteria criteria (TermCriteria::COUNT+TermCriteria::EPS, number_of_iterations, termination_eps);
        findTransformECC(cur_grey,first_grey,T, warp_mode,criteria);
        std::cout<<T<<std::endl;
        //be robust to some bad transformation
        if (T.at<float>(0,0)>1.15||T.at<float>(0,0)<0.985||T.at<float>(1,1)>1.15||T.at<float>(1,1)<0.985){
        T = Mat::eye(3,3,CV_32F);
        std::cout<<"bad estimation"<<std::endl;
        }
        // in rare cases no transform is found. We'll use the last known good transform.
        if(T.data == NULL) {
            last_T.copyTo(T);
        }
        Mat transformed_im;
        T.copyTo(last_T);
        warpPerspective(cur_new,transformed_im,T,cur_new.size());     
        std::string imname = imname_root+'/'+patch::to_string(k)+".png";
        cout<<imname<<endl;
        imwrite(imname,transformed_im);
        cv::perspectiveTransform(ori_points,trans_points,T);

        double left_cand = std::max(trans_points[0].x,trans_points[2].x);
        left_cand = round(left_cand);
        double up_cand = std::max(trans_points[0].y,trans_points[1].y);
        up_cand = round(up_cand);
        double right_cand = std::min(trans_points[1].x,trans_points[3].x);
        right_cand = round(right_cand);
        double bottom_cand = std::min(trans_points[2].y,trans_points[3].y);
        bottom_cand = round(bottom_cand);

        if (left_cand>final_bd_left){final_bd_left=left_cand;}
        if (up_cand>final_bd_up){final_bd_up=up_cand;}
        if (right_cand<final_bd_right){final_bd_right=right_cand;}
        if (bottom_cand<final_bd_down){final_bd_down=bottom_cand;}
        //file<<patch::to_string(left_cand)<<","<<patch::to_string(up_cand)<<","<<patch::to_string(right_cand)<<","<<patch::to_string(bottom_cand);
        std::cout<<"final_bd_left:"<<final_bd_left<<" "<<"final_bd_up:"<<final_bd_up<<" "<<"final_bd_right:"<<final_bd_right<<" "<<"final_bd_down:"<<final_bd_down<<std::endl;
        std::cout<<"one iter"<<std::endl;
        //end
        std::swap(prev_grey, cur_grey);
    }

 //Rect final_crop(5,5,1081-5,508-5);
Rect final_crop(final_bd_left,final_bd_up,final_bd_right-final_bd_left,final_bd_down-final_bd_up);
// this part is for read transformed images and crop them into same size
cout<< k<<endl;
   char filename[256];
   int num = 1;
   Mat readin;
   char number_im[] = "/%d.png";
   const char* im_to_read = strcat(newfile1_to_read,number_im);
   std::string imname_root2(result_fd2);
   cout<<imname_root2<<endl;
   while(num<k){
   std::sprintf(filename,im_to_read,num);
   readin = imread(filename,CV_LOAD_IMAGE_COLOR);
   readin = readin(final_crop);
   
   std::string imname = imname_root2+'/'+patch::to_string(num)+".png";
   imwrite(imname,readin);
   num++;
   }


      char* fd_list = argv[1];
      int iniframe = 1;
      double thre = 0.97;
      GMG =cv::bgsegm:: createBackgroundSubtractorGMG(iniframe,thre);
      GMG->setDefaultLearningRate(0.6);
      GMG->setSmoothingRadius(0);
      GMG->setBackgroundPrior(0.9);
      processVideo(fd_list);
      return 0;

}

