#include <iostream>
#include <iomanip>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "grabcut.hpp"

using namespace std;
using namespace cv;

int main (int argc, char * const argv[]) {
    
	// assumes a continous video sequence in the form of jpegs
	// assumes a certain directory path from root
	
	string root = "/home/yuboz/data/input_data/result/";
	string basename = "glov";
	string img_prefix = root + "img/" + basename + "/";
	string msk_prefix = root + "mask/"+ basename + "/";
	
	stringstream ss;
	
	ss.str("");
	ss << "mkdir -p " + root + "/mask/" + basename;
	system(ss.str().c_str());
	
	GrabCut gb;
	
	vector<int> q(2,100);
	q[0]=1;
	
	int f = 46;		// starting frame of the video
	while(f<48)
	{
		f+=1;			// use every 100 frames
		
		ss.str("");
		ss << img_prefix << f << ".png";
		cout <<"Opening: " << ss.str() << endl;
		
		Mat img = imread(ss.str());
//Mat cut(img,Rect(0,0,412,380));
		if(!img.data) continue; // break
		
		Mat mask;
		gb.run(img,mask);
		
		ss.str("");
		ss << msk_prefix <<  f << ".png";
		imwrite(ss.str(),mask,q);
		
		
		
	}
	
	
	
	
    return 0;
}
