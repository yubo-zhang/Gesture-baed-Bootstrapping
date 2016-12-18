/*
 *  HandDetector.cpp
 *  TrainHandModels
 *
 *  Created by Qun Li 07222014
 *
 */
#include "HandDetector.h"

Point seedi;
vector<Point> polygons;
vector<Vec3b> samples;

void MyMouseFunction(int event, int x, int y, int, void*)
{
	if( event == EVENT_LBUTTONDOWN )
	{
      // saves mouse pointer coordinates (x,y) send by button pressed message
        Point seed = Point(x,y);
		seedi = seed;
		polygons.push_back(seed);
		//cout << "Left mouse button is clicked - position (" << x << ", " << y << ")" << endl;
		//cout << "Current polygon points" << polygons << endl;
    }
}

void HandDetector::trainModels(string basename, string model_prefix, int max_models, double scaleResize, double startFrame, double skipFrame, double nFrames, int fillValue)
{
	cout << "HandDetector Training" << endl;
	
	VideoCapture cap(basename);

	double frameIndex = startFrame;
	int frameCount = 0;
	double frameIndexEnd = startFrame + skipFrame*nFrames;

	cap.set(CV_CAP_PROP_POS_FRAMES , startFrame);
	
	for (frameCount; frameCount < nFrames; frameCount++)
	{
		Mat frame;
		Mat frameSmall;
		int lineType = 8;
		polygons.clear();
		cap.read(frame); 
		resize(frame, frameSmall, Size(), scaleResize, scaleResize, INTER_LINEAR);
		cvtColor(frameSmall, _frameGray, CV_RGB2GRAY);
		Size s = _frameGray.size();
        int rows = s.height;
        int cols = s.width;
		cout << "Mask size" << rows<< " and "<< cols<< endl;
		_maskROI = Mat::zeros(rows, cols, CV_8UC3);

		namedWindow( "LocateSeed", 0 );
		imshow("LocateSeed", _frameGray);
		setMouseCallback( "LocateSeed", MyMouseFunction, NULL);
		waitKey(0);
		destroyWindow( "LocateSeed");
		cout << "number of points" << polygons.size()<< endl;

		const Point* ppt[1] = { &polygons[0] };
		int npt[] = {polygons.size()};
		
		fillPoly(_maskROI,ppt, npt, 1, Scalar(255, 255, 0));
		namedWindow( "Selected Mask", 0 );
		imshow("Selected Mask", _maskROI);
		waitKey(0);
		destroyWindow( "Selected Mask");

		frameIndex = frameIndex + skipFrame;
		cap.set(CV_CAP_PROP_POS_FRAMES , frameIndex);	
		for (int i = 0; i < cols; i++ ) 
		{
			for (int j = 0; j < rows; j++) {
				Vec3b tempVec1 = _maskROI.at<Vec3b>(j, i);
				if (tempVec1[0]!=tempVec1[2]) 
				{ 
					Vec3b tempVec = frameSmall.at<Vec3b>(j, i);
					samples.push_back(tempVec);
					//cout << "sample value at "<< j << "," << i << "is: "<<tempVec<<endl;
				}
			}
		} 
	}

    // prepare outputs        
	Mat labels;       
	Mat probs;        
	Mat log_likelihoods;   
	cout << "sample size "<< samples.size() <<endl;
	Mat samplesMat( samples.size(), 3, CV_32FC1 );
	cout << "original sampleMat size "<< samplesMat.size() <<endl;
	for (int i = 0; i< samples.size();i++)
	{
		samplesMat.at<Vec3f>(i, 0) = samples[i];
	}
	//cout << samplesMat <<endl;
	cout << "sampleMat size "<< samplesMat.size() <<endl;
	EM _GMM(max_models);
	// execute EM Algorithm        
    _GMM.train(samplesMat, log_likelihoods, labels, probs);
	//cout<<log_likelihoods<<endl;
	string filename= model_prefix + "test.xml";
	FileStorage fs(filename, FileStorage::WRITE);
	_GMM.write(fs);
	fs.release();
}

void HandDetector::testModels(string basename, string model_prefix, double thresh, double scaleResize, double startFrame, double skipFrame, double nFrames)
{
	cout << "HandDetector Testing" << endl;
	string filename= model_prefix + "test.xml";
	const FileStorage fs(filename, FileStorage::READ);
    if (fs.isOpened()) 
	{
        const FileNode& fn = fs["StatModel.EM"];
         _GMM.read(fn);
    }

	//modification of the virual images
	Mat means = _GMM.getMat("means");
	vector<Mat> covs = _GMM.get<vector<Mat>>("covs");
	int nclusters = _GMM.get<int>("nclusters");
	Mat sigMat3(means.rows, means.cols, means.type());
	double threshSig = 3;
	for (int t=0; t<nclusters;t++)
	{
		sigMat3.at<double>(t,0) = threshSig*sqrt(covs[t].at<double>(0,0));
		sigMat3.at<double>(t,1) = threshSig*sqrt(covs[t].at<double>(1,1));
		sigMat3.at<double>(t,2) = threshSig*sqrt(covs[t].at<double>(2,2));
	}
	Mat threshMatLower = means - sigMat3;
	Mat threshMatHigher = means + sigMat3;	
	
	VideoCapture cap(basename);

	double frameIndex = startFrame;
	int frameCount = 0;
	double frameIndexEnd = startFrame + skipFrame*nFrames;

	cap.set(CV_CAP_PROP_POS_FRAMES , startFrame);
	Mat frame;
	Mat testMat;
	Mat frameSmall;
	cap.read(frame); 
	resize(frame, frameSmall, Size(), scaleResize, scaleResize, INTER_LINEAR);
	vector<Mat> Low;
	vector<Mat> High;
	for (int t=0; t<nclusters; t++)
	{
		Mat LowTemp(frameSmall.rows, frameSmall.cols, frameSmall.type(), Scalar(threshMatLower.at<double>(t,0), threshMatLower.at<double>(t,1), threshMatLower.at<double>(t,2))); 
		Low.push_back(LowTemp);
		Mat HighTemp(frameSmall.rows, frameSmall.cols, frameSmall.type(), Scalar(threshMatHigher.at<double>(t,0), threshMatHigher.at<double>(t,1), threshMatHigher.at<double>(t,2))); 
		High.push_back(HighTemp);
	}

	// filtering parameters
	int element_shape = MORPH_RECT;
	int max_iters = 10;
	int open_close_pos = 0;
	int erode_dilate_pos = 0;
	int morph_elem = 2;
	int morph_size = 2;
	Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );

	cap.set(CV_CAP_PROP_POS_FRAMES , startFrame);
	for (frameCount; frameCount< nFrames; frameCount++)
	{    
	cap.read(frame); 
	resize(frame, frameSmall, Size(), scaleResize, scaleResize, INTER_LINEAR);
	cvtColor(frameSmall, _frameGray, CV_RGB2GRAY);
	Size s = _frameGray.size();

	//real change
	Mat mask(_frameGray.rows,_frameGray.cols, _frameGray.type(), Scalar(0));

	for (int t=0;t<nclusters;t++)
	{
		Mat result1 = (frameSmall>Low[t])&(frameSmall<High[t]);
		Mat channel[3];
	    split(result1, channel);
	    Mat result2 = channel[0]&channel[1]&channel[2];
	    mask = mask|result2;
	}

	_mask = mask;

	Mat dst;
    
	// Apply the specified morphology operation
	//morphologyEx( _mask, dst, CV_MOP_OPEN, element );
	morphologyEx( _mask, dst, CV_MOP_CLOSE, element );
	//namedWindow( "Output Mask after Filtering", 0 );
	//imshow("Output Mask after Filtering", dst);
	//waitKey(0);
	//destroyWindow( "Output Mask after Filtering");
	frameIndex = frameIndex + skipFrame;
	cap.set(CV_CAP_PROP_POS_FRAMES , frameIndex);	
	//please change this output path as well
	//imwrite( format( "C:\\Users\\usx26461\\Desktop\\MUHB\\videos\\License\\Raja\\MaskDetection%d.jpg", frameCount), _mask);
	//imwrite( format( "C:\\Users\\usx26461\\Desktop\\MUHB\\videos\\License\\Raja\\MaskFiltered%d.jpg", frameCount), dst);
	//imwrite( format( "C:\\Users\\usx26461\\Desktop\\MUHB\\videos\\License\\Raja\\Frame%d.jpg", frameCount), frameSmall);
	}
	cout << "Finish testing" <<endl;
}





