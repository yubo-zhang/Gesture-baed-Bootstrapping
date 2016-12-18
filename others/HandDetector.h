/*
 *  HandDetector.h
 *  Train and test HandModels
 *
 *  Created by Qun Li on 7/28/2014.
 *  Copyright 2014 PARC. All rights reserved.
 *
 */
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

class HandDetector
{
public:
		
	void trainModels(string basename, string model_prefix, int max_models, double scaleResize, double startFrame, double skipFrame, double nFrames, int fillValue);
	void testModels(string basename, string model_prefix, double thresh, double scaleResize, double startFrame, double skipFrame, double nFrames);

	// Global variables
	EM _GMM;
	Mat _mask;
	Mat _maskROI;
	Mat _frameGray;
	Mat _frameFloat;
};