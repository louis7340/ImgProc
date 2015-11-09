#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <highgui.h>
#include "cv.h"

#include <cmath>

using namespace std;
using namespace cv;


//#define e 2.71828182845
//#define pi 3.1415926535

#define e 2.718281828459
#define pi 3.14159265359
#define eps 1e-8

double GammaDefault_min = 0.5;
double GammaDefault_max = 4.0;
int kernel_size = 3;
double alpha = 2.0;
double beta = 0.1;
double C1 = 0.01;
double C2 = 0.01;


double Gaussian(int x, int y, double delta);
void MapToImage(Mat srcImg, Mat &dstImg);


int main(){
	Mat SrcImg = imread(".\\testimg\\test8.jpg", CV_LOAD_IMAGE_COLOR);
	Mat RGBChannels[3];

	//分成三個channels
	split(SrcImg, RGBChannels);

	/*
	* Local Gamma 計算
	*/

	//灰階影像  -> average
	Mat f_grayImg(SrcImg.rows, SrcImg.cols, CV_64FC1);
	for (int i = 0; i < SrcImg.rows; i++)
		for (int j = 0; j < SrcImg.cols; j++)
			f_grayImg.at<double>(i,j) = ( RGBChannels[0].at<uchar>(i, j) + RGBChannels[1].at<uchar>(i, j) + RGBChannels[2].at<uchar>(i, j)) / (3.0*255.0);

	//Gaussian Blur
	Mat f_dog1(SrcImg.rows, SrcImg.cols, CV_64FC1);
	GaussianBlur(f_grayImg, f_dog1, Size(kernel_size, kernel_size), 0.5, 0.5);

	Mat f_dog2(SrcImg.rows, SrcImg.cols, CV_64FC1);
	GaussianBlur(f_grayImg, f_dog2, Size(kernel_size, kernel_size), 5.0, 5.0);


	//計算 r^0
	Mat tmpLocalGamma(SrcImg.rows, SrcImg.cols, CV_64FC1);
	for (int i = 0; i < SrcImg.rows; i++){
		for (int j = 0; j < SrcImg.cols; j++){
			
			double f_v1 = f_dog1.at<double>(i, j);
			double f_v2 = f_dog2.at<double>(i, j);

			double min = f_v1;
			double max = f_v1;

			if(f_v2 < min)
				min = f_v2;

			if(f_v2 > max)
				max = f_v2;

			double gamma_val;

			//exist value equal to 0 (min)
			if( fabs(min-0) <eps){
				gamma_val =  tmpLocalGamma.at<double>(i, j)  = 0.01;
			}
			else if( fabs(max -1) <eps){//exist value equal to 1 (min)
				gamma_val =  tmpLocalGamma.at<double>(i, j)  = 5.0;
			}
			else{
				if( fabs(f_v1-f_v2)<eps)
					gamma_val=tmpLocalGamma.at<double>(i, j)=-1.0 / log(f_v2);
				else
					gamma_val=tmpLocalGamma.at<double>(i, j) = ( log(-log(f_v1)) - log(-log(f_v2)) ) /( (log(f_v1) - log(f_v2) + C1));
			}
			//條件限制
			if (tmpLocalGamma.at<double>(i, j) >5.0){
				tmpLocalGamma.at<double>(i, j) = 5.0;
				//printf("Exceed\n");
			}
			else if (tmpLocalGamma.at<double>(i, j) < 0.01){
				//printf("too low %lf\n",gamma_val);
				tmpLocalGamma.at<double>(i, j) = 0.01;
			}
		}
	}

	//計算 r^0'
	Mat GammaDefault(SrcImg.rows, SrcImg.cols, CV_64FC1);
	for (int i = 0; i < SrcImg.rows; i++){
		for (int j = 0; j < SrcImg.cols; j++){
			//Gamma Default caculation
			double GammaDefault_val =  pow( 2, ((int)(f_dog2.at<double>(i,j)*255.0+0.5)-127.5)/127.5 );
			//map to GammaDefault_min ~ GammaDefault_max
			GammaDefault.at<double>(i,j) = (GammaDefault_val-0.5)/1.5*(GammaDefault_max-GammaDefault_min)+GammaDefault_min;
			GammaDefault_val = GammaDefault.at<double>(i,j);
			tmpLocalGamma.at<double>(i, j) = (tmpLocalGamma.at<double>(i, j)*fabs((double)(f_dog1.at<double>(i, j) - f_dog2.at<double>(i, j))) + C2*GammaDefault_val) / (double)(fabs((double)( f_dog1.at<double>(i, j) - f_dog2.at<double>(i, j))) + C2);
		}
	}

	//計算 r^g
	Mat LocalGamma(SrcImg.rows, SrcImg.cols, CV_64FC1);
	GaussianBlur(tmpLocalGamma, LocalGamma, Size(kernel_size, kernel_size), 5.0, 5.0);

	//Release
	tmpLocalGamma.release();
	f_dog1.release();
	f_dog2.release();

	//local gamma transform
	Mat LocalDstImg(SrcImg.rows, SrcImg.cols, CV_8UC1);
	for (int i = 0; i < SrcImg.rows; i++){
		for (int j = 0; j < SrcImg.cols; j++){
			LocalDstImg.at<uchar>(i, j) = (uchar)(pow(f_grayImg.at<double>(i, j), LocalGamma.at<double>(i, j))*255.0 + 0.5);
		}
	}


	/*
	Channel-wise Gamma
	*/
	Mat RGB_Gamma[3];
	for (int i = 0; i < 3; i++)
		RGB_Gamma[i] = Mat(SrcImg.rows, SrcImg.cols, CV_64FC1);

	
	for (int i = 0; i < SrcImg.rows; i++){
		for (int j = 0; j < SrcImg.cols; j++){
			double Saturation;
			double average = f_grayImg.at<double>(i,j);//(RGBChannels[0].at<uchar>(i, j) + RGBChannels[1].at<uchar>(i, j) + RGBChannels[2].at<uchar>(i, j)) / 3.0;
			int p = 0;

			//find p
			for (int k = 1; k < 3; k++){
				if (fabs(RGBChannels[k].at<uchar>(i, j) - average) > fabs(RGBChannels[p].at<uchar>(i, j) - average)){
					p = k;
				}
			}

			Saturation = alpha * fabs(RGBChannels[p].at<uchar>(i, j) - average) + beta;
			for (int k = 0; k < 3; k++){
				double GammaDefault_val = GammaDefault.at<double>(i,j);

				RGB_Gamma[k].at<double>(i, j) = (LocalGamma.at<double>(i, j)* average + GammaDefault_val*Saturation) / (Saturation + RGBChannels[k].at<uchar>(i, j));
			}

		}
	}

	//do local gamma correction
	Mat DstImg(SrcImg.rows, SrcImg.cols, CV_8UC3);
	for (int i = 0; i < SrcImg.rows; i++){
		for (int j = 0; j < SrcImg.cols; j++){
			for (int k = 0; k < 3;k++)
				DstImg.at<Vec3b>(i, j)[k] = (uchar)(pow(RGBChannels[k].at<uchar>(i, j) / 255.0, RGB_Gamma[k].at<double>(i, j))*255.0 + 0.5);
		}
	}

	
	Mat grayImg(SrcImg.rows,SrcImg.cols,CV_8UC1);
	MapToImage(f_grayImg,grayImg);
	imwrite("grayImg.jpg", grayImg);

	Mat dog1(SrcImg.rows,SrcImg.cols,CV_8UC1);
	MapToImage(f_dog1,dog1);
	imwrite("dog1.jpg", dog1);

	Mat dog2(SrcImg.rows,SrcImg.cols,CV_8UC1);
	MapToImage(f_dog2,dog2);


	imwrite("dog2.jpg", dog2);
	imwrite("LocalDstImg.jpg", LocalDstImg);
	imwrite("Result.jpg", DstImg);

	return 0;
}

//map 0~1 image to 0~255 image 
void MapToImage(Mat srcImg, Mat &dstImg)
{
	for (int i = 0; i < srcImg.rows; i++){
		for (int j = 0; j < srcImg.cols; j++){
			dstImg.at<uchar>(i, j) = (int)(srcImg.at<double>(i, j)*255.0 + 0.5);
		}
	}
	return;
}