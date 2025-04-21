#include "HoleExtractforCalibra.hpp"
#include <algorithm>
#include <string>
//using namespace std;
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern "C"
{
	#include "CLAPACK/f2c.h"
	#include "CLAPACK/clapack.h"
};


#define Rofbox(b)  (b.size.height+b.size.width)/2.0


HoleExtract::HoleExtract(void)
	:LUT_TM(NULL)
	,LUT_Adc(NULL)
{
	bmppath = "";
	oppath = "";
	averintensity = 0;
	iscoded = false;

}

HoleExtract::HoleExtract(std::string strimgpath)
	: LUT_TM(NULL)
	, LUT_Adc(NULL)
{
	m_OriImg = imread(strimgpath);
	m_dstCoded = m_OriImg.clone();

	//m_GrayImg = m_OriImg.clone();
	cvtColor(m_OriImg, m_GrayImg, CV_BGR2GRAY);   //灰度化图像 注意是 BGR2 
	m_height = m_GrayImg.rows;
	m_width = m_GrayImg.cols;
	m_cannyFactorLow = 30;
	m_cannyFactorHigh = 120;

	bmppath = "";
	bmppath = strimgpath;

	auto nfirst = strimgpath.rfind('\\');
	m_imgFileName = strimgpath.substr(nfirst + 1);
		
	oppath = strimgpath.substr(0, nfirst) + "\\result\\r" + m_imgFileName;

	averintensity = 0;
	iscoded = false;

	holearea = 78400;		//SelectContour 和 prelocation 中使用


}

HoleExtract::HoleExtract(string strimgpath, int holewidth)
	:m_width(0),m_height(0)
	,LUT_TM(NULL)
	,LUT_Adc(NULL)
{
	m_OriImg = imread(strimgpath);
	m_dstCoded = m_OriImg.clone();

	//m_GrayImg = m_OriImg.clone();
	cvtColor(m_OriImg,m_GrayImg,CV_BGR2GRAY);   //灰度化图像 注意是 BGR2 
	m_height=m_GrayImg.rows; 
	m_width=m_GrayImg.cols;
	m_cannyFactorLow = 30;
	m_cannyFactorHigh = 120;

	bmppath="";
	bmppath = strimgpath;

	auto nfirst = strimgpath.rfind('\\');
	m_imgFileName = strimgpath.substr(nfirst + 1);

	oppath = strimgpath.substr(0, nfirst) + "\\r" + m_imgFileName;

	averintensity=0;
	iscoded=false;
	
	holearea = holewidth*holewidth;			//SelectContour 和 prelocation 中使用

}

HoleExtract::HoleExtract(std::string strimgpath, int holewidth, std::string outdic)
	:m_width(0), m_height(0)
	, LUT_TM(NULL)
	, LUT_Adc(NULL)
{
	m_OriImg = imread(strimgpath);
	m_dstCoded = m_OriImg.clone();

	//m_GrayImg = m_OriImg.clone();
	cvtColor(m_OriImg, m_GrayImg, CV_BGR2GRAY);   //灰度化图像 注意是 BGR2  
	m_height = m_GrayImg.rows;
	m_width = m_GrayImg.cols;
	m_cannyFactorLow = 30;
	m_cannyFactorHigh = 120;


	bmppath = "";
	bmppath = strimgpath;

	auto nfirst = strimgpath.rfind('\\');
	m_imgFileName = strimgpath.substr(nfirst + 1);
	oppath = outdic + "\\r" + m_imgFileName;

	averintensity = 0;
	iscoded = false;

	holearea = holewidth*holewidth;			//SelectContour 和 prelocation 中使用

}

HoleExtract::HoleExtract(std::string strimgpath, int holewidth, CANNYPARAM cannyftr,std::string outdic)
	:m_width(0), m_height(0)
	, LUT_TM(NULL)
	, LUT_Adc(NULL)
{
	m_OriImg = imread(strimgpath);
	m_dstCoded = m_OriImg.clone();

	//m_GrayImg = m_OriImg.clone();
	cvtColor(m_OriImg, m_GrayImg, CV_BGR2GRAY);   //灰度化图像 注意是 BGR2 
	m_height = m_GrayImg.rows;
	m_width = m_GrayImg.cols;
	m_cannyFactorLow = cannyftr.lowParam;
	m_cannyFactorHigh = cannyftr.highParam;


	bmppath = "";
	bmppath = strimgpath;

	auto nfirst = strimgpath.rfind('\\');
	m_imgFileName = strimgpath.substr(nfirst + 1);
	oppath = outdic + "\\r" + m_imgFileName;

	averintensity = 0;
	iscoded = false;

	holearea = holewidth*holewidth;			//SelectContour 和 prelocation 中使用

}

HoleExtract::HoleExtract(std::string strimgpath, int holewidth, CANNYPARAM cannyftr, double calholenum, std::string outdic)
	:m_width(0), m_height(0)
	, LUT_TM(NULL)
	, LUT_Adc(NULL)
{
	m_OriImg = imread(strimgpath);
	m_dstCoded = m_OriImg.clone();

	//m_GrayImg = m_OriImg.clone();
	cvtColor(m_OriImg, m_GrayImg, CV_BGR2GRAY);    //灰度化图像 注意是 BGR2 
	m_height = m_GrayImg.rows;
	m_width = m_GrayImg.cols;
	m_cannyFactorLow = cannyftr.lowParam;
	m_cannyFactorHigh = cannyftr.highParam;
	m_calHolenum = calholenum;

	bmppath = "";
	bmppath = strimgpath;

	auto nfirst = strimgpath.rfind('\\');
	m_imgFileName = strimgpath.substr(nfirst + 1);
	oneholecalsign = atoi(m_imgFileName.c_str())-1;

	oppath = outdic + "\\r" + m_imgFileName;

	averintensity = 0;
	iscoded = false;

	holearea = holewidth*holewidth;			//SelectContour 和 prelocation 中使用

}

HoleExtract::HoleExtract(cv::Mat _srcImage, int holewidth, CANNYPARAM cannyftr, double calholenum, std::string outdic)
	:m_width(0), m_height(0)
	, LUT_TM(NULL)
	, LUT_Adc(NULL)
{

	m_OriImg =_srcImage;
	m_dstCoded = m_OriImg.clone();

	//m_GrayImg = m_OriImg.clone();
	m_GrayImg = m_OriImg.clone();

	// cvtColor(m_OriImg, m_GrayImg, CV_BGR2GRAY);   //灰度化图像 注意是 BGR2 
	m_height = m_GrayImg.rows;
	m_width = m_GrayImg.cols;
	m_cannyFactorLow = cannyftr.lowParam;
	m_cannyFactorHigh = cannyftr.highParam;
	m_calHolenum = calholenum;

	// bmppath = "";
	// /home/user/use6_github/JointCalibC2L/data
	// bmppath = strimgpath;

	 auto nfirst = outdic.rfind('\\');
	 m_imgFileName = outdic.substr(nfirst + 1);
	// oneholecalsign = atof(m_imgFileName.c_str());
	// std::stringstream ss(m_imgFileName);
	// ss >> oneholecalsign;

	oppath = "/home/user/use6_github/JointCalibC2L/data/images/results.jpg";

	averintensity = 0;
	iscoded = false;

	holearea = holewidth*holewidth;			//SelectContour 和 prelocation 中使用

}


HoleExtract::~HoleExtract(void)
{
	if (LUT_TM!=NULL)
	{
		delete []LUT_TM;
	}
	if (LUT_Adc!=NULL)
	{
		delete []LUT_Adc;
	}
}



bool HoleExtract::doEdgeExtract(void)
{	
//	std::cout<<bmppath<<std::endl;
//	std::cout<<"Begin the hole extracting"<<std::endl;
	// TRACE(_T("Begin the hole extracting��\n"));
	cv::Mat gaussmat;
	GaussianBlur(m_GrayImg, gaussmat, Size(3, 3), 0, 0);	//高斯滤波
	
	m_GrayImg = gaussmat.clone();

	GetHistogram(gaussmat);									//计算直方图
	
	cv::Mat thredMat;
	doThreshold(gaussmat, thredMat, 1);   //二值化，注意在计算直方图后面
	Mat element = getStructuringElement(MORPH_RECT, Size(11, 11));
	morphologyEx(thredMat, thredMat, MORPH_CLOSE, element);

	FindHole(thredMat);

	Mat cannyMask = Mat::zeros(thredMat.rows, thredMat.cols, CV_8UC1);
	for (int i = 0; i < preboxVector.size(); i++)
	{
		cv::RotatedRect tempMASK(
			preboxVector[i].center, 
			cv::Size2f(1.2*preboxVector[i].size.width, 1.2*preboxVector[i].size.height), 
			preboxVector[i].angle
		);
		cv::RotatedRect tempMASKTWO(
			preboxVector[i].center,
			cv::Size2f(0.8*preboxVector[i].size.width, 0.8*preboxVector[i].size.height),
			preboxVector[i].angle
			);

		ellipse(cannyMask, tempMASK, Scalar(255, 255, 255), CV_FILLED, CV_AA);
		ellipse(cannyMask, tempMASKTWO, Scalar(0, 0, 0), CV_FILLED, CV_AA);

	}

	//Get ROI.
	Mat cnnybtmp;
	gaussmat.copyTo(cnnybtmp, cannyMask);

	Mat ttb;
	Canny(cnnybtmp, ttb, m_cannyFactorLow, m_cannyFactorHigh, 3);

	
	
	// TRACE(_T("PreLocationv1 done\n"));
	std::cout<<"PreLocationv1 done"<<std::endl;

	cv::namedWindow("ttb",WINDOW_NORMAL);
	cv::imwrite("ttb.bmp",ttb);
   // cv::waitKey(0);

	{
		std::sort(preboxVector.begin(), preboxVector.end(), SortPreVectByArea);
		RotatedRect bgbox = preboxVector.front();
		RotatedRect abgbox = preboxVector[1];
		RotatedRect endbox = preboxVector.back();
		char str[100];
		Mat img_test;
		cannyMask.copyTo(img_test);
		for (int i = 0; i<preboxVector.size(); ++i)
		{
			// itoa(temp.num,str,10);
			sprintf(str, "%d", i);
			putText(img_test, str, preboxVector[i].center, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2, CV_AA);
		}
	}

	AccuLocation_v1(ttb);

	if (iscoded)
	{
		for (int i = 0; i < sp_boxVector.size(); i++)
		{
			ellipse(m_dstCoded, sp_boxVector[i], Scalar(255, 255, 255), 3, CV_AA);
		}

		cv::imwrite(oppath, m_dstCoded);
	}
	return iscoded;
}



bool HoleExtract::doEdgeExtract(cv::Mat _srcImage,double cannylow,double canntscale)
{
	std::cout<<"Begin the hole extracting"<<std::endl;

	cv::Mat gaussMat;
	GaussianBlur(_srcImage, gaussMat, Size(3, 3), 0, 0);	///高斯滤波
	
	m_GrayImg = gaussMat.clone();

	GetHistogram(gaussMat);									//计算直方图

	cv::Mat threshedMat;
	doThreshold(gaussMat, threshedMat, 1);   //二值化，注意在计算直方图后面
	Mat element = getStructuringElement(MORPH_RECT, Size(11, 11));
	morphologyEx(threshedMat, threshedMat, MORPH_CLOSE, element);


	FindHole(threshedMat);

	Mat cannyMask = Mat::zeros(threshedMat.rows, threshedMat.cols, CV_8UC1);
	for (int i = 0; i < preboxVector.size(); i++)
	{
		cv::RotatedRect tempMASK(
			preboxVector[i].center,
			cv::Size2f(1.2*preboxVector[i].size.width, 1.2*preboxVector[i].size.height),
			preboxVector[i].angle
			);
		cv::RotatedRect tempMASKTWO(
			preboxVector[i].center,
			cv::Size2f(0.8*preboxVector[i].size.width, 0.8*preboxVector[i].size.height),
			preboxVector[i].angle
			);

		ellipse(cannyMask, tempMASK, Scalar(255, 255, 255), CV_FILLED, CV_AA);
		ellipse(cannyMask, tempMASKTWO, Scalar(0, 0, 0), CV_FILLED, CV_AA);

	}
	Mat cnnybtmp;
	gaussMat.copyTo(cnnybtmp, cannyMask);

	Mat ttb;
	Canny(cnnybtmp, ttb, cannylow, canntscale*cannylow, 3);

	// TRACE(_T("PreLocationv1 done\n"));
	std::cout<<"PreLocationv1 done\n"<<std::endl;

	cv::imshow("ttb",ttb);
    cv::waitKey(0);

	{
		std::sort(preboxVector.begin(), preboxVector.end(), SortPreVectByArea);
		RotatedRect bgbox = preboxVector.front();
		RotatedRect abgbox = preboxVector[1];
		RotatedRect endbox = preboxVector.back();
		char str[100];
		Mat img_test;
		cannyMask.copyTo(img_test);
		for (int i = 0; i<preboxVector.size(); ++i)
		{
			// itoa(temp.num,str,10);
			sprintf(str, "%d", i);
			putText(img_test, str, preboxVector[i].center, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2, CV_AA);
		}
	}


	AccuLocation_v1(ttb);

	if (iscoded)
	{
		for (int i = 0; i < sp_boxVector.size(); i++)
		{
			ellipse(m_dstCoded, sp_boxVector[i], Scalar(255, 255, 255), 3, CV_AA);
		}

		imwrite(oppath, m_dstCoded);
	}
	return iscoded;
}


// 图像二值化
int HoleExtract::doThreshold(const Mat& input, Mat& output,int type)
{
	int width = input.cols;
	int height = input.rows;
	int num = input.rows*input.cols;
	Mat threshtemp;

	int u1, u2, n1, n2;
	int tnext = 0;
	int tpre = 0;

	int averp = 0, averi = 0, ind = -1;
	double pro, inte, gt = 0, tep;

	switch (type)
	{
	case 0:
		//最大间隔方差
		for (int j = 0; j < 256; ++j)
		{
			averp = averi = 0;
			for (int i = 0; i <= j; ++i)
			{
				averp += HisList[i];
				averi += HisList[i] * i;
			}
			pro = (double)(averp) / (double)(num);
			inte = (double)(averi) / (double)(averp);
			double inteu1 = (averintensity - pro*inte) / (1.0 - pro);
			tep = pro*(1 - pro)*(inte - inteu1)*(inte - inteu1);
			if (tep > gt)
				gt = tep, ind = j;
		}
		mmthre = ind;
		threshold(input, output, mmthre, 255, THRESH_BINARY);
		break;
	case 1:
		//取背景色、前景色平均灰度的平均
		tpre = (int)averintensity;
		while (tpre != tnext)
		{
			u1 = u2 = n1 = n2 = 0;
			threshold(input, threshtemp, tpre, 255, THRESH_BINARY);
			for (int i = 0; i < height; ++i)
				for (int j = 0; j < width; ++j)
				{
					int intt = threshtemp.at<uchar>(i, j);
					if (intt > 0)
						u1 += input.at<uchar>(i, j), n1++;
					else
						u2 += input.at<uchar>(i, j), n2++;
				}
			//	tfb=(n2*u1+n1*u2)/(2*n1*n2);
			u1 /= n1; u2 /= n2;
			u1 += u2; u1 /= 2;
			tnext = u1;

			tnext += tpre;
			tpre = tnext - tpre;
			tnext -= tpre;
		}
		bfthre = tpre;
		threshold(input, output, bfthre, 255, THRESH_BINARY);
		break;
	case 2:
		//迭代法
		tpre = (int)averintensity;
		while (tpre != tnext)
		{
			u1 = u2 = n1 = n2 = 0;
			for (int i = 0; i < height; ++i)
				for (int j = 0; j < width; ++j)
				{
					int intt = input.at<uchar>(i, j);
					if (intt > tpre)
						u1 += intt, n1++;
					else
						u2 += intt, n2++;
				}
			u1 /= n1; u2 /= n2;
			tnext = (u1 + u2) / 2;
			tnext += tpre;
			tpre = tnext - tpre;
			tnext -= tpre;
		}
		iterthre = tpre;
		threshold(input, output, iterthre, 255, THRESH_BINARY);
		break;
	default:
		break;
	}


	return 0;
}

// 形态学操作
void HoleExtract::doMorphologyEx(const Mat& Input, Mat& output, int type)
{
	int m_type;
 
	m_type = MORPH_ELLIPSE; 
	int m_size=3;
	Mat element = getStructuringElement( m_type,
										 Size( 2*m_size + 1, 2*m_size+1 ),
										 Point( m_size, m_size ) );

	morphologyEx(Input,output,type,element);

}

int HoleExtract::GetHistogram(const Mat& _input)
{
	
	memset(HisList,0,sizeof(HisList));

	int height = _input.rows;
	int width = _input.cols;
	int num=height*width;
	int step = _input.step;
	int i,j;
	i=j=0;
	for (i=0;i<height;++i)
		for(j=0;j<width;++j)
		{
			HisList[_input.at<uchar>(i, j)]++;
			averintensity += _input.at<uchar>(i, j);
		}
	averintensity/=num;

	return 0;
}

void HoleExtract::FindHole(const cv::Mat& _input)
{
	/// 寻找轮廓
	Mat timg=_input.clone();
	vector<vector<Point> > contours;
	vector<Vec4i>	hierarchy;

	findContours( timg, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE, Point(0, 0) );
	
	double e=0.0;
	int m = 0;
	int i = 0;
	for (i = 0; i< contours.size(); ++i)
	{
		//判断轮廓
		
		vector<Point> apppoints;
		
		if (SelectContour(contours[i],apppoints))
		{
			if(!doFitEllipse(contours[i]))                    //结果存在类成员变量 box 里面
				continue;
			double error=ComputeError(box, contours[i]);      //计算椭圆拟合误差
			e=max(e,error);
			if(error<0.04) 
			{

				m++;
				preboxVector.push_back(box);

			}
		}
		
	}

	UnionSimilarRect(preboxVector);
	return;
}

bool HoleExtract::UnionSimilarRect(vector<RotatedRect>& allRects)
{
	if (allRects.size()<2)
	{
		return false;
	}
	for (int i = 0; i < allRects.size();i++)
	{
		for (int j = i+1; j < allRects.size(); j++)
		{
			Point2f a = allRects[i].center;
			Point2f b = allRects[j].center;
		
			double dis = sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
			double d1 = (allRects[i].size.width + allRects[i].size.height)/2;
			double d2 = (allRects[j].size.width + allRects[j].size.height)/2;

			//double wj = allRects[j].size.width;

			if (dis<2&&(d1/d2<1.1||d1/d2>0.9))
			{
				allRects.erase(allRects.begin() + j);
				j--;
			}
			

		}
	}
	
	return true;
}

bool HoleExtract::SelectContour(vector<Point> &points, vector<Point> &apppoints)
{
	const double pi=3.1415926;
	double length = arcLength(points,true); //轮廓周长
	double area   = contourArea(points);    //轮廓面积
	approxPolyDP(points,apppoints,0.02,true);


	if (area>2*holearea||area<holearea/4)
		return false;

	double formFactor=(length*length)/(4.0*pi*area);//椭圆圆度
	
	bool flag=true;
	flag=flag&&(formFactor>=1.0&&formFactor<1.2);
	return flag;	
}
// 拟合椭圆
bool HoleExtract::doFitEllipse(vector<Point> &points)
{
	double errn=0.0, errp=0.0;
	box = fitEllipse(points);//拟合操作
	errn = ComputeError(box, points);//计算误差

	int nCount = 0;
	while(true)
	{
		const double pi=3.1415926;
		double  sinOfAngle,cosOfAngle;
		double dx,dy,x0,y0;
		double fittingError,square=0;
		size_t count=points.size();
		double erraver=errn*errn*(double)count;

		sinOfAngle=sin(pi*box.angle/180);
		cosOfAngle=cos(pi*box.angle/180);
		x0=box.center.x;
		y0=box.center.y;

		for(size_t i=0;i<count;i++)
		{
			dx=points[i].x-x0;
			dy=points[i].y-y0;
			fittingError=pow(2.0*(dx*cosOfAngle+dy*sinOfAngle)/box.size.width,2.0) 
				+pow(2.0*(dy*cosOfAngle-dx*sinOfAngle)/box.size.height,2.0)-1.0;
			square=abs(fittingError);
			if (square>(errn*2.618))
			{
				points.erase(points.begin()+i);
				count--;
				if (count<15)   //点数过少的话拟合失败
					return false;
				i--;
			}
		}
		errp=errn;
		box =fitEllipse(points);
		nCount++;


		errn=ComputeError(box, points);
		if (abs(errn - errp)<errn*0.01 || errn<0.005 || nCount>50)
			break;
	}
	return true;
}

double HoleExtract::ComputeError(RotatedRect ibox,vector<Point> &points)
{
	const double pi=3.1415926;
	double  sinOfAngle,cosOfAngle;
	double dx,dy,x0,y0;
	double fittingError,square=0;
	size_t count=points.size();
	
	sinOfAngle=sin(pi*ibox.angle/180);
	cosOfAngle=cos(pi*ibox.angle/180);
	x0=ibox.center.x;
	y0=ibox.center.y;

	for(size_t i=0;i<count;i++)
	{
		dx=points[i].x-x0;
		dy=points[i].y-y0;
		fittingError=pow(2.0*(dx*cosOfAngle+dy*sinOfAngle)/ibox.size.width,2.0) 
			+pow(2.0*(dy*cosOfAngle-dx*sinOfAngle)/ibox.size.height,2.0)-1.0;
		square+=abs(fittingError);//pow(fittingError,2.0);
	}	
	return sqrt(square)/(float)count;
}

double HoleExtract::ComputeError(RotatedRect ibox,vector<Point2f> &points)
{
	const double pi=3.1415926;
	double  sinOfAngle,cosOfAngle;
	double dx,dy,x0,y0;
	double fittingError,square=0;
	size_t count=points.size();

	sinOfAngle=sin(pi*ibox.angle/180);
	cosOfAngle=cos(pi*ibox.angle/180);
	x0=ibox.center.x;
	y0=ibox.center.y;

	for(size_t i=0;i<count;i++)
	{
		dx=points[i].x-x0;
		dy=points[i].y-y0;
		fittingError=pow(2.0*(dx*cosOfAngle+dy*sinOfAngle)/ibox.size.width,2.0) 
			+pow(2.0*(dy*cosOfAngle-dx*sinOfAngle)/ibox.size.height,2.0)-1.0;
		square+=abs(fittingError);//pow(fittingError,2.0);
	}	
	return sqrt(square)/(float)count;
}

double HoleExtract::ComputeAError(Point point, const RotatedRect &ibox)
{
	const double pi=3.1415926;
	double  sinOfAngle,cosOfAngle;
	double dx,dy,x0,y0;
	double fittingError,square=0;
//	size_t count=points.size();

	sinOfAngle=sin(pi*ibox.angle/180);
	cosOfAngle=cos(pi*ibox.angle/180);
	x0=ibox.center.x;
	y0=ibox.center.y;

	dx=point.x-x0;
	dy=point.y-y0;
	fittingError=pow(2.0*(dx*cosOfAngle+dy*sinOfAngle)/ibox.size.width,2.0) 
		+pow(2.0*(dy*cosOfAngle-dx*sinOfAngle)/ibox.size.height,2.0)-1.0;
	square=abs(fittingError);//pow(fittingError,2.0);

	return square;
}


// 精确定位 version1.0
void HoleExtract::AccuLocation_v1(const Mat& _Input)
{
	if (!prelocisvalid(preboxVector))
	{
		//std::cout << " 提取的孔不能满足要求 " << std::endl;
		std::cout << "The extracted holes do not fulfill the requirements \n " << std::endl;
		return;
	}

	std::sort(preboxVector.begin(), preboxVector.end(), SortPreVectByArea);

	RotatedRect bgbox = preboxVector.front();
	RotatedRect abgbox = preboxVector[1];
	RotatedRect endbox = preboxVector.back();

	char str[100];
	Mat img_test;
	_Input.copyTo(img_test);
	for (int i = 0; i<preboxVector.size(); ++i)
	{
		sprintf(str, "%d", i);
		putText(img_test, str, preboxVector[i].center, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2, CV_AA);
	}

	double err;
	vector<int> resultcont;
	if(!FinalContours.empty())
		FinalContours.clear();
	// 用prelocation筛选
	Mat img;
	_Input.copyTo(img);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours( img, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE, Point(0, 0) );//此函数会改变原始图像
	for (size_t t = 0; t<preboxVector.size(); ++t)
	{
		vector<Point> choosed;
		RotatedRect alv1box=preboxVector[t];
		for (size_t i = 0; i<contours.size(); ++i)
			for (size_t j = 0; j<contours[i].size(); ++j)
			{
				Point pt = contours[i][j];
				err = ComputeAError(pt, alv1box);  //点到椭圆的距离
				if (err<0.1)                                  //可以设置的高一点，以后还要筛选
				{
					choosed.push_back(pt);
				}
			}
		if (choosed.size()>5)             //拟合椭圆最少点数
		{
			FinalContours.push_back(choosed);
		}
	}
	if (FinalContours.size()<4)        //有可能边缘图像上点数不够
	{
		// TRACE(_T("ellipse is not enough\n"));
		std::cout<<"ellipse is not enough\n"<<std::endl;
		return;
	}
	//最后筛选+拟合
	
	double e=0.0,eg=0.0;
	//FILE *file=fopen(".\\temp\\calib.fiterror", "w");

	Init_LUT();   ////////////亚像素初始化
	sp_boxVector.clear();
	for (size_t i = 0; i<FinalContours.size(); ++i)
	{
		if (!doFitEllipse(FinalContours[i]))                    //结果存在类成员变量 box 里面
			continue;
		double error = ComputeError(box, FinalContours[i]);      //计算椭圆拟合误差
		e=max(e,error);
		if(error<0.02) 
		{
			
			//fprintf(file, "%d i的拟合误差%f   \n", i, error);
			vector<Point2f> NewpointArray;	
			for (size_t it=0;it<FinalContours[i].size();++it)
			{
				Point itt=FinalContours[i][it];
				NewpointArray.push_back(Point2f(itt.x,itt.y));
			}	

			boxVector.push_back(box);

 			GetSubPixel_TM(FinalContours[i],NewpointArray,box);
 			sp_boxVector.push_back(box);
			eg=max(eg,ComputeError(box,NewpointArray));
		}
	}	
	// TRACE(_T("acculocation v1 max error: %f\n"),e);
	// TRACE(_T("subpixel v1 max error: %f\n"),eg);
	std::cout << "acculocation v1 max error: "<< e <<std::endl;
	std::cout << "subpixel v1 max error: "<< eg <<std::endl;

	//fprintf(file, "该幅图像最大椭圆拟合误差%f   \n", e);
	//fclose(file);


	if (acculocisvalid(sp_boxVector))
	{
		if (m_calHolenum == 63)
		{
			std::cout << "63 hole calibrate board \n" << std::endl;
			EncodePI(sp_boxVector);
		}
		if (m_calHolenum == 1)
		{
			std::cout << "63 hole calibrate board \n" << std::endl;
			EncodePInew_dankong(sp_boxVector);
		}
		if (m_calHolenum == 35)
		{
			EncodePInew(sp_boxVector);
		}
	}
	else
		return;

	//TRACE(_T("椭圆编码完成\n"));
	//TRACE(_T("acculocation done!!!\n"));
	std::cout << "coding done\n"<<std::endl;
	std::cout << "acculocation done!!!\n"<<std::endl;


}

// 建立图像坐标与物理坐标联系
float distance(Point2f a,Point2f b)
{
	return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y));
}
float angel(Point2f a,Point2f b)
{
	float m=a.x*b.x+a.y*b.y;
	float n=sqrt(a.x*a.x+a.y*a.y)*sqrt(b.x*b.x+b.y*b.y);
	return m/n;
}

//输入的box按尺寸从大到小排列进行编码，将结果画在m_CannyImg上，最大孔是31 33 最小孔42
int HoleExtract::EncodePI(vector<RotatedRect> &bVinput) 
{
	if(!CodedPI.empty())
		CodedPI.clear();
	RotatedRect t1,t2,t3;
	size_t s=bVinput.size();
	size_t i,j,k;   i=j=k=0;
	float dx,dy,dis,dia;
	MatchPoint temp, temp1;
	Point2f pc,px,py,pp;
	char str[100];

	dx=dy=dis=dia=0.0f;
	t1=bVinput[0];
	t2=bVinput[1];
	t3=bVinput[s-1];
	dia+=t1.size.height+t1.size.width+t2.size.height+t2.size.width; //ֱ直径
	dia/=4.0f;

	pc=t1.center+t2.center;
	pc.x/=2.0f;  
	pc.y/=2.0f;
	
	//最小孔
	temp.num = 42;
	temp.imgx = t3.center.x;	temp.imgy = t3.center.y;
	temp.objy = (42 % 9 - 5) * 40;  temp.objx = (ceil((double)42 / 9) - 4) * 40; temp.objz = 0;
	CodedPI.push_back(temp);
	// itoa(temp.num,str,10);
	sprintf(str, "%d", temp.num);
	putText(m_dstCoded, str, t3.center, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 2, CV_AA);


	//最大孔
	temp.imgx=t1.center.x;  temp.imgy=t1.center.y;
	temp1.imgx=t2.center.x;  temp1.imgy=t2.center.y;
	if(distance(t1.center,t3.center)>distance(t2.center,t3.center))
	{
		py=t2.center-pc;   px=t3.center-t2.center;
		temp.num=31;
		temp.objy=(31%9-5)*40;  temp.objx=(ceil((double)31/9)-4)*40; temp.objz=0;
		temp1.num=33;
		temp1.objy=(33%9-5)*40;  temp1.objx=(ceil((double)33/9)-4)*40; temp1.objz=0;
		CodedPI.push_back(temp);
		CodedPI.push_back(temp1);
	}
	else
	{
		py=t1.center-pc;   
		px=t3.center-t1.center;
		temp.num=33; temp1.num=31;
		temp.objy=(33%9-5)*40;  	temp.objx=(ceil((double)33/9)-4)*40; 	temp.objz=0;//�汾��
		temp1.objy=(31%9-5)*40;  temp1.objx=(ceil((double)31/9)-4)*40;	temp1.objz=0;//�汾��
		CodedPI.push_back(temp);
		CodedPI.push_back(temp1);
	}
	// itoa(temp.num,str,10);
	sprintf(str,"%d",temp.num);
	putText(m_dstCoded, str, t1.center, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 2, CV_AA);
	// itoa(temp1.num,str,10);
	sprintf(str,"%d",temp1.num);
	putText(m_dstCoded, str, t2.center, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 2, CV_AA);
	//其余孔
	for (i=2;i<s-1;++i)
	{
		t1=bVinput[i];
		temp.imgx=t1.center.x;  temp.imgy=t1.center.y;
		dis=distance(pc,t1.center);
		pp=t1.center-pc;
		dx=angel(px,pp)*dis;  dy=angel(py,pp)*dis;
		temp.num = 32 +  9* cvRound(dx / dia) + cvRound(dy / dia);
		if (temp.num%9==0)
			temp.objy=160.0;
		else
			temp.objy=(temp.num%9-5)*40; 
		temp.objx=(ceil((double)temp.num/9)-4)*40; temp.objz=0;
		if (temp.num<0)
		{
			continue;
		}
		CodedPI.push_back(temp);
		
		// itoa(temp.num,str,10);
		sprintf(str,"%d",temp.num);
		putText(m_dstCoded, str, t1.center, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255),2,CV_AA);
	}


	//img title
	putText(m_dstCoded, m_imgFileName, cv::Point2f(100, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 3, CV_AA);

	return 0;
}

//35孔标定板
int HoleExtract::EncodePInew(vector<RotatedRect> &bVinput)
{
	if (!CodedPI.empty())
		CodedPI.clear();
	RotatedRect t1, t2, t3;
	size_t s = bVinput.size();
	size_t i, j, k;   i = j = k = 0;
	float dx, dy, dis, dia;
	MatchPoint temp, temp1;
	Point2f pc, px, py, pp;
	char str[100];

	dx = dy = dis = dia = 0.0f;
	t1 = bVinput[0];
	t2 = bVinput[1];
	t3 = bVinput[s - 1];
	dia += t1.size.height + t1.size.width + t2.size.height + t2.size.width+t3.size.height + t3.size.width; //ֱ��
	dia /= 4.0f;

	pc = t1.center + t2.center;
	pc.x /= 2.0f;
	pc.y /= 2.0f;

	//最大孔
	temp.imgx = t1.center.x;  temp.imgy = t1.center.y;
	temp1.imgx = t2.center.x;  temp1.imgy = t2.center.y;
	if (distance(t1.center, t3.center) > distance(t2.center, t3.center))
	{
		py = t2.center - pc;   px = t3.center - t2.center;
		temp.num = 17;
		temp.objy = (17 % 7 - 4) * 2.5;  temp.objx = (ceil((double)17 / 7) - 3) * 2.5; temp.objz = 0;
		temp1.num = 19;
		temp1.objy = (19 % 7 - 4) * 2.5;  temp1.objx = (ceil((double)19 / 7) - 3) * 2.5; temp1.objz = 0;
		CodedPI.push_back(temp);
		CodedPI.push_back(temp1);
	}
	else
	{
		py = t1.center - pc;
		px = t3.center - t1.center;
		temp.num = 19; temp1.num = 17;
		temp.objy = (19 % 7 - 4) * 2.5;  	temp.objx = (ceil((double)19 / 7) - 3) * 2.5; 	temp.objz = 0;//�汾��
		temp1.objy = (17 % 7 - 4) * 2.5;  temp1.objx = (ceil((double)17 / 7) - 3) * 2.5;	temp1.objz = 0;//�汾��
		CodedPI.push_back(temp);
		CodedPI.push_back(temp1);
	}
	// itoa(temp.num, str, 10);
	sprintf(str,"%d",temp.num);
	putText(m_dstCoded, str, t1.center, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2, CV_AA);
	// itoa(temp1.num, str, 10);
	sprintf(str,"%d",temp1.num);
	putText(m_dstCoded, str, t2.center, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2, CV_AA);
	//其余孔
	for (i = 2; i < s - 1; ++i)
	{
		t1 = bVinput[i];
		temp.imgx = t1.center.x;  temp.imgy = t1.center.y;
		dis = distance(pc, t1.center);
		pp = t1.center - pc;
		dx = angel(px, pp)*dis;  dy = angel(py, pp)*dis;
		temp.num = 18 + 7 * cvRound(dx / dia) + cvRound(dy / dia);
		if (temp.num % 7 == 0)
			temp.objy = 7.50;
		else
			temp.objy = (temp.num % 7 - 4) * 2.50;
		temp.objx = (ceil((double)temp.num / 7) - 3) * 2.50; temp.objz = 0;
		if (temp.num < 0)
		{
			continue;
		}
		CodedPI.push_back(temp);

		// itoa(temp.num, str, 10);
		sprintf(str,"%d",temp.num);

		putText(m_dstCoded, str, t1.center, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2, CV_AA);
	}
	//��С��
	temp.num = 26;
	temp.imgx = t3.center.x;	temp.imgy = t3.center.y;
	temp.objy = (26 % 7 - 4) * 2.5;  temp.objx = (ceil((double)26 / 7) - 3) * 2.5; temp.objz = 0;
	CodedPI.push_back(temp);
	// itoa(temp.num, str, 10);
	sprintf(str,"%d",temp.num);
	putText(m_dstCoded, str, t3.center, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2, CV_AA);

	//img title
	putText(m_dstCoded, m_imgFileName, cv::Point2f(100, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 3, CV_AA);

	return 0;
}

//单孔
int HoleExtract::EncodePInew_dankong(vector<RotatedRect> &bVinput)
{

	if (!CodedPI.empty())
		CodedPI.clear();
	RotatedRect t1;
	size_t s = bVinput.size();
	size_t i, j, k;   i = j = k = 0;

	MatchPoint temp;
	
	char str[100];
	t1 = bVinput[0];
	

//	temp.num = 0;
	temp.imgx = t1.center.x;	temp.imgy = t1.center.y;
	temp.num=oneholecalsign + 1;
	
	

	int signNum = oneholecalsign % 6+1;
	// itoa(signNum, str, 10);
	sprintf(str,"%d",signNum);
	putText(m_dstCoded, str, t1.center, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2, CV_AA);

	//img title
	char img_title[100];
	int imgNum = oneholecalsign / 6 + 1;

	if (imgNum==1)
	{
		temp.objx = 0; temp.objy = 0; temp.objz = 0;
	}
	if (imgNum == 2)
	{
		temp.objx = 1; temp.objy = 1; temp.objz =1;
	}	
	if (imgNum == 3)
	{
		temp.objx = 2; temp.objy = 2; temp.objz = 2;
	}	
	if (imgNum == 4)
	{
		temp.objx = 3; temp.objy = 3; temp.objz = 3;
	}
	if (imgNum == 5)
	{
		temp.objx = 4; temp.objy = 4; temp.objz = 4;
	}
	if (imgNum == 6)
	{
		temp.objx = 5; temp.objy = 5; temp.objz = 5;
	}

	CodedPI.push_back(temp);
	// itoa(imgNum, img_title, 10);
	sprintf(img_title,"%d",imgNum);

	string str1 = str;
	string img_title1 = img_title;

	string txt = img_title1 + "_" + str1;
	putText(m_dstCoded, txt, cv::Point2f(100, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 3, CV_AA);

	return 0;

}

bool HoleExtract::SortPreVectByArea(RotatedRect& _one, RotatedRect& _two)
{
	return _one.size.area() > _two.size.area();
}


// 判断初定位得到的孔是否能够完成编码和求解单应
bool HoleExtract::prelocisvalid(vector<RotatedRect> &prebVec)
{
	if (prebVec.size()<4)
	{
		//std::cout<<"椭圆不够\n"<<std::endl;
		std::cout << "not enough ellipses\n" << std::endl;

		return false;
	}
	//给preboxVector排个序先
	std::sort(prebVec.begin(), prebVec.end(), SortPreVectByArea);

	//之前已经判断preboxVector 的尺寸，不小于4
	RotatedRect bgbox=prebVec.front();
	RotatedRect abgbox=prebVec[1];
	RotatedRect endbox=prebVec.back();
	double abb1=Rofbox(bgbox);                //#define Rofbox(b)  (b.size.height+b.size.width)/2.0
	double abb2=Rofbox(abgbox);
	double abb=(abb1+abb2)/2.0;
	double abst=Rofbox(endbox);
	if (abb1/abb2>1.3||abb1/abb2<0.85)
	{
		//std::cout<<"没有提取到两个大圆\n"<<std::endl;
		std::cout << "no two big one \n" << std::endl;

		return false;
	}
	if (abb/abst>2.1||abb/abst<1.9)
	{
		//std::cout<<"没有提取到两个大圆\n"<<std::endl;
		std::cout << "no two big one \n" << std::endl;

		return false;
	}
	//其他30mm孔
	size_t i = 0, j = 0;
	for (i=2;i<prebVec.size()-1;++i)
	{
		double abtemp=Rofbox(prebVec[i]);
		if (abtemp / abb >0.85 || abtemp/abb<0.65)
		{
			prebVec.erase(prebVec.begin()+i);
			i--;
		}
	}
	if (prebVec.size()<4)
	{
		std::cout<<"not enough ellipses\n"<<std::endl;

		return false;
	}
	return true;
}


// 判断最终得到的孔是否能够完成编码和求解单应
bool HoleExtract::acculocisvalid(vector<RotatedRect> &accubVec)
{
	size_t i=0,t=0;
	double raver=0.0;
	//之前已经判断preboxVector 的尺寸，不小于4
	RotatedRect bgbox=accubVec.front();
	RotatedRect abgbox=accubVec[1];
	RotatedRect endbox=accubVec.back();
	double abb1=Rofbox(bgbox);                //#define Rofbox(b)  (b.size.height+b.size.width)/2.0
	double abb2=Rofbox(abgbox);
	double abb=(abb1+abb2)/2.0;
	double abst=Rofbox(endbox);
	raver+=2*abb*0.75;                           //记得乘以2
	raver+=abst;
	for (i=2;i<accubVec.size()-1;++i)
	{
		raver+=Rofbox(accubVec[i]);
	}
	raver/=(double)accubVec.size();
	
	if (abb1/(raver*2/1.5)>1.3||abb1/(raver*2/1.5)<0.8)
	{
	//	std::cout << "没有提取到两个大圆\n";
		std::cout << "no two big one \n" << std::endl;

		return false;
	}
	if (abb2/(raver*2/1.5)>1.3||abb2/(raver*2/1.5)<0.8)
	{
	//	std::cout << "没有提取到两个大圆\n";
		std::cout << "no two big one \n" << std::endl;

		return false;
	}
	if ((raver/1.5)/abst>1.3||(raver/1.5)/abst<0.7)
	{
		//std::cout << "没有提取到最小圆\n";
		std::cout << "no two snallest one \n" << std::endl;

		return false;
	}
	//其他30mm孔
	for (i=2;i<accubVec.size()-1;++i)
	{
		double abtemp=Rofbox(accubVec[i]);
		if (raver/abtemp>1.1||raver/abtemp<0.9)
		{
			accubVec.erase(accubVec.begin()+i);
			i--;
		}
	}
	if (accubVec.size()<4)
	{
		std::cout << "accul is valid , no enough ellipses" << std::endl;
		return false;
	}
	iscoded=true;
	return true;
}

void HoleExtract::Init_LUT()
{
	{
		double wwt[81]={	
			0.0f,    0.0409f, 0.5182f, 0.8775f, 0.9907f, 0.8775f, 0.5182f, 0.0406f, 0.0f,
			0.0409f, 0.7904f, 1.0f,    1.0f,    1.0f,    1.0f,    1.0f,    0.7904f, 0.0409f,
			0.5182f, 1.0f,    1.0f,    1.0f,    1.0f,    1.0f,    1.0f,    1.0f,    0.5182f,
			0.8775f, 1.0f,    1.0f,    1.0f,    1.0f,    1.0f,    1.0f,    1.0f,    0.8775f,
			0.9907f, 1.0f,    1.0f,    1.0f,    1.0f,    1.0f,    1.0f,    1.0f,    0.9907f,
			0.8775f, 1.0f,    1.0f,    1.0f,    1.0f,    1.0f,    1.0f,    1.0f,    0.8775f,
			0.5182f, 1.0f,    1.0f,    1.0f,    1.0f,    1.0f,    1.0f,    1.0f,    0.5116f,
			0.0409f, 0.7904f, 1.0f,    1.0f,    1.0f,    1.0f,    1.0f,    0.7904f, 0.0409f,
			0.0f,    0.0409f, 0.5182f, 0.8722f, 0.9907f, 0.8722f, 0.5182f, 0.0409f, 0.0f
		};
		for (int i=0;i<81;++i)
			ww[i]=wwt[i];
	}
	{
		M0=0.0;
		int j,k,t=0;
		for(j=-4;j<=4;j++)
			for(k=-4;k<=4;k++)
			{//构造M0
				if(abs(j)+abs(k)>=7)
					continue;
				M0+=ww[(j+4)*9+k+4];
			}
	}
	//	cout<<"*************************************************************************/"<<endl;
	//	cout<<"THIS PROGRAM IS USED FOR STV ONLY WITH AUTHORIZATION FROM ZIMMERMAN!"<<endl;
	//	cout<<"THIS PROGRAM produced the A d c look-up-table\n\tfor MCP method for sub-pixel estimation!"<<endl;
	//	cout<<"THIS PROGRAM is programmed by Zimmerman from NUAA!"<<endl;
	//	cout<<"FOR detail descrption see "<<endl;
	//	cout<<"\t\"Moment and Curvature Preserving Technique\n\tfor Accurate Ellipse Boundar Detection\""<<endl;
	//	cout<<"For contact with programmer email to zimmerman520@163.com"<<endl;
	//	cout<<"\tTHANK YOU!"<<endl;
	//	cout<<"/************************************************************************/"<<endl<<endl;
	//////////////////////////////////////////////////////////////////////////
	//allocate the LUT memory and fill it
	LUT_Adc=NULL;
	LUT_Adc = new float[M0_NUMBER_ONE][M0_NUMBER_TWO][2];
	double d,c,A2,AR,Ar,R,sinalpha,sinbeta,cos2alpha,sin2alpha,sin2beta,alpha,beta,deltac,deltac_from;
	const double r=4.5;
	int i,k;
	assert(LUT_Adc);
	//	string strFileName("AdcLUT.txt");
	//	ofstream SaveFile(strFileName.c_str());
	//	cout<<"Open File for output Name:"<<strFileName.c_str()<<endl;
	//	SaveFile<<strFileName.c_str()<<endl;
	//	SaveFile<<"*************************************************************************/"<<endl;
	//	SaveFile<<"THIS PROGRAM IS USED FOR STV ONLY WITH AUTHORIZATION FROM ZIMMERMAN!"<<endl;
	//	SaveFile<<"THIS PROGRAM produced the A d c look-up-table\n\tfor MCP method for sub-pixel estimation!"<<endl;
	//	SaveFile<<"THIS PROGRAM is programmed by Zimmerman from NUAA!"<<endl;
	//	SaveFile<<"FOR detail descrption see "<<endl;
	//	SaveFile<<"\t\"Moment and Curvature Preserving Technique\n\tfor Accurate Ellipse Boundar Detection\""<<endl;
	//	SaveFile<<"For contact with programmer email to zimmerman520@163.com"<<endl;
	//	SaveFile<<"\tTHANK YOU!"<<endl;
	//	SaveFile<<"/************************************************************************/"<<endl<<endl;
	for(i=0,d=0.5;d<=63.5;d+=0.5,i++)
	{
		//c must be subject to following constraint
		//	firstly
		//		(d+c)<d+r		==>	c<=r
		//	secondly
		//		(d+c)>| |r-d |	==>	c>=fabs(r-d)-d
		// above for being a triangle
		//	thirdly	R=d+c/(d+c)^2+d^2-r^2>=0
		//		(d*d+R*R-r*r)>0	==>	c>=sqrt(r^2-d^2)-d	need to check only when d<r
		//above for law of cosines alpha/2<90d

		//below for adjust beta
		//	when R^2>r^2+d^2	R=d+c
		//		beta/2=pi-beta/2
		//floor
		deltac_from=(r-d>0)?( max( sqrt(r*r-d*d),(r-d) ) )
			:( d-r );
		deltac_from-=d;
		deltac=r-deltac_from;//r-(fabs(r-d)-d);/
		deltac/=M0_NUMBER_TWO-1;
		for(k=0,c=deltac_from;k<=M0_NUMBER_TWO-1;k++)
		{
			c=deltac_from+k*deltac;
			R=d+c;
			cos2alpha=(d*d+R*R-r*r)/(2*d*R+ZERO_DOUBLE);
			sin2alpha=sqrt(1-cos2alpha*cos2alpha);
			alpha=2*acos(cos2alpha);
			sinalpha=sin(alpha);

			sin2beta=R*sin2alpha/r;
			beta=2*asin(sin2beta);
			if(R*R>=(r*r+d*d))
				beta=2*PI-beta;
			sinbeta=sin(beta);

			AR=R*R*(alpha-sinalpha)/2;
			Ar=r*r*(beta-sinbeta)/2;

			A2=AR+Ar;
			assert(A2>=0);
			LUT_Adc[i][k][0]=(float)A2;
			LUT_Adc[i][k][1]=(float)c;
			//			SaveFile<<d<<" "<<A2<<" "<<c<<endl;
		}
	}
	assert(i==M0_NUMBER_ONE);
	//	for(i=0;i<M0_NUMBER_ONE;i++)
	//		for(k=0;k<M0_NUMBER_TWO;k++)
	//		{
	//			TRACE("%d %d %f %f\n",i,k,LUT[i][k][0],LUT[i][k][1]);
	//			assert(LUT[i][k][0]<64);
	//			assert(LUT[i][k][0]>=0);
	//		}
	//	SaveFile.close();
	//////////////////////////////////////////////////////////////////////////
	/////TM lut初始化
//	CString strFileName("AdcLUT2.txt");
	{
		FILE * fp;
		fp = fopen(".\\calib.AdcLUT","w");

		fprintf(fp,"AdcLUT2");
		fprintf(fp,"*************************************************************************/\n");
		fprintf(fp,"THIS PROGRAM IS USED FOR STV ONLY WITH AUTHORIZATION FROM NUAA!\n");
		fprintf(fp,"THIS PROGRAM produced the A d c look-up-table\n\tfor MCP method for sub-pixel estimation!\n");
		fprintf(fp,"THIS PROGRAM is programmed by ... from NUAA!\n");
		fprintf(fp,"FOR detail descrption see \n");
		fprintf(fp, "\t\"\n\tMoment and Curvature Preserving Technique\n\tfor Accurate Ellipse Boundar Detection\n\t\"");
		fprintf(fp,"For contact with programmer email to ...\n");
		fprintf(fp,"\tTHANK YOU!\n");
		fprintf(fp,"/************************************************************************/\n");
		fprintf(fp,"%f %f\n",sin(0.251831),sin(4.067072));
		fprintf(fp,"A2\talpha\tl\n");
		LUT_TM = new float[M0_NUMBER_THREE][2];
		const double r=4.5;
		double l,A2,alpha,deltac,deltac_from,temp;
		int k;
		assert(LUT_TM);
		deltac_from = 0;		
		deltac=2*PI;//r-(fabs(r-d)-d);/
		deltac/=M0_NUMBER_THREE-1;
		for(k=0,alpha=deltac_from;k<=M0_NUMBER_THREE-1;k++)
		{
			alpha = deltac_from + k*deltac;
			temp  = (alpha - sin(alpha))*0.5;
			A2	  = r*r*temp;
			LUT_TM[k][0] = A2;
			LUT_TM[k][1] = alpha;
			l = r* cos(0.5*alpha);
			//SaveFile<<d<<" "<<A2<<" "<<c<<endl;
			fprintf(fp,"%f\t%f\t%f \n",A2, alpha , l);
		}
		fclose(fp);
	}
}
// 亚像素边缘提取
void getgraxy(Mat &input,int x,int y,double g[3][3])
{
	for (int i=y-1;i<=y+1;++i)
		for (int j=x-1;j<=x+1;j++)
		{
			g[i+1-y][j+1-x]=input.at<uchar>(i,j);
		}
	
}

/*
//张博方法
void HoleExtract::GetSubPixel_ZH(const vector<Point2i>&pointArray, vector<Point2f>&NewpointArray,RotatedRect &box)
{
Mat derix,deriy;
//	Scharr(m_GrayImg,derix,CV_32F,1,0);
//	Scharr(m_GrayImg,deriy,CV_32F,0,1);
// 	Sobel(m_GrayImg,derix,CV_32F,1,0,3);
// 	Sobel(m_GrayImg,deriy,CV_32F,0,1,3);
// 	Mat kenelx=Mat::zeros(3,3,CV_32F);
// 	kenelx.at<float>(0,0)=kenelx.at<float>(1,0)=kenelx.at<float>(2,0)=-1;
// 	kenelx.at<float>(0,2)=kenelx.at<float>(1,2)=kenelx.at<float>(2,2)=1;
//
// 	Mat kenely=Mat::zeros(3,3,CV_32F);
// 	kenely.at<float>(0,0)=kenely.at<float>(0,1)=kenely.at<float>(0,2)=1;
// 	kenely.at<float>(2,0)=kenely.at<float>(2,1)=kenely.at<float>(2,2)=-1;

Mat kenelx=Mat::zeros(5,5,CV_32F);
kenelx.at<float>(0,1)=kenelx.at<float>(1,0)=kenelx.at<float>(3,0)=kenelx.at<float>(4,1)=-1;
kenelx.at<float>(1,1)=kenelx.at<float>(2,0)=kenelx.at<float>(3,1)=-3;
kenelx.at<float>(2,1)=-5;
kenelx.at<float>(0,3)=kenelx.at<float>(1,4)=kenelx.at<float>(3,4)=kenelx.at<float>(4,3)=1;
kenelx.at<float>(1,3)=kenelx.at<float>(2,4)=kenelx.at<float>(3,3)=3;
kenelx.at<float>(2,3)=5;
Mat kenely=Mat::zeros(5,5,CV_32F);
kenely=kenelx.t();
flip(kenely,kenely,0);
std::cout<<kenelx<<kenely;
Mat gray_not;
bitwise_not(m_GrayImg,gray_not);
//	imwrite(oppath+"graynot.bmp",gray_not);

filter2D(m_GrayImg,derix,CV_32F,kenelx);
filter2D(m_GrayImg,deriy,CV_32F,kenely);
//	std::cout<<derix<<std::endl;
//	std::cout<<deriy<<std::endl;

size_t count=pointArray.size();
NewpointArray.resize(count);
size_t i;int j,k,l,t;
double x,y,sintheta,costheta,temp;
int tx,ty;
const float r=4.5f;
Point2f ptsNew;

//////////////////////////计算用参数////////////////////////////////
double g[69];
double N[69][3];
double M[69][2];
double a[69];
double A[3][3];
double p[3];
A[0][0]=0; A[0][1]=0; A[0][2]=0;
A[1][0]=0; A[1][1]=0; A[1][2]=0;
A[2][0]=0; A[2][1]=0; A[2][2]=0;
p[0]=1.0f; p[1]=1.0f; p[2]=1.0f;
char jobz='V';
char uplo='U';
integer n=3;
integer lda=3;
doublereal w[3];
w[0]=0.0;	w[1]=0.0;	w[2]=0.0;
integer lwork=300;
doublereal *work=new doublereal [lwork];
integer info=0;
integer liwork=200;
integer *iwork=new integer [liwork];
////////////////////////////////////////////////////////////////////////
for(i=0;i<count;i++)
{//for each edge point adjust the sub-pixel place
x  =  tx = pointArray[i].x;
y  =  ty = pointArray[i].y;
t=0;
for(j=-4;j<=4;j++)
for(k=-4;k<=4;k++)
{//构造矩阵M a
if((abs(j)+abs(k))>=7)   //这些位置不在单位圆内
continue;
int ti=ty+j,tj=tx+k;
double gg[3][3];
getgraxy(m_GrayImg,tj,ti,gg);
sintheta=deriy.at<float>(ty+j, tx+k);//gy
costheta=derix.at<float>(ty+j, tx+k);//gx
g[t]=temp=hypot(sintheta,costheta)+ZERO_DOUBLE;//zero_double;
sintheta/=temp;
costheta/=temp;//sinn,cosn
N[t][0] =  M[t][0] = sintheta;//        求解矩阵N N=[M a]
N[t][1] =  M[t][1] = costheta;
N[t][2] =  a[t]    = (tx+k)*sintheta + (ty+j)*costheta;
N[t][0] *= ww[(j+4)*9+k+4]*g[t];
N[t][1] *= ww[(j+4)*9+k+4]*g[t];	//need debug
N[t][2] *= ww[(j+4)*9+k+4]*g[t];	//multiply weight matirx here
t++;
}
assert(t==69);
for(j=0;j<3;j++)
for(k=0;k<3;k++)
{//A[j][k]
A[j][k]=0;
for(l=0;l<69;l++)
{//i row j col
A[j][k]+=N[l][j]*N[l][k];
}                                  //  求解NTWN
}
/ * Subroutine * /
dsyevd_(&jobz, &uplo, &n, A[0],&lda,w,work,&lwork,iwork, &liwork, &info);
for(l=0;l<3;l++)
p[l]=A[0][l];//
p[0]/=-p[2];
p[1]/=-p[2];
//根据矩阵算法，得到UR，VR坐标
//输出调试，矩阵运算的调试已经完成
double d=hypot(p[1]-y,p[0]-x);	//distance from (u0,v0) to (uR,vR)
if(d<0.5||d>63.5)	//debug	only happend for small ellipses 4.5 3 2.75 2.5!~!
{
OutputDebugString(_T("ERROR in detection memeo!\n"));
break;
}
double theta=atan2(p[1]-y,p[0]-x);	//for less than zero because of direction problem
//////////////////////////////////////////////////////////////////////////
//			  /					^y       /		before situation
//		    /					|	     /
//		  /					    |	   /
//		/					    |   /
//	  /	theta				| /	theta
//	O/________x__>			O/__________x____>
//	|
//	|
//	|
//	|
//	V y    our situation
//
//////////////////////////////////////////////////////////////////////////
//get A2 from moment presevation
double A2=MomentGetArea(m_GrayImg,tx,ty);
//get c from d and A2 from LUT table using bilinear interpolation
int i1 = int( 2*d );	//得到低值
int i2 = i1+1;			//得到高值
int j1=1,j2=1;
assert(i1==i2-1);//floor and ceil
assert(i1>=0);
assert(i2<=M0_NUMBER_ONE);//range
//above first step found ibound up&down
if(A2<LUT_Adc[i1][0][0])
j1=0;
if(A2<LUT_Adc[i2][0][0])
j2=0;
assert(j1+j2);	//not in the A2 range error
assert(A2<63.6173f&&A2>0);//of size
bool b1=true,b2=true;//not found the suitable j1,j2
for(j=0;j<M0_NUMBER_TWO;j++)//remember that A2 is increasing
{//find the range for second step ibound up&down
if(b1)//if found no need to find again
if(A2<=LUT_Adc[i1][j][0])
{//find the first big one got it
j1=j;
b1=false;
if(!b2)
break;//jump for fast
}
if(b2)//if found no need to find again
if(A2<=LUT_Adc[i2][j][0])
{//find the first big one got it
j2=j;
b2=false;
if(!b1)
break;//jump for fast
}
}
//check if not find in the range
assert(j<M0_NUMBER_TWO);
//still now have been found
//	LUT[i][j][1]=f(g(i),LUT[i][j][0])
//the four corner
//
//		A00( d(i1) , A2(j1-1) )	A10( d(i2) , A2(j2-1) )
//		A01( d(i1) , A2(j1) )		A11( d(i2) , A2(j2) )
//
//		A LUT[i1][j1][0]	 A LUT[i1][j1][0]
//		A LUT[i2][j2][0]	 A LUT[i2][j2][0]
//bug when j1=0 or j2=0
//			assert(j1!=0&&j2!=0);
double A00=LUT_Adc[i1][j1-1][0];
double A01=LUT_Adc[i1][j1][0];
double A10=LUT_Adc[i2][j2-1][0];
double A11=LUT_Adc[i2][j2][0];
//fake billinear
//because only d axis ok
//	A2 axis is not ok
//steps first same d , then
sintheta = (A2-A00)/(A01-A00);	//first d half
temp = (1-sintheta)*LUT_Adc[i1][j1-1][1]+(sintheta)*LUT_Adc[i1][j1][1];
sintheta = (A2-A10)/(A11-A10);	//second d half
costheta = (1-sintheta)*LUT_Adc[i2][j2-1][1]+(sintheta)*LUT_Adc[i2][j2][1];//same with temp usage
sintheta =2*d-i1;	//combine two d sample
double c = (1-sintheta)*temp + (sintheta)*costheta;
//继续进行其他公式的计算，还没有完成。。。。。
//uc=u0-c*cos(theta)
//vc=v0-c*sin(theta)
//		TRACE("A2:%f c%f",A2,c);
ptsNew.x=(float)(x-c*cos(theta));
ptsNew.y=(float)(y-c*sin(theta));
//		TRACE(" dx%f dy%f\n",ptsNew.x - x,ptsNew.y - y);
if (fabs(ptsNew.x - x) > 2 || fabs(ptsNew.y - y) > 2 )
continue;	//assert(0);              //这个地方会出问题
else
NewpointArray[i]=(ptsNew);
}
box=fitEllipse(NewpointArray);
double error=ComputeError(box,NewpointArray);
error+=0;
}
*/

//灰度矩原始方法
void HoleExtract::GetSubPixel_TM(const vector<Point2i>&pointArray, vector<Point2f>&NewpointArray,RotatedRect &box)
{
	box=fitEllipse(pointArray);
	double error1=ComputeError(box,NewpointArray);

	size_t count=pointArray.size();
	NewpointArray.resize(count);
	size_t i;
	int j,k=0,t=0;
	double x,y,sinalpha,cosalpha,temp=0.0,theta,l;
	int tx,ty;
	const float r=4.5f;
	Point2f ptsNew;

	for(i=0;i<count;i++)
	{//for each edge point adjust the sub-pixel place
		x  =  tx = pointArray[i].x;
		y  =  ty = pointArray[i].y;
		if (x+4>=m_GrayImg.cols||x-4<0||y+4>=m_GrayImg.rows||y-4<0)
		{	
			continue;
		}

		double A2=MomentGetArea(m_GrayImg,tx,ty);
				
		int j1=0,j2=0;
		if(A2<LUT_TM[j2][0])
			break;
		for(j=0;j<M0_NUMBER_THREE;j++)//remember that A2 is increasing 
		{
			if(A2<=LUT_TM[j][0])
			{//find the first big one got it
				j2=j;
				break;
			}
		}
		if(j==M0_NUMBER_THREE)
			break;
		j1=j2-1;
		double A0=LUT_TM[j1][0];
		double A1=LUT_TM[j2][0];
		double theta0 = LUT_TM[j1][1];
		double theta1 = LUT_TM[j2][1];
		theta = (theta1-theta0)*(A2-A0)/(A1-A0)+theta0;
		l = r*cos(theta*0.5);	//get l distance
		
		MomentGetGravity(m_GrayImg,tx,ty,x,y);	// where angle
		sinalpha = y/(x*x+y*y);
		cosalpha = x/(x*x+y*y);

		ptsNew.x=(float)(tx+l*cosalpha);
		ptsNew.y=(float)(ty+l*sinalpha);
		//		TRACE(" dx%f dy%f\n",ptsNew.x - x,ptsNew.y - y);
		if (fabs(ptsNew.x - tx) > 2 || 
			fabs(ptsNew.y - ty) > 2 )
			continue;//	ASSERT(0);////////////////�޸�
		else
			NewpointArray[i]=(ptsNew);  
	}
	box=fitEllipse(NewpointArray);
//	doFitEllipse()
	double error=ComputeError(box,NewpointArray);
	error+=0;
}


double HoleExtract::MomentGetArea(Mat srcImage,const int x,const int y)
	//Get A2 Area from parameters srcImage is IPL_DEPTH_8U
{
	double M1=0.0,M2=0.0,M3=0.0,p1,h1,h2,grey;
	int j,k,t=0;
	for(j=-4;j<=4;j++)
		for(k=-4;k<=4;k++)
		{//����M0 M1 M2
			if(abs(j)+abs(k)>=7)
				continue;
			grey=srcImage.at<uchar>(y+j, x+k);//gy
			M1+=ww[(j+4)*9+k+4]*grey;//M1 M2 M3 һ�� ���� ���׾�
			M2+=ww[(j+4)*9+k+4]*grey*grey;
			M3+=ww[(j+4)*9+k+4]*grey*grey*grey;
			t++;
		}
		//////////////////////////////////////////////////////////////////////////
		// use to calculate p1,h1,h2; and debug to check it	
		M1/=M0;	M2/=M0;	M3/=M0;///M0 ��һ��
		const float r=4.5f;
		double a=M1*M1-M2;
		double b=M3-M1*M2;
		double c=M2*M2-M1*M3;
		double del=b*b-4*a*c;
		double A2;
		if(a<0)
		{	a*=-1;b*=-1;c*=-1;}
		if(del>0)  //���p1 p2 h1 h2
		{
			h1=( -b - sqrt(del) )/( 2*a+ZERO_DOUBLE);
			h2=( -b + sqrt(del) )/( 2*a+ZERO_DOUBLE);
			assert(h2>0);
			//		h1=(M2-M1*h2)/(M1-h2);
			p1=(M1-h2)/(h1-h2);
			A2=(1-p1)*M0;//area of circle is not the same with sigma of M0
			//debug for check
			//		double check=p1*h1+(1-p1)*h2-M1;
			//		check=p1*h1*h1+(1-p1)*h2*h2-M2;
			//		check=p1*h1*h1*h1+(1-p1)*h2*h2*h2-M3;
		}
		else
			assert(0);
		return A2;
}

void HoleExtract::MomentGetGravity(const Mat srcImage,const int x,const int y,double &dx,double &dy)
{
	double grey,ddx=0,ddy=0,sigma=0;
	int j,k,t=0;
	for(j=-4;j<=4;j++)
		for(k=-4;k<=4;k++)
		{//构造M0 M1 M2
			if(abs(j)+abs(k)>=7)
				continue;
			grey=srcImage.at<uchar>((y+j),(x+k));//gy

			sigma += ww[(j+4)*9+k+4]*grey;

			ddx   += ww[(j+4)*9+k+4]*grey*k;
			ddy   += ww[(j+4)*9+k+4]*grey*j;

			t++;
		}
		dx  = ddx/sigma;
		dy  = ddy/sigma;
}
