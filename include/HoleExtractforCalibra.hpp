#pragma once
/*
这里是为了标定特别设计的圆孔提取

*/

#include "GouTypes.h"

#include <iostream>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
//using namespace std;
#include <vector>
using std::vector;

#include <string>
using std::string;

#include <opencv2/opencv.hpp>
using namespace cv;
// using cv::Point;
#include <sstream>

class HoleExtract
{
public:
	HoleExtract(void);
	HoleExtract(std::string strimgpath);
	HoleExtract(std::string strimgpath, int holewidth);
	HoleExtract(std::string strimgpath, int holewidth, std::string outdic);
	HoleExtract(std::string strimgpath, int holewidth, CANNYPARAM cannyFactor,std::string outdic);
	HoleExtract(std::string strimgpath, int holewidth, CANNYPARAM cannyFactor, double m_calholenum,std::string outdic);	//m_calholenum 表示孔直径
	HoleExtract(cv::Mat _srcImage, int holewidth, CANNYPARAM cannyFactor, double m_calholenum,std::string outdic);		//直接传入图像

	// CANNYPARAM	m_cannnFactor;

	~HoleExtract(void);
public:
	bool	doEdgeExtract(void);										// 标定板圆孔提取并编码
	bool    doEdgeExtract(cv::Mat _srcImage, double cannylow, double canntscale);
	void	doMorphologyEx(const Mat& Input, Mat& output, int type);	// 形态学操作	

public:
	Mat							m_OriImg;					//该类处理原始图像
	Mat							m_GrayImg;					//灰度图像
	Mat							m_dstCoded;					//最终的编码的图像
	bool						iscoded;					//是否编码成功

	int							HisList[256];				//图像直方图
	vector<RotatedRect>			preboxVector;				//预提取椭圆信息
	vector<vector<cv::Point>>	FinalContours;				//最终筛选出来的用于计算亚像素点的轮廓信息
	vector<RotatedRect>			boxVector;					//亚像素计算前对应椭圆的拟合椭圆信息
	RotatedRect					box;						//保存每次拟合的椭圆

	vector<MatchPoint>		CodedPI;					//存放孔物理坐标与图像坐标对应
	vector<MatchPoint>		CodedoheHolePI;				
	int Serial_number = 0;

	vector<cv::Point2f>     codeOnehole;

	double					averintensity;				//图像平均灰度值
	int						mmthre;						//二值化的阀值：最大间隔
	int						iterthre;					//二值化的阀值：迭代
	int						bfthre;						//二值化的阀值：背景前景

	double					m_cannyFactorLow;
	double					m_cannyFactorHigh;
	int                     m_calHolenum;
	int						m_height;					//图像的高
	int						m_width;					//图像的宽
	string					bmppath;					//图像文件路径
	string					oppath;						//输出路径
	string					m_imgFileName;
	int 					oneholecalsign;
	double					holearea;					//孔的大致面积   =img.width*img.height*0.479/holenum

    //////////////////////////亚像素相关./////////////////////////////
	vector<RotatedRect>		sp_boxVector;				//亚像素拟合后的椭圆信息
	double					M0;
	float(*LUT_Adc)[M0_NUMBER_TWO][2];			//亚像素查找表   look-up table A and d ->c
	float					ww[81];
	float(*LUT_TM)[2];                         //亚像素查找表，tm方法

private:
	int		GetHistogram(const Mat& _input);												// 获取直方图信息
	int		doThreshold(const Mat& input, Mat& output, int type = 0);									// 尝试各种二值化阀值选取方法	
	void	FindHole(const cv::Mat& _input);												// 预定位寻找孔大概位置
	bool	SelectContour(vector<Point> &points, vector<Point> &apppoints);		// 根据圆度筛选轮廓
	bool	doFitEllipse(vector<Point> &points);								// 拟合椭圆	
	bool	UnionSimilarRect(vector<RotatedRect>& allRects);					// 合并相似圆
	bool	prelocisvalid(vector<RotatedRect> &prebVec);						// 判断初定位得到的孔是否能够完成编码和求解单应	
	
	
	void	AccuLocation_v1(const Mat& Input);											// 精确定位 version1.0
	bool	acculocisvalid(vector<RotatedRect> &prebVec);						// 判断最终得到的孔是否能够完成编码和求解单应	
	int		EncodePI(vector<RotatedRect> &bVinput);								// 建立图像坐标与物理坐标联系	
	int     EncodePInew_dankong(vector<RotatedRect> &bVinput);
	int     EncodePInew(vector<RotatedRect> &bVinput);


	static bool SortPreVectByArea(RotatedRect& _one, RotatedRect& _two);

	double	ComputeError(RotatedRect ibox, vector<Point> &points);				// 计算实际轮廓到拟合椭圆的误差
	double	ComputeError(RotatedRect ibox, vector<Point2f> &points);				// 计算实际轮廓到拟合椭圆的误差
	double	ComputeAError(Point point, const RotatedRect &ibox);				// 计算几个点到椭圆距离
	
	

	void	Init_LUT();        //初始化求亚像素的东西
	void	GetSubPixel_TM(const vector<Point2i>&pointArray, vector<Point2f>&NewpointArray, RotatedRect &box);		//Tabatabai and Mitchell 方法
	/*
	void	GetSubPixel_ZH(const vector<Point2i>&pointArray, vector<Point2f>&NewpointArray,RotatedRect &box);		//张辉博士的方法，将上面的TM的作为初值
	*/
	double	MomentGetArea(Mat srcImage, const int x, const int y);													//亚像素点极坐标距离
	void	MomentGetGravity(const Mat srcImage, const int x, const int y, double &dx, double &dy);						//亚像素点极坐标方向

};

