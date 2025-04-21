#ifndef _TYPES_DEFINITION_BY_GOU_H_
#define _TYPES_DEFINITION_BY_GOU_H_

#ifndef _SPECIAL_VAR_DEFINTION_
#define _SPECIAL_VAR_DEFINTION_


#define			ZERO_DOUBLE					1.0E-18

#define				calTargetHolenum35				35
#define				calTargetHolenum63				63
#define				calTargetHolenum1				1


#define			PI							(3.14159265358979323846264338327950288419716939937511)

#endif

#ifndef _SUB_PIXEL_VAR_DEF_
#define _SUB_PIXEL_VAR_DEF_

#define			M0_NUMBER_ONE				127
#define			M0_NUMBER_TWO				101			//21
#define			M0_NUMBER_THREE				1000		//21
#endif // !_SUB_PIXEL_VAR_DEF_




#ifndef _CAMERA_PIXEL_SIZE_DEF_
#define _CAMERA_PIXEL_SIZE_DEF_

#define			PIXELSIZE_CAMERA			(3.45*1E-3)				//像素物理尺寸大小   
#endif // !_CAMERA_PIXEL_SIZE_DEF_



#ifndef _IMAGE_2_WORLD_RELATIONSHIP_
#define _IMAGE_2_WORLD_RELATIONSHIP_
//物理圆与图像圆对应关系；Physic Image
class MatchPoint
{
public:
	MatchPoint()
		:num(-1), imgx(0), imgy(0), imgR(0)
		, objx(0), objy(0), objz(0), reprojectDeviation(-1)
	{}
	~MatchPoint(){}

public:
	int		num;		//序号, 如果没有可以忽略
	double	imgx;		//图像x坐标
	double	imgy;		//图像y坐标
	double	imgR;		//图像中孔的半径（像素单位）

	double	objx;		//物理x坐标
	double	objy;		//物理y坐标
	double	objz;		//物理z坐标

	double reprojectDeviation;
};
#endif // !_IMAGE_2_WORLD_RELATIONSHIP_


#ifndef _INTRINSIC_PARAM_DEFINITION_
#define _INTRINSIC_PARAM_DEFINITION_
//内参数矩阵，张博定义
class INTRC_PARAM
{
public:
	INTRC_PARAM()
		: sx(0), f(0), alpha(0), beta(0), gamma(0), Cx(0), Cy(0)
		, k1(0), k2(0), k3(0), p1(0), p2(0), a1(0), a2(0), a3(0)
		, a4(0), a5(0), a6(0), a7(0), a8(0)
	{
	}
	~INTRC_PARAM(){}

public:
	double		sx;					// []        Scale factor to compensate for any error in dpx   
	double		f;

	double		alpha;					// dx/f
	double		beta;					//dy/f
	double		gamma;					//偏斜系数

	double		Cx;					// [pix]     Z axis intercept of camera coordinate system      
	double		Cy;					// [pix]     Z axis intercept of camera coordinate system      

	// for distortion
	double		k1;						//	1/mm^2
	double		k2;						//	1/mm^4
	double		k3;						//	1/mm^6

	double		p1;						//tagental
	double		p2;						//tagental
	//from 4 step back projection 8 parameters
	double		a1;
	double		a2;
	double		a3;
	double		a4;
	double		a5;
	double		a6;
	double		a7;
	double		a8;

};

#endif // !_INTRINSIC_PARAM_DEFINITION_

#ifndef _EXTRINSIC_PARAM_DEFINITION_
#define _EXTRINSIC_PARAM_DEFINITION_
//外参数矩阵 张博定义
class EXTRC_PARAM
{
public:
	double	T[3];						//mm  transform
	double	R[3];						//gamma,beta,alpha. in rad（弧度） not in centigrade
	double	e[4];						//in term of Euler parameters 
	double	r[9];						//row first sequence for the consistence of Tsai & Zhang

	int		index;
	double  mean;
	double 	max;
	double  rmse;
	double  stddev;

public:
	EXTRC_PARAM()
	{
		T[0] = 0; T[1] = 0; T[2] = 0;
		R[0] = 0; R[1] = 0; R[2] = 0;
		e[0] = 0; e[1] = 0; e[2] = 0; e[3] = 0;
		r[0] = 0; r[1] = 0; r[2] = 0;
		r[3] = 0; r[4] = 0; r[5] = 0;
		r[6] = 0; r[7] = 0; r[8] = 0;

		index = -1; mean = 0; max = 0; rmse = 0; stddev = 0;
	}
	~EXTRC_PARAM(){}
};

typedef  EXTRC_PARAM	RIGHT2LEFT;
typedef  EXTRC_PARAM	LEFT2RIGHT;
typedef  EXTRC_PARAM	T4VIEW;

#endif // !_EXTRINSIC_PARAM_DEFINITION_


#ifndef _WORLD_POINT_DEFINITION_
#define _WORLD_POINT_DEFINITION_
//世界坐标点
class POINT_WORLD
{
public:
	double x;	//mm	world coordinate x or camera
	double y;	//mm	world coordinate y or camera
	double z;	//mm	world coordinate z or camera
public:
	POINT_WORLD() :x(0), y(0), z(0){}
	~POINT_WORLD(){}
};
typedef		POINT_WORLD		POINT_CAMERA;
#endif // !_WORLD_POINT_DEFINITION_


#ifndef _CALIB_NC_CODE_DEFINITION_
#define _CALIB_NC_CODE_DEFINITION_

/*!
* \class NCValue
*
* \brief NC machine value
*
*/
class  NCValue
{
public:
	double	x;		//x coordinate
	double	y;		//y coordinate
	double	z;		//z c0ordinate
	double	C;		//angle around C axis
	double	B;		//angle around B axis
	double	A;		//angle around A axis
	NCValue()
		: x(0.0f), y(0.0f), z(0.0f)
		, A(0.0f), B(0.0f), C(0.0f)
	{}
	~NCValue(){}
};

/*!
* \class CalibNCValue
*
* \brief Record the NC value corresponding to calib.
*
*/
struct CNCValue               //机床读数    VCP710
{
	CNCValue(){ num = -1; x = 0.0; y = 0.0; z = 0.0; C = 0.; B = 0.; A = 0.; CR = 0.; BR = 0.; AR = 0.; }
	int		num;   //对应求解的刀轴矢量编号

	double x;		//x 读数
	double y;		//y 读数
	double z;		//z 读数
	double C;		//C轴转角
	double B;		//B轴转角
	double A;		//A轴转角
	double CR;		//转角对应弧度  radian
	double BR;
	double AR;

};
#endif


struct CANNYPARAM               
{
	CANNYPARAM(){ lowParam = 0.; highParam = 0.; }
	double lowParam;
	double highParam;

};

class LIC_PARAM
{
public:
	double materialtype1;     //materialtype1=lic_theor_l/lic_u
	double materialtype2;

	double lic_u;//计算的当值. lic_u=lic_theor_l / lic_d_pix;   
	double lic_theor_l;//理论直径.
	double lic_d_pix;//经过亚像素提取算法得到的对应圆的直径，pix为单位。      

	double lic_d_mm;//经过亚像素提取算法得到的对应圆的直径，mm。//   lic_d_mm=lic_d_pix   *   lic_dpsize

	double lic_f;//焦距
	double lic_Dz;//最佳成像物距参数，

	double lic_cellsize;//像元大小，pix

	double lic_dpsize;//当前拍摄位置上图像上 1 个图像像素对应的物理尺寸，单位为 mm/pixel 。lic_dpsize =  lic_cellsize * lic_Dz /f；  用来计算实际的对应元直径  mm/pix




public:
	LIC_PARAM(){ materialtype1 = 0.; materialtype2 = 0.;
	
	lic_u=0;
	lic_theor_l=0; 
    lic_d_pix=0;

	lic_d_mm=0;//经

	lic_f=0;//焦距

	lic_Dz=0;//最佳拍摄距离

	lic_cellsize=4.65/1000;

	lic_dpsize=0;//
	
	}

	~LIC_PARAM(){}
private:

};



#endif