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

#define			PIXELSIZE_CAMERA			(3.45*1E-3)				//��������ߴ��С   
#endif // !_CAMERA_PIXEL_SIZE_DEF_



#ifndef _IMAGE_2_WORLD_RELATIONSHIP_
#define _IMAGE_2_WORLD_RELATIONSHIP_
//����Բ��ͼ��Բ��Ӧ��ϵ��Physic Image
class MatchPoint
{
public:
	MatchPoint()
		:num(-1), imgx(0), imgy(0), imgR(0)
		, objx(0), objy(0), objz(0), reprojectDeviation(-1)
	{}
	~MatchPoint(){}

public:
	int		num;		//���, ���û�п��Ժ���
	double	imgx;		//ͼ��x����
	double	imgy;		//ͼ��y����
	double	imgR;		//ͼ���п׵İ뾶�����ص�λ��

	double	objx;		//����x����
	double	objy;		//����y����
	double	objz;		//����z����

	double reprojectDeviation;
};
#endif // !_IMAGE_2_WORLD_RELATIONSHIP_


#ifndef _INTRINSIC_PARAM_DEFINITION_
#define _INTRINSIC_PARAM_DEFINITION_
//�ڲ��������Ų�����
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
	double		gamma;					//ƫбϵ��

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
//��������� �Ų�����
class EXTRC_PARAM
{
public:
	double	T[3];						//mm  transform
	double	R[3];						//gamma,beta,alpha. in rad�����ȣ� not in centigrade
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
//���������
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
struct CNCValue               //��������    VCP710
{
	CNCValue(){ num = -1; x = 0.0; y = 0.0; z = 0.0; C = 0.; B = 0.; A = 0.; CR = 0.; BR = 0.; AR = 0.; }
	int		num;   //��Ӧ���ĵ���ʸ�����

	double x;		//x ����
	double y;		//y ����
	double z;		//z ����
	double C;		//C��ת��
	double B;		//B��ת��
	double A;		//A��ת��
	double CR;		//ת�Ƕ�Ӧ����  radian
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

	double lic_u;//����ĵ�ֵ. lic_u=lic_theor_l / lic_d_pix;   
	double lic_theor_l;//����ֱ��.
	double lic_d_pix;//������������ȡ�㷨�õ��Ķ�ӦԲ��ֱ����pixΪ��λ��      

	double lic_d_mm;//������������ȡ�㷨�õ��Ķ�ӦԲ��ֱ����mm��//   lic_d_mm=lic_d_pix   *   lic_dpsize

	double lic_f;//����
	double lic_Dz;//��ѳ�����������

	double lic_cellsize;//��Ԫ��С��pix

	double lic_dpsize;//��ǰ����λ����ͼ���� 1 ��ͼ�����ض�Ӧ������ߴ磬��λΪ mm/pixel ��lic_dpsize =  lic_cellsize * lic_Dz /f��  ��������ʵ�ʵĶ�ӦԪֱ��  mm/pix




public:
	LIC_PARAM(){ materialtype1 = 0.; materialtype2 = 0.;
	
	lic_u=0;
	lic_theor_l=0; 
    lic_d_pix=0;

	lic_d_mm=0;//��

	lic_f=0;//����

	lic_Dz=0;//����������

	lic_cellsize=4.65/1000;

	lic_dpsize=0;//
	
	}

	~LIC_PARAM(){}
private:

};



#endif