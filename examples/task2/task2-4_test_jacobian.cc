//
// Created by caoqi on 2018/8/31.
//


//3D:  1.36939, -1.17123, 7.04869
//obs: 0.180123 -0.156584


#include "sfm/bundle_adjustment.h"
#include"math/matrix.h"
/*
 * This function computes the Jacobian entries for the given camera and
 * 3D point pair that leads to one observation.
 *
 * The camera block 'cam_x_ptr' and 'cam_y_ptr' is:
 * - ID 0: Derivative of focal length f
 * - ID 1-2: Derivative of distortion parameters k0, k1
 * - ID 3-5: Derivative of translation t0, t1, t2
 * - ID 6-8: Derivative of rotation w0, w1, w2
 *
 * The 3D point block 'point_x_ptr' and 'point_y_ptr' is:
 * - ID 0-2: Derivative in x, y, and z direction.
 *
 * The function that leads to the observation is given as follows:
 *
 *   u = f * D(x,y) * x  (image observation x coordinate)
 *   v = f * D(x,y) * y  (image observation y coordinate)
 *
 * with the following definitions:
 *
 *   xc = R0 * X + t0  (homogeneous projection)
 *   yc = R1 * X + t1  (homogeneous projection)
 *   zc = R2 * X + t2  (homogeneous projection)
 *   x = xc / zc  (central projection)
 *   y = yc / zc  (central projection)
 *   D(x, y) = 1 + k0 (x^2 + y^2) + k1 (x^2 + y^2)^2  (distortion)
 */

 /**
  * /description 给定一个相机参数和一个三维点坐标，求解雅各比矩阵，即公式中的df(theta)/dtheta
  * @param cam       相机参数
  * @param point     三维点坐标
  * @param cam_x_ptr 重投影坐标x 相对于相机参数的偏导数，相机有9个参数： [0] 焦距f; [1-2] 径向畸变系数k1, k2; [3-5] 平移向量 t1, t2, t3
  *                                                               [6-8] 旋转矩阵（角轴向量）
  * @param cam_y_ptr    重投影坐标y 相对于相机参数的偏导数，相机有9个参数
  * @param point_x_ptr  重投影坐标x 相对于三维点坐标的偏导数
  * @param point_y_ptr  重投影坐标y 相对于三维点坐标的偏导数
  */
void jacobian(sfm::ba::Camera const& cam,
              sfm::ba::Point3D const& point,
              double* cam_x_ptr, double* cam_y_ptr,
              double* point_x_ptr, double* point_y_ptr)
{
    const double f = cam.focal_length;
    const double *R = cam.rotation;
    const double *t = cam.translation;
    const double *X = point.pos;
    const double k0 = cam.distortion[0];
    const double k1 = cam.distortion[1];
	
#pragma region modified by rui
	//计算当前三维点的重投影坐标和一些必要变量
	math::Vec2d p;
	math::Vec3d p3d(point.pos);
	math::Matrix<double, 3, 3> rotationW2C(cam.rotation);   //使用math::matrix 的explicit构造函数
	math::Vec3d trans(cam.translation);                            //使用math::vector 的explicit构造函数
	//像素坐标observation，相机坐标系坐标cameraCoor
	auto cameraCoor = rotationW2C * p3d + trans;   //（xc,yc,zc）=R*p3d+t
	auto normlizedCoor = cameraCoor;     
	normlizedCoor = normlizedCoor / normlizedCoor[2];   //利用zc归一化
	//std::cout << observation[0] << std::endl << observation[1] << std::endl;

    // 相机焦距的偏导数
	//推导结果： du/df=(1 + (k0*r2 + k1 * r2*r2))*x
	//          dv/df=(1 + (k0*r2 + k1 * r2*r2))*y
	double r2 = (pow(normlizedCoor[0], 2) + pow(normlizedCoor[1], 2));
	cam_x_ptr[0] = (1 + (k0*r2 + k1 * r2*r2))*normlizedCoor[0];
    cam_y_ptr[0] = (1 + (k0*r2 + k1 * r2*r2))*normlizedCoor[1];

    // 相机畸变的偏导数
    //推导结果： du/dk0=f*x*r2
	//          du/dk1=f*x*r2*r2
	//          dv/dk0=f*y*r2
	//          dv/dk1=f*y*r2*r2
	cam_x_ptr[1] = f * normlizedCoor[0] * r2;
	cam_x_ptr[2] = f * normlizedCoor[0] * r2*r2;
    cam_y_ptr[1] = f * normlizedCoor[1] * r2;
    cam_y_ptr[2] = f * normlizedCoor[1] * r2*r2;

    //// 相机将向畸变系数的偏导数
    //cam_x_ptr[1] = 0.0;
    //cam_x_ptr[2] = 0.0;
    //cam_y_ptr[1] = 0.0;
    //cam_y_ptr[2] = 0.0;

    // 相机平移向量的偏导数
	//推导结果：du/dt0=du/dxc=f*x*(k0+2*k1*r2)*2*x/zc+f*d(k0,k1,r2)/zc
	//         du/dt1=du/dyc=f*x*(k0+2*k1*r2)*2*y/zc
	//         du/dt2=du/dzc=f*x*(-k0-2*k1*r2)*2*r2^2/zc+f* (1 + (k0*r2 + k1 * r2*r2))*(-x)/zc
	//         dv/dt1=dv/dyc=f*y*(k0+2*k1*r2)*2*y/zc+f*(1 + (k0*r2 + k1 * r2*r2))/zc
	//         dv/dt0=dv/dxc=f*y*(k0+2*k1*r2)*2*y/zc
	//         dv/dt2=dv/dzc=f*y*(-k0-2*k1*r2)*2*r2^2/zc+f* (1 + (k0*r2 + k1 * r2*r2))*(-y)/zc
	cam_x_ptr[3] = f*normlizedCoor[0]*(k0+2*k1*r2)*2*normlizedCoor[0]/cameraCoor[2]+f* (1 + (k0*r2 + k1 * r2*r2))/cameraCoor[2];
    cam_x_ptr[4] = f * normlizedCoor[0] * (k0 + 2 * k1*r2) * 2 * normlizedCoor[1] / cameraCoor[2];
    cam_x_ptr[5] = f * normlizedCoor[0] * (-k0 - 2 * k1*r2) * 2 * r2  / cameraCoor[2]+ f * (1 + (k0*r2 + k1 * r2*r2))*(-normlizedCoor[0]) / cameraCoor[2];
    cam_y_ptr[3] = f * normlizedCoor[1] * (k0 + 2 * k1*r2) * 2 * normlizedCoor[0] / cameraCoor[2];
    cam_y_ptr[4] = f * normlizedCoor[1] * (k0 + 2 * k1*r2) * 2 * normlizedCoor[1] / cameraCoor[2]+ f * (1 + (k0*r2 + k1 * r2*r2)) / cameraCoor[2];
    cam_y_ptr[5] = f * normlizedCoor[1] * (-k0 - 2 * k1*r2) * 2 * r2 / cameraCoor[2] + f * (1 + (k0*r2 + k1 * r2*r2))*(-normlizedCoor[1]) / cameraCoor[2];

    // 相机旋转矩阵的偏导数
	//求出RX
	auto RX = rotationW2C * p3d;
	//由于旋转矩阵偏导数求解中，需要求du/dxc，其已经由上一步中求出，即du/dt0=du/dxc(一类导数)，所以在下式中直接使用cam_x_ptr已经计算的结果代替
	//推导结果： du/dw0=du/dyc*(-RX[2]) +du/dzc*(RX[1])
	//          du/dw1=du/dxc*(RX[2]) +du/dzc*(-RX[0])
	//          du/dw2=du/dxc*(-RX[1]) +du/dyc*(RX[0])
	//          dv/dw0=dv/dyc*(-RX[1]) +dv/dzc*(RX[1])
	//          dv/dw1=dv/dxc*(RX[2]) +dv/dzc*(-RX[0])
	//          dv/dw2=dv/dxc*(-RX[1])+dv/dyc*(RX[0])
	cam_x_ptr[6] = cam_x_ptr[4] * (-RX(2)) + cam_x_ptr[5] * (RX[1]);
	cam_x_ptr[7] = cam_x_ptr[3] * (RX(2)) + cam_x_ptr[5] * (-RX(0));
	cam_x_ptr[8] = cam_x_ptr[3] * (-RX(1)) + cam_x_ptr[4] * (RX(0));
	cam_y_ptr[6] = cam_y_ptr[4] * (-RX(2)) + cam_y_ptr[5] * (RX(1));
	cam_y_ptr[7] = cam_y_ptr[3] * (RX(2)) + cam_y_ptr[5] * (-RX(0));
	cam_y_ptr[8] = cam_y_ptr[3] * (-RX(1)) + cam_y_ptr[4] * (RX(0));

    // 三维点的偏导数
    //由于下列偏导数求解中，需要求du/dxc，其已经由上一步中求出，即du/dt0=du/dxc(一类导数)，所以在下式中直接使用cam_x_ptr已经计算的结果代替
	//推导结果： du/dX=du/dxc*R00+du/dyc*R10+du/dzc*R20
	//          du/dY=du/dxc*R01+du/dyc*R11+du/dzc*R21
	//          du/dZ=du/dxc*R02+du/dyc*R12+du/dzc*R22
	//          dv/dX=dv/dxc*R00+dv/dyc*R10+dv/dzc*R20
	//          dv/dY=dv/dxc*R01+dv/dyc*R11+dv/dzc*R21
	//          dv/dZ=dv/dxc*R02+dv/dyc*R12+dv/dzc*R22
	// 由上式可知，[du/dX,du/dY,du/dZ]=[X,Y,Z]*R
	//            [dv/dX,dv/dY,dv/dZ]=[X,Y,Z]*R
	math::Vec3d uDerivedRegard3dTranslation(cam_x_ptr[3], cam_x_ptr[4], cam_x_ptr[5]);
	math::Vec3d vDerivedRegard3dTranslation(cam_y_ptr[3], cam_y_ptr[4], cam_y_ptr[5]);
	auto uDerivedRegard3dPosition = rotationW2C.transposed() * uDerivedRegard3dTranslation;   //
	auto vDerivedRegard3dPosition = rotationW2C.transposed() * vDerivedRegard3dTranslation;
    point_x_ptr[0] = uDerivedRegard3dPosition(0);
    point_x_ptr[1] = uDerivedRegard3dPosition(1);
    point_x_ptr[2] = uDerivedRegard3dPosition(2);
    point_y_ptr[0] = vDerivedRegard3dPosition(0);
    point_y_ptr[1] = vDerivedRegard3dPosition(1);
    point_y_ptr[2] = vDerivedRegard3dPosition(2);
#pragma endregion
}
int main(int argc, char*argv[])
{

    sfm::ba::Camera cam;
    cam.focal_length  =  0.919654;
    cam.distortion[0] = -0.108298;
    cam.distortion[1] =  0.103775;

    cam.rotation[0] = 0.999999;
    cam.rotation[1] = -0.000676196;
    cam.rotation[2] = -0.0013484;
    cam.rotation[3] = 0.000663243;
    cam.rotation[4] = 0.999949;
    cam.rotation[5] = -0.0104095;
    cam.rotation[6] = 0.00135482;
    cam.rotation[7] = 0.0104087;
    cam.rotation[8] = 0.999949;

    cam.translation[0]=0.00278292;
    cam.translation[1]=0.0587996;
    cam.translation[2]=-0.127624;

    sfm::ba::Point3D pt3D;
    pt3D.pos[0]= 1.36939;
    pt3D.pos[1]= -1.17123;
    pt3D.pos[2]= 7.04869;

    double cam_x_ptr[9]={0};
    double cam_y_ptr[9]={0};
    double point_x_ptr[3]={0};
    double point_y_ptr[3]={0};

    jacobian(cam, pt3D, cam_x_ptr, cam_y_ptr, point_x_ptr, point_y_ptr);


   std::cout<<"Result is :"<<std::endl;
    std::cout<<"cam_x_ptr: ";
    for(int i=0; i<9; i++){
        std::cout<<cam_x_ptr[i]<<" ";
    }
    std::cout<<std::endl;

    std::cout<<"cam_y_ptr: ";
    for(int i=0; i<9; i++){

        std::cout<<cam_y_ptr[i]<<" ";
    }
    std::cout<<std::endl;

    std::cout<<"point_x_ptr: ";
    std::cout<<point_x_ptr[0]<<" "<<point_x_ptr[1]<<" "<<point_x_ptr[2]<<std::endl;

    std::cout<<"point_y_ptr: ";
    std::cout<<point_y_ptr[0]<<" "<<point_y_ptr[1]<<" "<<point_y_ptr[2]<<std::endl;


    std::cout<<"\nResult should be :\n"
       <<"cam_x_ptr: 0.195942 0.0123983 0.000847141 0.131188 0.000847456 -0.0257388 0.0260453 0.95832 0.164303\n"
       <<"cam_y_ptr: -0.170272 -0.010774 -0.000736159 0.000847456 0.131426 0.0223669 -0.952795 -0.0244697 0.179883\n"
       <<"point_x_ptr: 0.131153 0.000490796 -0.0259232\n"
       <<"point_y_ptr: 0.000964926 0.131652 0.0209965\n";


    return 0;
}
