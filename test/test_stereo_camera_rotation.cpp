// add by liangliang.pan (liangliang.pan@tusimple.ai)
// test the rotation by measurement match pair

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <Eigen/Eigen>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Lmeds.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/RotationOnlySacProblem.hpp>
#include <sstream>

#include "experiment_helpers.hpp"
#include "random_generators.hpp"
#include "time_measurement.hpp"

using namespace std;
using namespace Eigen;
using namespace opengv;

#define degreesToRadians(x) (M_PI * x / 180.0)
#define radiansToDegrees(x) (x * (180.0/M_PI))

bool loadMeasurement(const string str_match_path, std::vector<Eigen::Vector2d>& pre_pt2ds, 
		     std::vector<Eigen::Vector2d>& cur_pt2ds){
  pre_pt2ds.clear();
  cur_pt2ds.clear();
  
  pre_pt2ds.reserve(100);
  cur_pt2ds.reserve(100);
  
  FILE* file = fopen(str_match_path.c_str(), "r");
  if (file == nullptr) {
    return false;
  }

  double u1, v1, u2, v2;
  while (fscanf(file, "%lf %lf %lf %lf\n", &u1, &v1, &u2, &v2) == 4) {
    Eigen::Vector2d pre_pt2d = Eigen::Vector2d(u1, v1);
    Eigen::Vector2d cur_pt2d = Eigen::Vector2d(u2, v2);
    pre_pt2ds.push_back(pre_pt2d);
    cur_pt2ds.push_back(cur_pt2d);
  }

  // std::cerr << "Loaded " << str_match_path << ": " << pre_pt2ds.size() << " poses." << std::endl;
  fclose(file);
  return true;
}

Eigen::Vector2d undistortPoint(const Eigen::Vector2d& pt2d_pixel,
			       const Eigen::Matrix3d& camera_intrinsic,
			       const std::vector<double>& distortion,
			       int max_count = 20, float epsilon = 0.01,
			       Eigen::Matrix4d tform = Eigen::Matrix4d::Identity()){

    double k[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for(int i = 0; i < distortion.size(); i++) k[i] = distortion[i];

    double fx = camera_intrinsic(0, 0), fy = camera_intrinsic(1, 1);
    double cx = camera_intrinsic(0, 2), cy = camera_intrinsic(1, 2);
    double ifx = 1. / fx, ify = 1. / fy;

    Eigen::Matrix3d R_mat = tform.block<3, 3>(0, 0);
    Eigen::Vector3d t_mat = tform.block<3, 1>(0, 3);

    double x = (pt2d_pixel(0) - cx)*ifx;
    double y = (pt2d_pixel(1) - cy)*ify;

    // compensate tilt distortion
    Eigen::Vector3d vecUntilt(x, y, 1);
    double invProj = vecUntilt(2) ? 1./vecUntilt(2) : 1;
    double x0 = x = invProj * vecUntilt(0);
    double y0 = y = invProj * vecUntilt(1);
    double error = std::numeric_limits<double>::max();

    // compensate distortion iteratively
    for( int j = 0; ; j++ ){
      if (j >= max_count) break;
      double r2 = x*x + y*y;
      double icdist = (1 + ((k[7]*r2 + k[6])*r2 + k[5])*r2)/(1 + ((k[4]*r2 + k[1])*r2 + k[0])*r2);
      double deltaX = 2*k[2]*x*y + k[3]*(r2 + 2*x*x)+ k[8]*r2+k[9]*r2*r2;
      double deltaY = k[2]*(r2 + 2*y*y) + 2*k[3]*x*y+ k[10]*r2+k[11]*r2*r2;
      x = (x0 - deltaX)*icdist;
      y = (y0 - deltaY)*icdist;
    }
    double xx = R_mat(0, 0)*x + R_mat(0, 1)*y + R_mat(0, 2) + t_mat(0);
    double yy = R_mat(1, 0)*x + R_mat(1, 1)*y + R_mat(1, 2) + t_mat(1);
    double ww = 1.0/(R_mat(2, 0)*x + R_mat(2, 1)*y + R_mat(2, 2) + t_mat(2));
    x = xx * ww, y = yy * ww;
    Eigen::Vector2d pt2d_out(x, y);
    return pt2d_out;
  }

void normalizedPt(const std::vector<Eigen::Vector2d>& pts_2d, const Eigen::Matrix3d& camera_intrinsic, 
		  const std::vector<double>& distortion, std::vector<Eigen::Vector3d>& pts_3d){

  for(int i = 0; i < pts_2d.size(); i++){
    Eigen::Vector2d pt_pixel = pts_2d[i];
    Eigen::Vector2d pt_cam = undistortPoint(pt_pixel, camera_intrinsic, distortion);
    Eigen::Vector3d pt_norm_cam = Eigen::Vector3d(pt_cam(0), pt_cam(1), 1.0);
    pt_norm_cam = pt_norm_cam/pt_norm_cam.norm();
    pts_3d.push_back(pt_norm_cam);
  }
  
  // std::cout << "pts_3d size = " << pts_3d.size() << std::endl;
}

double evaluateRotationError(const Eigen::Matrix3d& rot_est, const Eigen::Matrix3d& rot_gt){
  
  Eigen::Matrix3d error_R = rot_gt * rot_est.inverse();
  double d1 = error_R(1, 2) - error_R(2, 1);
  double d2 = error_R(2, 0) - error_R(0, 2); 
  double d3 = error_R(0, 1) - error_R(1, 0);
  double dmag = sqrt(d1*d1 + d2*d2 + d3*d3);
  double phi = asin(dmag/2);
  return radiansToDegrees(phi);
}


int main(int argc, char **argv) {
  Eigen::Matrix4d mat_cam15a_to_imu, mat_cam15b_to_imu;
  mat_cam15a_to_imu << 0.9996683198310983, 0.0009681739164341932, -0.024375458258418966, -0.03810000000000005,
		     0.024378953346914495, -0.042458067140117337, 0.9987857573916467, 1.0320000000000018,
		     -7.041008028368272e-05, -0.9991429306174537, -0.04242180146552876, 2.2850000000000015,
		     0.0, 0.0, 0.0, 1.0;
  
  mat_cam15b_to_imu << 0.9998890888084616, -0.006064200344317255, 0.010811362354273539, 1.2000030124647536,
		   -0.011071214916174875, -0.04167807090381191, 0.9990547415462313, 1.032018877221977,
		   -0.005612617267434261, -0.9991577693602306, -0.04169484925623787, 2.285499656890998,
		    0.0, 0.0, 0.0, 1.0;
		    
  std::vector<double> distortion1, distortion2;		    
  distortion1.resize(4, 0.0);
  distortion2.resize(4, 0.0);
  distortion1[0] = -0.0612;
  distortion1[1] = 1.9296;
  distortion1[2] = -0.0043;
  distortion1[3] = 0.0014;

  distortion2[0] = 0.0422;
  distortion2[1] = -2.0858;
  distortion2[2] = -0.0025;
  distortion2[3] = 0.0010;

  Eigen::Matrix3d cam15a_intrinsic, cam15b_intrinsic;
  cam15a_intrinsic << 3744.9266, 0., 512.0034, 0., 3735.5891, 288.0368, 0.0, 0.0, 1.0;	// MY00_15a
  cam15b_intrinsic << 3854.4260, 0., 526.8109, 0., 3835.6137, 312.4926, 0.0, 0.0, 1.0;  // MY00_15b
  
  // rotation_pre_to_cur = > rotation_cam15a_to_cam15b
  Eigen::Matrix3d R_cp = (mat_cam15b_to_imu.inverse() * mat_cam15a_to_imu).block<3, 3>(0, 0);

  std::vector<string> matches_paths;
  matches_paths.clear();
//   const string str_match_path = "/home/pan/Desktop/stereo_data_test/matching_results/";
//    for(int i = 0; i < 31; i++){
//      int idx = 2 * i +1;
//      string cur_str_path = str_match_path + "MY00_15a_" + to_string(idx) + ".txt";
//      matches_paths.push_back(cur_str_path);
//    }
  
  const string str_match_path = "/home/pan/Desktop/stereo_data_test/stereo_match_vo/";
  for(int i = 0; i < 30; i++){
    int idx = 2 * i;
    string cur_str_path = str_match_path  + to_string(idx) + ".txt";
    matches_paths.push_back(cur_str_path);
  }

  std::cout << " matches_paths size = " << matches_paths.size() << std::endl;
  
  double max_error_ransac = 0.0, min_error_ransac = 10.0, max_error_lmeds = 0.0, min_error_lmeds = 10.0; 
  double ave_error_ransac = 0.0, ave_error_lmeds = 0.0;
  int ave_inlier_num_ransac = 0, ave_inlier_num_lmeds = 0;
  int ave_iter_num_ransac = 0, ave_iter_num_lmeds = 0;
  int matches_num = 0;

  int sum_time = 0;
  int less_05 = 0;
  int large_05 = 0;
  int large_1 = 0;
  int large_15 = 0;
  for(int i = 0; i < 31; i++){
    // std::cout << "=================  test result " << i << " ================= " << std::endl;
    std::vector<Eigen::Vector2d> pre_pt2ds, cur_pt2ds;
    bool load_flag = loadMeasurement(matches_paths[i], pre_pt2ds, cur_pt2ds);
    if(load_flag){
      std::vector<Eigen::Vector3d> pre_pts3d, cur_pts3d;
      normalizedPt(pre_pt2ds, cam15a_intrinsic, distortion1, pre_pts3d);
      normalizedPt(cur_pt2ds, cam15b_intrinsic, distortion2, cur_pts3d);
      bearingVectors_t bearingVectors1, bearingVectors2;
      for(int i = 0; i < pre_pts3d.size(); i++){
	bearingVectors2.push_back(pre_pts3d[i]);
	bearingVectors1.push_back(cur_pts3d[i]);
      }
      matches_num += bearingVectors1.size();
      // create a central relative RANSAC adapter
      relative_pose::CentralRelativeAdapter adapter(bearingVectors1, bearingVectors2);
      sac::Ransac<sac_problems::relative_pose::RotationOnlySacProblem> ransac;
      std::shared_ptr<sac_problems::relative_pose::RotationOnlySacProblem> relposeproblem_ptr( new sac_problems::relative_pose::RotationOnlySacProblem(adapter));
      ransac.sac_model_ = relposeproblem_ptr;
      ransac.threshold_ = 2.0 * (1.0 - cos(atan(sqrt(2.0) * 0.5 / 800.0)));
      ransac.max_iterations_ = 50;
      
      // Create Lmeds
      sac::Lmeds<sac_problems::relative_pose::RotationOnlySacProblem> lmeds;
      lmeds.sac_model_ = relposeproblem_ptr;
      lmeds.threshold_ = 2.0 * (1.0 - cos(atan(sqrt(2.0) * 0.5 / 800.0)));
      lmeds.max_iterations_ = 50;

      for(int j = 0; j < 10; j++){
	ransac.computeModel(0);
	sum_time++;
	Eigen::Matrix3d R_est = ransac.model_coefficients_;
	double deg_ransac = evaluateRotationError(R_est, R_cp);
	int iteration_num_ransac = ransac.iterations_;
	int inlier_num_ransac = ransac.inliers_.size();
	std::cout << "deg_ransac = " << deg_ransac << std::endl;
	if(deg_ransac > 1.5) large_15++;
	else if(deg_ransac > 1.0 && deg_ransac < 1.5) large_1++;
	else if(deg_ransac > 0.5 && deg_ransac < 1.0) large_05++;
	else if(deg_ransac < 0.5) less_05++;
	if(deg_ransac > max_error_ransac) max_error_ransac = deg_ransac;
	if(deg_ransac < min_error_ransac) min_error_ransac = deg_ransac;
	ave_error_ransac += deg_ransac;
	ave_inlier_num_ransac += inlier_num_ransac;
	ave_iter_num_ransac += iteration_num_ransac;
	
	// lmeds method
	lmeds.computeModel(0);
	R_est = lmeds.model_coefficients_;
	double deg_lmed = evaluateRotationError(R_est, R_cp);
	int iteration_num_lmed = lmeds.iterations_;
	int inlier_num_lmed = lmeds.inliers_.size();
	
	if(deg_lmed > max_error_lmeds) max_error_lmeds = deg_lmed;
	if(deg_lmed < min_error_lmeds) min_error_lmeds = deg_lmed;
	ave_error_lmeds += deg_lmed;
	ave_inlier_num_lmeds += inlier_num_lmed;
	ave_iter_num_lmeds += iteration_num_lmed;
      }
    }
  }
  std::cout << "sum_time = " << sum_time << std::endl;
  ave_error_ransac *= (1.0/sum_time);
  ave_inlier_num_ransac *= (1.0/sum_time);
  ave_iter_num_ransac *= (1.0/sum_time);
  ave_error_lmeds *= (1.0/sum_time);
  ave_inlier_num_lmeds *= (1.0/sum_time);
  ave_iter_num_lmeds *= (1.0/sum_time);
  
  std::cout << " less_05 = " << less_05 << std::endl;
  std::cout << " large_05 = " << large_05 << std::endl;
  std::cout << " large_1 = " << large_1 << std::endl;
  std::cout << " large_15 = " << large_15 << std::endl;
  
  std::cout << "ave matches = " << matches_num/31.0 << std::endl;
  std::cout << "ave_error_ransac = " << ave_error_ransac << std::endl;
  std::cout << "max_error_ransac = " << max_error_ransac << std::endl;
  std::cout << "min_error_ransac = " << min_error_ransac << std::endl;
  std::cout << "ave_inlier_num_ransac = " << ave_inlier_num_ransac << std::endl;
  std::cout << "ave_iter_num_ransac = " << ave_iter_num_ransac << std::endl;
  
  std::cout << std::endl << std::endl;
  std::cout << "ave_error_lmeds = " << ave_error_lmeds << std::endl;
  std::cout << "max_error_lmeds = " << max_error_lmeds << std::endl;
  std::cout << "min_error_lmeds = " << min_error_lmeds << std::endl;
  std::cout << "ave_inlier_num_lmeds = " << ave_inlier_num_lmeds << std::endl;
  std::cout << "ave_iter_num_lmeds = " << ave_iter_num_lmeds << std::endl;

}
