//Copyright(c) 2015 Shuda Li[lishuda1980@gmail.com]
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files(the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions :
//
//The above copyright notice and this permission notice shall be included in
//all copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//FOR A PARTICULAR PURPOSE AND NON - INFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR
//COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.


#include <thrust/device_ptr.h>
#include <thrust/sort.h>
//#include <thrust/functional.h>

#define INFO
#include <GL/glew.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
//stl
#include <iostream>
#include <string>
#include <vector>
#include <limits>

#ifdef __gnu_linux__
#include <sys/types.h>
#include <sys/stat.h>
#elif _WIN32
#include <direct.h>
#else 
#error "OS not supported!"
#endif

#define _USE_MATH_DEFINES
#include <math.h>
#include <numeric>
#include <experimental/filesystem>

//openncv
#include <opencv2/opencv.hpp>
// #include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudawarping.hpp>

#include <utility>
#include <OpenNI.h>
#include "Converters.hpp"
#include "GLUtil.hpp"
#include "CVUtil.hpp"
#include <sophus/se3.hpp>
#include "EigenUtil.hpp"
#include "pcl/internal.h"
#include <map>
#include "Camera.h"
#include "RGBDFrame.h"
#include "VideoSourceKinect.hpp"
#include "CubicGrids.h"

#define  SURF_KEYPOINT
#include "KinfuTracker.h"
//#include "Helper.hpp"
#include "KinfuTracker.cuh"
#include "FeatureVolume.cuh"
#include "CudaLib.cuh" 

#include "ICPFrame2Frame.cuh"
#include "RayCaster.cuh"
#include <limits>
#include "Utility.hpp"
#include "Kinect.h"
#include "pcl/vector_math.hpp"
#include "Converters.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::cuda;
using namespace std;
using namespace btl::kinect;
using namespace btl::utility;
using namespace btl::image;
using namespace btl::device;
using namespace pcl::device;
using namespace Eigen;

namespace btl{ namespace geometry
{
void convert(const Eigen::Affine3f& eiM_, Mat* pM_){
	pM_->create(4, 4, CV_32FC1);
	for (int r = 0; r < 4; r++)
		for (int c = 0; c < 4; c++)
		{
		pM_->at<float>(r, c) = eiM_(r, c);
		}

	return;
}

void convert(const Eigen::Matrix4f& eiM_, Mat* pM_ ){
	pM_->create(4,4,CV_32FC1);
	for (int r=0; r<4; r++)
		for (int c=0; c<4; c++)
		{
			pM_->at<float>(r,c) = eiM_(r,c);
		}

		return;
}

void convert(const Mat& M_, Eigen::Affine3f* peiM_){

	for (int r = 0; r < 4; r++)
		for (int c = 0; c < 4; c++)
		{
		(*peiM_)(r, c) = M_.at<float>(r, c);
		}

	return;
}

void convert(const Mat& M_, Eigen::Matrix4f* peiM_ ){

	for (int r=0; r<4; r++)
		for (int c=0; c<4; c++)
		{
			(*peiM_)(r,c) = M_.at<float>(r,c);
		}

		return;
}

CKinFuTracker::CKinFuTracker(CRGBDFrame::tp_ptr pKeyFrame_, CCubicGrids::tp_shared_ptr pCubicGrids_, int nResolution_/*=0*/, int nPyrHeight_/*=3*/, string& initialPoseEstimationMathod_, string& ICPMethod_, string& matchingMethod_)
	:_pCubicGrids(pCubicGrids_),_nResolution(nResolution_),_nPyrHeight(nPyrHeight_)
{
	_buffer_size = 1500;
	_nFeatureScale = pCubicGrids_->_nFeatureScale;
	_nFeatureName = pCubicGrids_->_nFeatureName;
	_nGlobalKeyPointDimension = pCubicGrids_->_nKeyPointDimension;
	setKinFuTracker(initialPoseEstimationMathod_, ICPMethod_, matchingMethod_);
	CKinFuTracker::reset();
}

void CKinFuTracker::reset(){
	_descriptor_bytes = 64;
	if (_nFeatureName == btl::BINBOOST)
		_descriptor_bytes = 8;

	_nAppearanceMatched = 0;
	_v_prj_c_f_w_training.clear();

	_bRelocWRTVolume = true;
	_bTrackingOnly = false;
	_nMinFeatures[0] = 50;
	_nMinFeatures[1] = 10;
	_nMinFeatures[2] = 5;
	_nMinFeatures[3] = 3;
	_nMinFeatures[4] = 1;

	_aMinICPEnergy[0] = .012f;//6.f;
	_aMinICPEnergy[1] = .04f;//100.f;
	_aMinICPEnergy[2] = .1f;//2000.f;
	_aMinICPEnergy[3] = 1.f;// 2000.f;
	_aMinICPEnergy[4] = 2000.f;//1000.f;

	_aMaxChainLength[0] = 70;
	_aMaxChainLength[1] = 35;
	_aMaxChainLength[2] = 20;
	_aMaxChainLength[3] = 7;

	//hamming bf matcher
	_pMatcherGpuHamming = cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	//flann matcher
	Ptr<flann::IndexParams> ptr(new flann::LshIndexParams(20,10,2) );
	_pMatcherFlann.reset();
	_pMatcherFlann.reset( new FlannBasedMatcher( ptr ) );

	//allocate surf
	_nHessianThreshold = 10;
	//allocate surf cpu
	if (_nFeatureName == btl::BINBOOST){
		_pSurf = cv::xfeatures2d::SURF::create(_nHessianThreshold, 4, 2, false, false);
	}
	else {
		_pSurf = cv::xfeatures2d::SURF::create(_nHessianThreshold, 4, 2, false, true);
	}

	//allocate brisk
	_pBRISK = BRISK::create(30, 4);

	_pGpuTM_bw = cuda::createTemplateMatching(CV_8U, CV_TM_CCORR_NORMED );

	//reloc
	_gpu_idx_all_curr.create(1, _buffer_size, CV_16SC1);//gpu 
	_gpu_DOs_all_curr.create(1, _buffer_size, CV_32FC3);//gpu 
	_gpu_other_all_curr.create(1, _buffer_size, CV_32FC3);//gpu 
	_gpu_counter.create(1, _buffer_size, CV_8UC1);//gpu 

	_gpu_pts_all_curr.create(1, _buffer_size, CV_32FC3);//gpu 
	_gpu_pts_all_prev.create(1, _buffer_size, CV_32FC3);//gpu 
	_gpu_nls_all_curr.create(1, _buffer_size, CV_32FC3);//gpu 
	_gpu_nls_all_prev.create(1, _buffer_size, CV_32FC3);//gpu 
	_gpu_mds_all_curr.create(1, _buffer_size, CV_32FC3);//gpu 
	_gpu_mds_all_prev.create(1, _buffer_size, CV_32FC3);//gpu 
	_gpu_mask_all_curr.create(1, _buffer_size, CV_8UC1);//gpu 
	_gpu_mask_all_prev.create(1, _buffer_size, CV_8UC1);//gpu 
	_gpu_mask_all_2d_curr.create(1, _buffer_size, CV_8UC1);//gpu 
	_gpu_mask_all_2d_prev.create(1, _buffer_size, CV_8UC1);//gpu 
	_gpu_keypoints_all_curr.create(10, _buffer_size, CV_32FC1);//gpu 
	_gpu_keypoints_all_prev.create(10, _buffer_size, CV_32FC1);//gpu 
	_gpu_descriptor_all_curr.create(_buffer_size, _descriptor_bytes, CV_8UC1);//gpu 
	_gpu_descriptor_all_prev.create(_buffer_size, _descriptor_bytes, CV_8UC1);//gpu 
	_gpu_c2p_idx.create(1, _buffer_size*2, CV_16SC2);//gpu 
	_gpu_mask_counting.create(1, _buffer_size*2, CV_8UC1);//gpu 

	_gpu_idx_curr_2_prev_reloc.create(1, _buffer_size, CV_32SC2);//gpu 
	_gpu_hamming_distance_reloc.create(1, _buffer_size, CV_32FC1);//gpu 
	_gpu_distinctive_reloc.create(1, _buffer_size, CV_32FC1);//gpu 
	_gpu_pts_curr_reloc.create(1, _buffer_size, CV_32FC3);//gpu 
	_gpu_nls_curr_reloc.create(1, _buffer_size, CV_32FC3);//gpu 
	_gpu_mds_curr_reloc.create(1,_buffer_size,CV_32FC3);//gpu 
	_gpu_pts_global_reloc.create(1,_buffer_size,CV_32FC3);//gpu 
	_gpu_nls_global_reloc.create(1,_buffer_size,CV_32FC3);//gpu 
	_gpu_mds_global_reloc.create(1,_buffer_size,CV_32FC3);//gpu 
	_gpu_2d_curr_reloc.create(1, _buffer_size, CV_32FC2);
	_gpu_bv_curr_reloc.create(1, _buffer_size, CV_32FC3);
	_gpu_weight_reloc.create(3, _buffer_size, CV_16SC1);
	_gpu_keypoint_curr_reloc.create(10,_buffer_size,CV_32FC1);
	_gpu_relined_inliers.create(1, _buffer_size, CV_32SC1);

	_gpu_descriptor_curr_reloc.create(_buffer_size, _descriptor_bytes, CV_8UC1);

	_gpu_descriptor_prev.resize(_nFeatureScale);
	_gpu_key_points_prev.resize(_nFeatureScale);
	_gpu_pts_prev.resize(_nFeatureScale);
	_gpu_nls_prev.resize(_nFeatureScale);
	_gpu_mds_prev.resize(_nFeatureScale);

	_gpu_pts_curr.resize(_nFeatureScale);
	_gpu_nls_curr.resize(_nFeatureScale);
	_gpu_mds_curr.resize(_nFeatureScale);
	_gpu_2d_curr.resize(_nFeatureScale);
	_gpu_bv_curr.resize(_nFeatureScale);
	_gpu_weights.resize(_nFeatureScale);
	_gpu_descriptor_curr.resize(_nFeatureScale);
	_gpu_keypoints_curr.resize(_nFeatureScale);

	_quadratic = gluNewQuadric();                // Create A Pointer To The Quadric Object ( NEW )
	// Can also use GLU_NONE, GLU_FLAT
	gluQuadricNormals(_quadratic, GLU_SMOOTH); // Create Smooth Normals
	gluQuadricTexture(_quadratic, GL_TRUE);   // Create Texture Coords ( NEW )

	_fMatchThreshold = 30.f;
	_nSearchRange = 50;
	_nRansacIterationasTracking = 15;
	_nRansacIterationasRelocalisation = 100;

	_K_tracking = 1;
	_K_relocalisation = 5;

	_bLoadVolume = false;
	_fPercantage = 0.6f;
	_bIsVolumeLoaded = false;

	return;
}
	
void CKinFuTracker::setKinFuTracker(string& initialPoseEstimationMethodName_, string& ICPMethods_, string& MatchingMethodName_){
	if (!initialPoseEstimationMethodName_.compare("ICP")){
		setPoseEstimationMethod(CKinFuTracker::ICP);
	}
	else if(!initialPoseEstimationMethodName_.compare("RANSACPnP")){
		setPoseEstimationMethod(CKinFuTracker::RANSACPnP);
	}
	else if(!initialPoseEstimationMethodName_.compare("AO")){
		setPoseEstimationMethod(CKinFuTracker::AO);
	}
	else if(!initialPoseEstimationMethodName_.compare("AORansac")){
		setPoseEstimationMethod(CKinFuTracker::AORansac);
	}
	else if(!initialPoseEstimationMethodName_.compare("AONRansac")){
		setPoseEstimationMethod(CKinFuTracker::AONRansac);
	}
	else if(!initialPoseEstimationMethodName_.compare("AONn2dRansac")){
		setPoseEstimationMethod(CKinFuTracker::AONn2dRansac);
	}
	else if (!initialPoseEstimationMethodName_.compare("AONn2dRansac2")){
		setPoseEstimationMethod(CKinFuTracker::AONn2dRansac2);
	}
	else if(!initialPoseEstimationMethodName_.compare("AONMDn2dRansac")){
		setPoseEstimationMethod(CKinFuTracker::AONMDn2dRansac);
	}
	
	if (!ICPMethods_.compare("Frm_2_Frm_ICP")){
		setICPMethod(btl::Frm_2_Frm_ICP);
	}
	else if (!ICPMethods_.compare("Vol_2_Frm_ICP")){
		setICPMethod(btl::Vol_2_Frm_ICP);
	}
	else if (!ICPMethods_.compare("Combined_ICP")){
		setICPMethod(btl::Combined_ICP);
	}

	if(!MatchingMethodName_.compare("GM")){
		setMatchingMethod(CKinFuTracker::GM);
	}
	else if(!MatchingMethodName_.compare("GM_GPU")){
		setMatchingMethod(CKinFuTracker::GM_GPU);
	}
	else if(!MatchingMethodName_.compare("IDF")){
		setMatchingMethod(CKinFuTracker::IDF);
	}
	else if(!MatchingMethodName_.compare("ICP")){
		setMatchingMethod(CKinFuTracker::ICP);
	}
	
	return;
}

bool CKinFuTracker::init(const CRGBDFrame::tp_ptr pCurFrame_){

	_pCubicGrids->_fx = _pCubicGrids->_intrinsics.fx = _intrinsics.fx = _fx = pCurFrame_->_pRGBCamera->_fFx;
	_pCubicGrids->_fy = _pCubicGrids->_intrinsics.fy = _intrinsics.fy = _fy = pCurFrame_->_pRGBCamera->_fFy;
	_pCubicGrids->_cx = _pCubicGrids->_intrinsics.cx = _intrinsics.cx = _cx = pCurFrame_->_pRGBCamera->_u;
	_pCubicGrids->_cy = _pCubicGrids->_intrinsics.cy = _intrinsics.cy = _cy = pCurFrame_->_pRGBCamera->_v;

	_cvmA = pCurFrame_->_pRGBCamera->getcvmK();
	_visual_angle_threshold = atan(8.f / ((_fx + _fy) / 2)); //8 pixel visual angle
	_normal_angle_threshold = float( M_PI_4 / 3);//15 degree and 30 degree for 7 scenes
	_distance_threshold = 0.2f; //5mm for ours 5cm for 7-scenes
	_nCols = pCurFrame_->_acvgmShrPtrPyrRGBs[0]->cols;
	_nRows = pCurFrame_->_acvgmShrPtrPyrRGBs[0]->rows;

	_gpu_key_array_2d.create(_nRows, _nCols, CV_32SC1);

	if( !_bIsVolumeLoaded ){//tracking and mapping using gt
		//_pCubicGrids->reset();

		_v_relocalisation_p_matrices.clear();
		
		if (_nMatchingMethod == IDF || _nMatchingMethod == GM || _nMatchingMethod == GM_GPU ){
			//Note: the point cloud of the pKeyFrame is located in camera coordinate system

			//extract features
			extractFeatures(pCurFrame_, _nFeatureName ); 
			vector<int> vCandidates = cuda_extract_key_points(_intrinsics, _pCubicGrids->_fFeatureVoxelSizeM, _nFeatureScale, // basal feature voxel size
				*pCurFrame_->_acvgmShrPtrPyrPts[0], *pCurFrame_->_acvgmShrPtrPyrNls[0], *pCurFrame_->_acvgmShrPtrPyrDepths[0], *pCurFrame_->_acvgmShrPtrPyrReliability[2],
																		_gpu_descriptor_all_curr, _gpu_keypoints_all_curr, _n_detected_features_curr,
                                                                         &_gpu_descriptor_curr, &_gpu_keypoints_curr,
                                                                         &_gpu_pts_curr, &_gpu_nls_curr, &_gpu_mds_curr, &_gpu_2d_curr, &_gpu_bv_curr, &_gpu_weights );
			int nCandidates = std::accumulate(vCandidates.begin(), vCandidates.end(),0);
			//# of features _gpu_descriptor_curr = _gpu_keypoints_curr = _gpu_pts_curr = _gpu_nls_curr = _gpu_mds_curr = _gpu_2d_curr 

			//the initial frame must contains certain # of features
            int safe_num[]={1500,500};
            if ( nCandidates < 10 || nCandidates > safe_num[_nResolution] ) return false;
		}

		_pPrevFrameWorld.reset();
		_pPrevFrameWorld.reset(new CRGBDFrame(pCurFrame_));	//copy pKeyFrame_ to _pPrevFrameWorld
	}
	
	if (_bLoadVolume ) {
		if( !_bIsVolumeLoaded ){//load volume
			cout << ("Loading a global model...\n");
			_pCubicGrids->loadNIFTI(_path_to_global_model);
			loadGlobalFeaturesAndCameraPoses(_path_to_global_model);
			_bIsVolumeLoaded = true;
		}
		
		if (_bRelocWRTVolume) {//relocalisation goes here
			_bTrackingOnly = true;
			int nStage = _nStage;
			if (relocalise(pCurFrame_)){ //use previous frame
				//input key frame must be defined in local camera system
				//initialize pose
				_nStage = nStage;

				return true;
			}
			else{
				_nStage = nStage;
				return false;
			}
		}
	}

	pCurFrame_->gpuTransformToWorld();//transform from camera to world
	//integrate the frame into the world
	_pCubicGrids->integrateDepth(*pCurFrame_, _intrinsics);

	//initialize pose
	pCurFrame_->getPrjCfW(&_pose_feature_c_f_w);//use the proj_w_2_c of the current frame to set the initial
	pCurFrame_->getPrjCfW(&_pose_refined_c_f_w);
	if (_nStage == btl::Tracking_n_Mapping)
		_v_prj_c_f_w_training.push_back(_pose_refined_c_f_w);

	//store current key point
	storeCurrFrame( pCurFrame_ );
	cout << ("End of init\n");

	return true;
}

void CKinFuTracker::storeCurrFrame(CRGBDFrame::tp_ptr pCurFrame_){
	_pPrevFrameWorld->copyRTFrom(&*pCurFrame_);
	_pCubicGrids->rayCast(&*_pPrevFrameWorld);//this line is critical for the stability's of tracking_n_mapping in practice

	if (_nMatchingMethod == IDF || _nMatchingMethod == GM || _nMatchingMethod == GM_GPU ){
		_n_detected_features_prev = _n_detected_features_curr;
		Matrix3f R_; Vector3f T_;
		btl::utility::convertPrj2Rnt(_pose_refined_c_f_w, &R_, &T_);
		_gpu_pts_prev.clear();
		for (int i = 0; i < _nFeatureScale; i++)
		{
			_gpu_pts_prev.push_back(GpuMat());
			if (_gpu_keypoints_curr[i].empty() || _gpu_descriptor_curr[i].empty() || 
				_gpu_pts_curr[i].empty() || _gpu_nls_curr[i].empty() || 
				_gpu_mds_curr[i].empty())
			{
				_gpu_key_points_prev[i].release();
				_gpu_descriptor_prev[i].release();
				_gpu_pts_prev[i].release();
				_gpu_nls_prev[i].release();
				_gpu_mds_prev[i].release();
			}
			else
			{
				_gpu_key_points_prev[i] = _gpu_keypoints_curr[i].clone();
				_gpu_descriptor_prev[i] = _gpu_descriptor_curr[i].clone();
				btl::device::cuda_transform_local2world(R_.data(), T_.data(), &_gpu_pts_curr[i], &_gpu_nls_curr[i], &_gpu_mds_curr[i]);
				_gpu_pts_prev[i] = _gpu_pts_curr[i].clone();
				_gpu_nls_prev[i] = _gpu_nls_curr[i].clone();
				_gpu_mds_prev[i] = _gpu_mds_curr[i].clone();
			}
		}
	}
	return;
}

void CKinFuTracker::extractFeatures(const CRGBDFrame::tp_ptr pFrame_, int nFeatureType_){
	vector< cv::KeyPoint > vKeyPointsCurr;
	cv::Mat cvmDescriptorCurr;
	switch (nFeatureType_)
	{
	case btl::BRISKC:
		//extract BRISK features
		//using surf keypoints
		_pSurf->detect(*pFrame_->_acvmShrPtrPyrBWs[0], vKeyPointsCurr);
		if (vKeyPointsCurr.size() < _nMinFeatures[_nResolution]){ _n_detected_features_curr = 0;  return; }
		_pBRISK->compute(*pFrame_->_acvmShrPtrPyrBWs[0], vKeyPointsCurr, cvmDescriptorCurr);
		{
			Mat keypointsCPU(_gpu_keypoints_all_curr.size(), CV_32FC1);

			enum KeypointLayout
			{
				X_ROW = 0,
				Y_ROW,
				LAPLACIAN_ROW,
				OCTAVE_ROW,
				SIZE_ROW,
				ANGLE_ROW,
				HESSIAN_ROW,
				ROWS_COUNT
			};

			float* kp_x = keypointsCPU.ptr<float>(X_ROW);
			float* kp_y = keypointsCPU.ptr<float>(Y_ROW);
			int* kp_laplacian = keypointsCPU.ptr<int>(LAPLACIAN_ROW);
			int* kp_octave = keypointsCPU.ptr<int>(OCTAVE_ROW);
			float* kp_size = keypointsCPU.ptr<float>(SIZE_ROW);
			float* kp_dir = keypointsCPU.ptr<float>(ANGLE_ROW);
			float* kp_hessian = keypointsCPU.ptr<float>(HESSIAN_ROW);
			int size = int(vKeyPointsCurr.size() > _gpu_keypoints_all_curr.cols ? _gpu_keypoints_all_curr.cols : vKeyPointsCurr.size());
			for (int i = 0; i < size; ++i)
			{
				const KeyPoint& kp = vKeyPointsCurr[i];
				kp_x[i] = kp.pt.x;
				kp_y[i] = kp.pt.y;
				kp_octave[i] = kp.octave;
				kp_size[i] = kp.size;
				kp_dir[i] = kp.angle;
				kp_hessian[i] = kp.response;
				kp_laplacian[i] = 1;
			}

			cudaMemcpy2D(_gpu_keypoints_all_curr.data, _gpu_keypoints_all_curr.step, keypointsCPU.data, keypointsCPU.step, keypointsCPU.cols * keypointsCPU.elemSize(), _gpu_keypoints_all_curr.rows, cudaMemcpyHostToDevice);
			//keypointsGPU.upload(keypointsCPU);
			_n_detected_features_curr = size;
			assert(cvmDescriptorCurr.cols == _gpu_descriptor_all_curr.cols && cvmDescriptorCurr.elemSize() == _gpu_descriptor_all_curr.elemSize());
			int row = _gpu_descriptor_all_curr.rows > cvmDescriptorCurr.rows ? cvmDescriptorCurr.rows : _gpu_descriptor_all_curr.rows;
			cudaMemcpy2D(_gpu_descriptor_all_curr.data, _gpu_descriptor_all_curr.step, cvmDescriptorCurr.data, cvmDescriptorCurr.step, cvmDescriptorCurr.cols * cvmDescriptorCurr.elemSize(), row, cudaMemcpyHostToDevice);
		}

		break;
	default:
		cout << ("feature type Error\n");
		break;
	}
	return;
}

void CKinFuTracker::sparsePoseEstimation(const Mat& pts_World_, const Mat& nls_World_, const Mat& mds_World_, 
												  const Mat& pts_Cam_, const Mat& nls_Cam_,  const Mat& mds_Cam_, 
												  const Mat& m2D_, const Mat& bv_, const Mat& wg_, Matrix3f* pR_, Vector3f* pT_, Mat* ptr_inliers_)
{
	const float thre_3d = .2f;
	const float thre_2d = .001f;
	const float thre_nl = 0.1f;
	const float vis_ang_thre = _visual_angle_threshold;

	if ( pts_World_.empty() ) return;

	int nRansacIter = _nRansacIterationasTracking;
	if (_nStage == Relocalisation_Only){
		nRansacIter = _nRansacIterationasRelocalisation;
	}

	MatrixXf X_w(3, pts_World_.cols);  memcpy(X_w.data(), pts_World_.data, 3*sizeof(float)* pts_World_.cols);
	MatrixXf N_w(3, pts_World_.cols);  memcpy(N_w.data(), nls_World_.data, 3*sizeof(float)* nls_World_.cols);
	MatrixXf X_c(3, pts_World_.cols);  memcpy(X_c.data(), pts_Cam_.data, 3*sizeof(float)* pts_Cam_.cols);
	MatrixXf N_c(3, pts_World_.cols);  memcpy(N_c.data(), nls_Cam_.data, 3*sizeof(float)* nls_Cam_.cols);
	MatrixXf MD_w(3, pts_World_.cols); memcpy(MD_w.data(), mds_World_.data, 3*sizeof(float)* mds_World_.cols);//NOTE that cols might be zero
	MatrixXf MD_c(3, pts_World_.cols); memcpy(MD_c.data(), mds_Cam_.data, 3*sizeof(float)* mds_Cam_.cols);
	MatrixXf bv_c(3, pts_World_.cols); memcpy(bv_c.data(), bv_.data, 3 * sizeof(float)* bv_.cols);
	Matrix<short, Dynamic,Dynamic> wg_c(pts_World_.cols,3); 
	memcpy(wg_c.col(0).data(), wg_.ptr<short>(0), sizeof(short) * wg_.cols);
	memcpy(wg_c.col(1).data(), wg_.ptr<short>(1), sizeof(short) * wg_.cols);
	memcpy(wg_c.col(2).data(), wg_.ptr<short>(2), sizeof(short) * wg_.cols);
	
	float fErrorBest, fS2; Vector2f EA; int nIt = nRansacIter; MatrixXf result(18, 1);
	switch( _nPoseEstimationMethod ){
	case AORansac:
		fErrorBest = btl::utility::aoRansac<float>(X_w, X_c, thre_3d, nRansacIter, &*pR_, &*pT_, &*ptr_inliers_, ARUN);
		break;
	case AONn2dRansac:
		EA = btl::utility::aoWithNormalWith2dConstraintRansac<float>(X_w, N_w, X_c, N_c, thre_3d, 0.56f, vis_ang_thre, nRansacIter, &*pR_, &*pT_, &*ptr_inliers_);
		break;
	case AONn2dRansac2:
		EA = btl::utility::aoWithNormaln2dConstraintRansac2<float>(X_w, N_w, X_c, N_c, thre_3d, 0.56f, vis_ang_thre, nRansacIter, &*pR_, &*pT_, &*ptr_inliers_);
		break;
	default:
		cout << ("Failure - unrecognized pose estimation problem.\n");
		break;
	}

	return;
}

void CKinFuTracker::initialPoseEstimation(CRGBDFrame::tp_ptr pCurFrame_, const vector<GpuMat>& gpu_descriptors_, const vector<GpuMat>& gpu_3d_key_points_, const vector<int> vTotal_, 
	vector<Eigen::Affine3f>* ptr_v_k_hypothese_poses ){
	ptr_v_k_hypothese_poses->clear();
	_v_selected_world.clear();
	_v_selected_curr.clear();
	_v_mds_selected_curr.clear();
	_v_selected_2d_curr.clear();
	_v_selected_bv_curr.clear();
	_v_selected_weights.clear();
	_v_selected_inliers.clear();
	_v_selected_inliers_reloc.clear();
	_v_nls_selected_curr.clear();
	_v_nls_selected_world.clear();
	_v_mds_selected_world.clear();

	_pose_feature_c_f_w.linear().setConstant(numeric_limits<float>::quiet_NaN()); _pose_feature_c_f_w.makeAffine();
	_pose_refined_c_f_w.linear().setConstant(numeric_limits<float>::quiet_NaN()); _pose_refined_c_f_w.makeAffine();

	double t = (double)getTickCount();
	if (_nMatchingMethod == IDF || _nMatchingMethod == GM || _nMatchingMethod == GM_GPU ){
		//extract features
		/////////////////////////////////////////////////////////////////
		extractFeatures(pCurFrame_, _nFeatureName);//extract features
		cout << _intrinsics(0).fx << endl;
		vector<int>  vCandidates = cuda_extract_key_points(  _intrinsics(0), _pCubicGrids->_fFeatureVoxelSizeM, _nFeatureScale,
			*pCurFrame_->_acvgmShrPtrPyrPts[0], *pCurFrame_->_acvgmShrPtrPyrNls[0], *pCurFrame_->_acvgmShrPtrPyrDepths[0], *pCurFrame_->_acvgmShrPtrPyrReliability[2],
																		_gpu_descriptor_all_curr, _gpu_keypoints_all_curr, _n_detected_features_curr,
																		&_gpu_descriptor_curr, &_gpu_keypoints_curr,
																		&_gpu_pts_curr, &_gpu_nls_curr, &_gpu_mds_curr, &_gpu_2d_curr, &_gpu_bv_curr, &_gpu_weights );
		
		t = ((double)getTickCount() - t) / getTickFrequency();
		cout << "extract_multiscale_3d_features_gpu()  in seconds: " << t << std::flush << endl;
		t = (double)getTickCount();
		 
		int nCandidates = std::accumulate( vCandidates.begin(), vCandidates.end(), 0);
		cout << "# of kp = " << nCandidates << endl;
		if (nCandidates < _nMinFeatures[_nResolution]){
			cout << ("Failure - not enough salient features in current frame\n");
			cout << (nCandidates) << endl;
			return;
		}

		//matching 
		/////////////////////////////////////////////////////////////////
		findCorrespondences(gpu_descriptors_, gpu_3d_key_points_, vTotal_, _K,
							&_v_selected_world, &_v_nls_selected_world, &_v_mds_selected_world,
							&_v_selected_curr, &_v_nls_selected_curr, &_v_mds_selected_curr,
							&_v_selected_2d_curr, &_v_selected_bv_curr, &_v_selected_weights, &_v_selected_inliers);

		assert(_v_selected_world.size() == _v_selected_curr.size() && _v_selected_world.size() == _v_selected_2d_curr.size() && _v_selected_2d_curr.size() == _v_selected_bv_curr.size() && _v_selected_world.size() == _v_selected_inliers.size() && _v_selected_world.size() == _v_selected_inliers_reloc.size());
		assert(_v_selected_world.size() == _v_selected_weights.size());
		t = ((double)getTickCount() - t) / getTickFrequency();
		cout << "findCorrepondences() in seconds: " << t << std::flush << endl;
		t = (double)getTickCount();

		//pose estimation from point correspondences
		/////////////////////////////////////////////////////////////////
		Mat inliers;
		for (int i = 0; i < _v_selected_world.size(); i++) { //fi use icp as a hyposethese v_refined_R.size() = 1; note that _v_selected_inliers, _v_selected_world and others have been modified by now 
			Mat& origin_inliers = _v_selected_inliers[i];
			Mat& reloc_inliers = _v_selected_inliers_reloc[i];
			if (origin_inliers.cols < 4) {
				Eigen::Affine3f m; m.linear().setZero(); m.makeAffine();
				ptr_v_k_hypothese_poses->push_back(m);
				continue;
			}
			Matrix3f Rw; Vector3f Tw;
			
			sparsePoseEstimation ( _v_selected_world[i], _v_nls_selected_world[i], _v_mds_selected_world[i],
											_v_selected_curr[i], _v_nls_selected_curr[i], _v_mds_selected_curr[i],
											_v_selected_2d_curr[i], _v_selected_bv_curr[i], _v_selected_weights[i], &Rw, &Tw, &inliers);

			//transform inlier idx from input of AO/RansacPnp to original idx
			if (!inliers.empty()){
				Mat selected_orig_inliers = inliers.clone();
				Mat selected_orig_inliers_reloc = inliers.clone();
				for (int c = 0; c < inliers.cols; c++){
					int idx = inliers.ptr<int>()[c];
					int idx_orig = origin_inliers.ptr<int>()[idx];
					selected_orig_inliers.ptr<int>()[c] = idx_orig;
					int idx_reloc = reloc_inliers.ptr<int>()[idx];
					selected_orig_inliers_reloc.ptr<int>()[c] = idx_reloc;
				}
				selected_orig_inliers.copyTo(origin_inliers);
				selected_orig_inliers_reloc.copyTo(reloc_inliers);
			}
			ptr_v_k_hypothese_poses->push_back(convertRnt2Prj(Rw, Tw));
		}

		t = ((double)getTickCount() - t) / getTickFrequency();
		cout << "pose estimation() in seconds: " << t << std::flush << endl;
		t = (double)getTickCount();
	}
	else{
		cout << (" Failure -- No effective initial matching method specified.\n");
	}
	return;
}

void CKinFuTracker::tracking( CRGBDFrame::tp_ptr pCurFrame_ )
{
	int nStage = _nStage;
	cout << ("track() starts.\n");
	//parameters
	_K = _K_tracking;
	_nStage = btl::Tracking_n_Mapping;

	_fPercantage = 0.6f;
	//initial pose estimation
	vector<Eigen::Affine3f> v_k_hypothese_poses;

	initialPoseEstimation(pCurFrame_, _gpu_descriptor_prev, _gpu_pts_prev, vector<int>(), &v_k_hypothese_poses);

	// pose refinement
	double t = (double)getTickCount();
	Matrix3f Rw; Vector3f Tw; 
	if (_pCubicGrids->verifyPoseHypothesesAndRefine(pCurFrame_, _pPrevFrameWorld.get(), v_k_hypothese_poses, _nICPMethod, _nStage, &Rw, &Tw, &_best_k) < _aMinICPEnergy[_nResolution]) {

		t = ((double)getTickCount() - t) / getTickFrequency();
		cout << "pose refinement() in seconds: " << t << endl;
		t = (double)getTickCount();

		//Mat pts;  pCurFrame_->_acvgmShrPtrPyrPts[0]->download(pts); cout << pts << endl;

		pCurFrame_->gpuTransformToWorld(); //transform from camera local into world reference
		if (!_bTrackingOnly /*&& pCurFrame_->isMovedwrtReferencInRadiusM( _pPrevFrameWorld.get(),M_PI_4/45.,0.01)*/) { //test if the current frame have been moving
			_pCubicGrids->integrateDepth(*pCurFrame_,_intrinsics); //integrate depth into global depth
			//insert features into feature-base
			if (_nMatchingMethod == IDF || _nMatchingMethod == GM || _nMatchingMethod == GM_GPU ) {
				_pose_feature_c_f_w = _pose_refined_c_f_w = btl::utility::convertRnt2Prj(Rw, Tw);
				_v_prj_c_f_w_training.push_back(_pose_refined_c_f_w);
				_total_refined_features = cuda_refine_inliers(_distance_threshold, cos(_normal_angle_threshold), cos(_visual_angle_threshold), _nAppearanceMatched,
													pcl::device::device_cast<matrix3_cmf>(Rw), pcl::device::device_cast<float3>(Tw),
													_gpu_pts_global_reloc, _gpu_nls_global_reloc, _gpu_pts_curr_reloc, _gpu_nls_curr_reloc, &_gpu_relined_inliers);
				if (_total_refined_features > 0){
					GpuMat matched_idx_previous(1, _n_detected_features_curr, CV_32SC2, _match_tmp.ptr(0));
					GpuMat matched_distance_previous(1, _n_detected_features_curr, CV_32FC2, _match_tmp.ptr(1));
					_pCubicGrids->integrateFeatures(_intrinsics, Rw, Tw, _nFeatureScale,
													_gpu_pts_curr_reloc, _gpu_nls_curr_reloc, _gpu_keypoint_curr_reloc, _gpu_descriptor_curr_reloc, matched_distance_previous, _nAppearanceMatched,
													_gpu_relined_inliers, _total_refined_features);
				}
			}
		}//if training

		_pose_feature_c_f_w = _pose_refined_c_f_w = btl::utility::convertRnt2Prj(Rw, Tw);
		_v_prj_c_f_w_training.push_back(_pose_refined_c_f_w);
		
		storeCurrFrame( pCurFrame_ );
		t = ((double)getTickCount() - t) / getTickFrequency();
		cout << "feature integration() in seconds: " << t << std::flush << endl;

	}//if current frame is lost aligned
	else{
		
		cout << ("Tracking fails.\n");
		//goto relocalisation
		cout << ("Relocalisation starts.\n");
		if (relocalise(pCurFrame_)){
			cout << ("Relocalisation succeeds.\n");
		}
		else{
			pCurFrame_->gpuTransformToWorld(); //transform from camera local into world reference
			cout << ("Relocalisation fails.\n");
		}
	}
	_nStage = nStage;

	return;
}//track

bool CKinFuTracker::relocalise(CRGBDFrame::tp_ptr pCurFrame_)
{
	int nStage = _nStage;
	_nStage = btl::Relocalisation_Only;
	//assertion
	if (_nMatchingMethod == IDF || _nMatchingMethod == GM || _nMatchingMethod == GM_GPU ){
		if (_pCubicGrids->totalFeatures() < 50) {
			cout << ("Failure - too less global features has been learnt before.\n");
			_nStage = nStage;
			return false; // if no effective depth or no features in the global feature set, nMatchedPairs will be 0. 
		}
	}

	//parameters
	if (_nMatchingMethod == ICP){
		_K = 0;
	}
	else{
		_K = _K_relocalisation;
	}

	_fPercantage = 0.8f;
	// initial pose estimation
	vector<Eigen::Affine3f> v_k_hypothese_poses;
	initialPoseEstimation(pCurFrame_, _pCubicGrids->_gpu_global_descriptors, _pCubicGrids->_gpu_global_3d_key_points, _pCubicGrids->_vTotalGlobal, &v_k_hypothese_poses);

	// pose refinement
	double t = (double)getTickCount();
	Matrix3f Rw; Vector3f Tw;
	bool is_relocalisation_success = false;
	if( _pCubicGrids->verifyPoseHypothesesAndRefine( pCurFrame_, _pPrevFrameWorld.get(),v_k_hypothese_poses, _nICPMethod, _nStage, &Rw, &Tw, &_best_k ) < _aMinICPEnergy[_nResolution] ) {
		//_pCubicGrids->calc_mask_for_icp_alignment(btl::Vol_2_Frm_ICP, pCurFrame_, _pPrevFrameWorld.get());
		//pCurFrame_->apply_mask();

		t = ((double)getTickCount() - t) / getTickFrequency();
		cout << "pose refinement() in seconds: " << t << std::flush << endl;
		t = (double)getTickCount();
		pCurFrame_->gpuTransformToWorld(); //transform from camera local into world reference

		//_pCubicGridsMoved->gpuIntegrateFrameIntoVolumeCVCV(*pCurFrame_, _intrinsics);
		_pose_refined_c_f_w = btl::utility::convertRnt2Prj(Rw, Tw);
		if (_nStage == btl::Tracking_n_Mapping || _nStage == btl::Mapping_Using_GT)
			_v_prj_c_f_w_training.push_back(_pose_refined_c_f_w);
		if (_nStage == btl::Relocalisation_Only|| _nStage == btl::Tracking_NonStation)
			_v_relocalisation_p_matrices.push_back(_pose_refined_c_f_w);

		is_relocalisation_success = true;
	}
	if (v_k_hypothese_poses.size()>0)
		_pose_feature_c_f_w = v_k_hypothese_poses[_best_k];
	else
		_pose_feature_c_f_w.matrix().fill(1000.f);

	_pose_refined_c_f_w = btl::utility::convertRnt2Prj(Rw, Tw);

	storeCurrFrame(pCurFrame_);

	_nStage = nStage;
	return is_relocalisation_success;
}

void CKinFuTracker::findCorrespondences( const vector<GpuMat>& gpu_descriptors_training_, const vector<GpuMat>& gpu_pts_training_, const vector<int> vTotal_, const int K_,
										   vector<Mat>* ptr_v_selected_world_,vector<Mat>* ptr_v_nls_selected_world_, vector<Mat>* ptr_v_mds_selected_world_,
										   vector<Mat>* ptr_v_selected_curr_, vector<Mat>* ptr_v_nls_selected_curr_,  vector<Mat>* ptr_v_mds_selected_curr_,
										   vector<Mat>* ptr_v_selected_2d_curr_, vector<Mat>* ptr_v_selected_bv_curr_, vector<Mat>* ptr_v_selected_weights_, vector<Mat>* ptr_v_selected_inliers_){
	int total = 0;
	for (int i = 0; i < _nFeatureScale; i++)
		total += _gpu_descriptor_curr[i].rows;
	assert( total> 0 );

	_fExpThreshold = 0.1f;

	//match
	_nAppearanceMatched = 0;
	for (int i=0; i<_nFeatureScale; i++) {
		if ( _gpu_descriptor_curr[i].empty() || _gpu_keypoints_curr[i].empty() || _gpu_nls_curr[i].empty() || _gpu_mds_curr[i].empty() || _gpu_pts_curr[i].empty() || _gpu_2d_curr[i].empty() ) continue; //if the level of 3d features are empty then skip the matching 

		int nKeyPointsPrev;
		if(gpu_pts_training_[i].rows == _nGlobalKeyPointDimension ){//relocalisation w.r.t. global map
			nKeyPointsPrev = vTotal_[i];
			if( nKeyPointsPrev == 0 ) continue; 

			_pMatcherGpuHamming->knnMatchAsync(_gpu_descriptor_curr[i], gpu_descriptors_training_[i].rowRange(0, nKeyPointsPrev), _match_tmp, 2);
			GpuMat matched_idx_previous(1, _gpu_descriptor_curr[i].rows, CV_32SC2, _match_tmp.ptr(0));
			GpuMat matched_distance_previous(1, _gpu_descriptor_curr[i].rows, CV_32FC2, _match_tmp.ptr(1));

			//_pMatcherGpuHamming->knnMatchSingle(_gpu_descriptor_curr[i], gpu_descriptors_training_[i].rowRange(0, nKeyPointsPrev), matched_idx_previous, matched_distance_previous, GpuMat(), 2);
			_nAppearanceMatched = btl::device::cuda_collect_point_pairs_2nn(512.f, _fPercantage, _nMatchingMethod, _nAppearanceMatched, nKeyPointsPrev, matched_idx_previous, matched_distance_previous, 
																		gpu_pts_training_[i],  _gpu_pts_curr[i], 
																		_gpu_2d_curr[i], _gpu_bv_curr[i], _gpu_weights[i], _gpu_keypoints_curr[i], _gpu_descriptor_curr[i],
																		_gpu_nls_prev[i], _gpu_mds_prev[i], _gpu_nls_curr[i], _gpu_mds_curr[i],
																		&_gpu_pts_global_reloc, &_gpu_pts_curr_reloc, 
																		&_gpu_nls_global_reloc, &_gpu_nls_curr_reloc, 
																		&_gpu_mds_global_reloc, &_gpu_mds_curr_reloc, 
																		&_gpu_2d_curr_reloc, &_gpu_bv_curr_reloc, &_gpu_weight_reloc, &_gpu_keypoint_curr_reloc, &_gpu_descriptor_curr_reloc,
																		&_gpu_idx_curr_2_prev_reloc, &_gpu_hamming_distance_reloc, &_gpu_distinctive_reloc );
		}
		else{//tracking and mapping
			nKeyPointsPrev = gpu_descriptors_training_[i].rows;
                        if (nKeyPointsPrev == 0) { continue; }
			_pMatcherGpuHamming->knnMatchAsync(_gpu_descriptor_curr[i], gpu_descriptors_training_[i], _match_tmp, 2);
			GpuMat matched_idx_previous(1, _gpu_descriptor_curr[i].rows, CV_32SC2, _match_tmp.ptr(0));
			GpuMat matched_distance_previous(1, _gpu_descriptor_curr[i].rows, CV_32FC2, _match_tmp.ptr(1));

			//collect points, Note that threshold used by jose and gm or in tracking or re-localization are different 
			// the # of collected matches may be more than # of input features for GM based matching, because all ambiguous matches are collected as well.
			_nAppearanceMatched = btl::device::cuda_collect_point_pairs_2nn(512.f, _fPercantage, _nMatchingMethod, _nAppearanceMatched, nKeyPointsPrev, matched_idx_previous, matched_distance_previous, 
																		gpu_pts_training_[i],  _gpu_pts_curr[i], 
																		_gpu_2d_curr[i], _gpu_bv_curr[i], _gpu_weights[i], _gpu_keypoints_curr[i], _gpu_descriptor_curr[i],
																		_gpu_nls_prev[i], _gpu_mds_prev[i], _gpu_nls_curr[i], _gpu_mds_curr[i],
																		&_gpu_pts_global_reloc, &_gpu_pts_curr_reloc, 
																		&_gpu_nls_global_reloc, &_gpu_nls_curr_reloc, 
																		&_gpu_mds_global_reloc, &_gpu_mds_curr_reloc, 
																		&_gpu_2d_curr_reloc, &_gpu_bv_curr_reloc, &_gpu_weight_reloc, &_gpu_keypoint_curr_reloc, &_gpu_descriptor_curr_reloc,
																		&_gpu_idx_curr_2_prev_reloc, &_gpu_hamming_distance_reloc, &_gpu_distinctive_reloc );
		}
	}
	
	if(_nAppearanceMatched == 0 ) {
		cout << ("Failure - There is no selected features in current frame.\n");
		ptr_v_selected_inliers_->clear(); // 
		_v_selected_inliers_reloc.clear();
		return;
	}
	Mat distance;	_match_tmp.row(1).download(distance);

	//try from the best match to these end for anchor point. 
	//note that the order of the following carry the same # of data. 
	//			1. _gpu_pts_global_reloc,		->_pts_global_reloc
	//			2. _gpu_nls_global_reloc,		->_nls_global_reloc
	//			3. _gpu_pts_curr_reloc,			->_pts_curr_reloc
	//			4. _gpu_nls_curr_reloc,			->_nls_curr_reloc
	//			5. _gpu_2d_curr_reloc,			->_2d_curr_reloc
	//			6. _gpu_keypoint_curr_reloc,	
	//			7. _gpu_descriptor_curr_reloc
	//			8. _gpu_idx_curr_2_prev_reloc   ->c_2_p_pairs
	//			9. _gpu_hamming_distance_reloc  ->hamming_distance_reloc

	//note that _gpu_descriptor_curr_reloc and _cvgmC2PPairs are different in the sense of, though store int2 holding the index of 1st NN and 2nd NN
	//			1. _cvgmC2PPairs holds all pairs of correspondences
	//			2. _gpu_descriptor_curr_reloc is selected out of _cvgmC2PPairs
	//          3. _gpu_descriptor_curr_reloc stores only current idx and 1 previous while _cvgmC2PPairs shores both 1st and 2nd NN
	//			4. the order _cvgmC2PPairs of is identical to _cvgmDistance, _gpu_descriptor_curr and _gpu_key_points_curr; _gpu_descriptor_curr_reloc is not

	//note that only first _nAppearanceMatched # of data is effective

	//Mat c_2_p_pairs_all;  _cvgmC2PPairs.download(c_2_p_pairs_all); //all 1st and 2nd matches before selection


	//credability matrix, i.e., the adjacency matrix, holds the credability possibility [0. - 1.], the higher the score the more confidence
	// that the pair is a correct match.
	vector<Mat> v_inliers;
	switch (_nMatchingMethod)
	{
	case GM:
	{
		// read out hamming distance from distance
		Mat hamming_distance_reloc; _gpu_hamming_distance_reloc.colRange(0, _nAppearanceMatched).download(hamming_distance_reloc);//float2 format
		Mat credability;
		_fWeight = 1.f;
		credability.create(_nAppearanceMatched, _nAppearanceMatched, CV_32FC1);	credability.setTo(0.f);
		for (int P_idx = 0; P_idx < _nAppearanceMatched; P_idx++) {
			//get hamming distance
			float hamming_distance = hamming_distance_reloc.ptr<float>()[P_idx];
			//set diagonal elements of the AM
			credability.at<float>(P_idx, P_idx) = calcSimilarity(hamming_distance); //convert hamming distance to 1.0 credability score
		}
		GpuMat gpu_credability;
		gpu_credability.upload(credability);

		GpuMat gpu_hamming_distance = _gpu_hamming_distance_reloc.colRange(0, _nAppearanceMatched);
		GpuMat gpu_pts_global = _gpu_pts_global_reloc.colRange(0, _nAppearanceMatched);
		GpuMat gpu_pts_curr = _gpu_pts_curr_reloc.colRange(0, _nAppearanceMatched);
		GpuMat gpu_nls_global = _gpu_nls_global_reloc.colRange(0, _nAppearanceMatched);
		GpuMat gpu_nls_curr = _gpu_nls_curr_reloc.colRange(0, _nAppearanceMatched);
		GpuMat gpu_2d_curr = _gpu_2d_curr_reloc.colRange(0, _nAppearanceMatched);
		GpuMat gpu_bv_curr = _gpu_bv_curr_reloc.colRange(0, _nAppearanceMatched);
		GpuMat gpu_w_curr = _gpu_weight_reloc.colRange(0, _nAppearanceMatched);
		
		gpuExtractCorrespondencesMaxScore_binary(gpu_hamming_distance, gpu_pts_global, gpu_pts_curr, gpu_nls_global, gpu_nls_curr, gpu_2d_curr, gpu_bv_curr, gpu_w_curr,
												K_, gpu_credability, 
												&*ptr_v_selected_world_, &*ptr_v_nls_selected_world_, 
												&*ptr_v_selected_curr_, &*ptr_v_nls_selected_curr_, &*ptr_v_selected_2d_curr_, &*ptr_v_selected_bv_curr_, &*ptr_v_selected_weights_, &v_inliers);
		for (int i = 0; i < ptr_v_nls_selected_world_->size(); i++)
		{
			ptr_v_mds_selected_world_->push_back(Mat());
			ptr_v_mds_selected_curr_->push_back(Mat());
		}
	}	
		break;
	case GM_GPU:
	{
		// read out hamming distance from distance
		Mat hamming_distance_reloc; _gpu_hamming_distance_reloc.colRange(0, _nAppearanceMatched).download(hamming_distance_reloc);//float2 format
		_fWeight = 1.f;
		Mat credability;
		credability.create(_nAppearanceMatched, _nAppearanceMatched, CV_32FC1);	credability.setTo(0.f);
		for (int P_idx = 0; P_idx < _nAppearanceMatched; P_idx++) {
			//get hamming distance
			float hamming_distance = hamming_distance_reloc.ptr<float>()[P_idx];
			//set diagonal elements of the AM
			credability.at<float>(P_idx, P_idx) = calcSimilarity(hamming_distance); //convert hamming distance to 1.0 credability score
		}
		GpuMat gpu_credability;
		gpu_credability.upload(credability);

		GpuMat gpu_hamming_distance = _gpu_hamming_distance_reloc.colRange(0, _nAppearanceMatched);
		GpuMat gpu_pts_global = _gpu_pts_global_reloc.colRange(0, _nAppearanceMatched);
		GpuMat gpu_pts_curr = _gpu_pts_curr_reloc.colRange(0, _nAppearanceMatched);
		GpuMat gpu_nls_global = _gpu_nls_global_reloc.colRange(0, _nAppearanceMatched);
		GpuMat gpu_nls_curr = _gpu_nls_curr_reloc.colRange(0, _nAppearanceMatched);
		GpuMat gpu_2d_curr = _gpu_2d_curr_reloc.colRange(0, _nAppearanceMatched);
		GpuMat gpu_bv_curr = _gpu_bv_curr_reloc.colRange(0, _nAppearanceMatched);
		GpuMat gpu_w_curr = _gpu_weight_reloc.colRange(0, _nAppearanceMatched);

		gpuExtractCorrespondencesMaxScore ( gpu_hamming_distance, 
											gpu_pts_global, 
											gpu_pts_curr, 
											gpu_nls_global, 
											gpu_nls_curr, 
											gpu_2d_curr,
											gpu_bv_curr,
											gpu_w_curr,
											K_,
											gpu_credability, 
											&*ptr_v_selected_world_, 
											&*ptr_v_nls_selected_world_, 
											&*ptr_v_selected_curr_, 
											&*ptr_v_nls_selected_curr_, 
											&*ptr_v_selected_2d_curr_, 
											&*ptr_v_selected_bv_curr_,
											&*ptr_v_selected_weights_,
											&v_inliers);
		for (int i = 0; i < ptr_v_nls_selected_world_->size(); i++)
		{
			ptr_v_mds_selected_world_->push_back(Mat());
			ptr_v_mds_selected_curr_->push_back(Mat());
		}
	}
		break;
	case IDF:
	case BUILD_GT:		
	{
		Mat pts_global_reloc; _gpu_pts_global_reloc.colRange(0, _nAppearanceMatched).download(pts_global_reloc);
		Mat nls_global_reloc; _gpu_nls_global_reloc.colRange(0, _nAppearanceMatched).download(nls_global_reloc);
		Mat mds_global_reloc; _gpu_mds_global_reloc.colRange(0, _nAppearanceMatched).download(mds_global_reloc);
		Mat pts_curr_reloc;   _gpu_pts_curr_reloc.colRange(0, _nAppearanceMatched).download(pts_curr_reloc); // a row of 3D coordinates
		Mat nls_curr_reloc;   _gpu_nls_curr_reloc.colRange(0, _nAppearanceMatched).download(nls_curr_reloc);
		Mat mds_curr_reloc;   _gpu_mds_curr_reloc.colRange(0, _nAppearanceMatched).download(mds_curr_reloc);
		Mat kp_2d_curr_reloc; _gpu_2d_curr_reloc.colRange(0, _nAppearanceMatched).download(kp_2d_curr_reloc);//float2 format
		Mat bv_curr_reloc;    _gpu_bv_curr_reloc.colRange(0, _nAppearanceMatched).download(bv_curr_reloc);//float2 format
		Mat w_curr_reloc;    _gpu_weight_reloc.colRange(0, _nAppearanceMatched).download(w_curr_reloc);//float2 format
		Mat key_pt_curr_reloc; _gpu_keypoint_curr_reloc.colRange(0, _nAppearanceMatched).download(key_pt_curr_reloc);

		Mat selected_inliers;
		selected_inliers.create(1, _nAppearanceMatched, CV_32SC1);
		//all 1st matches are selected as inliers
		for (int i = 0; i < _nAppearanceMatched; i++){
			selected_inliers.ptr<int>()[i] = i;
		}
		v_inliers.push_back(selected_inliers);

		ptr_v_selected_world_->push_back(pts_global_reloc);
		ptr_v_nls_selected_world_->push_back(nls_global_reloc);
		ptr_v_mds_selected_world_->push_back(mds_global_reloc);
		ptr_v_selected_curr_->push_back(pts_curr_reloc);
		ptr_v_nls_selected_curr_->push_back(nls_curr_reloc);
		ptr_v_mds_selected_curr_->push_back(mds_curr_reloc);
		ptr_v_selected_2d_curr_->push_back(kp_2d_curr_reloc);
		ptr_v_selected_bv_curr_->push_back(bv_curr_reloc);
		ptr_v_selected_weights_->push_back(w_curr_reloc);

	}
		break;
	
	default:
		break;
	}

	//transform inlier idx to original idx and reloc idx
	//ptr_v_selected_inliers_ is clear at the beginning of tracking or re-localization
	// and donot need to be cleared here
	Mat c_2_p_pairs;	        _gpu_idx_curr_2_prev_reloc.colRange(0, _nAppearanceMatched).download(c_2_p_pairs);
	for (int i = 0; i < v_inliers.size(); i++){ //note that the length of v_selected_inliers is not always equal to K_, it is always 1 for IDF.
		Mat& inliers = v_inliers[i];
		Mat selected_inlier_orig = inliers.clone();
		Mat selected_inlier_reloc = inliers.clone();
		for (int c=0; c< inliers.cols; c++){
			int idx = inliers.ptr<int>()[c];
			int idx_orig = c_2_p_pairs.ptr<int2>()[idx].x;
			selected_inlier_orig.ptr<int>()[c] = idx_orig;
		}
		ptr_v_selected_inliers_->push_back( selected_inlier_orig ); // 
		_v_selected_inliers_reloc.push_back( selected_inlier_reloc );
	}
	return;
}

void CKinFuTracker::gpuExtractCorrespondencesMaxScore(const GpuMat& hamming_distance_,
	const GpuMat& gpu_global_pts_, const GpuMat& gpu_curr_pts_,
	const GpuMat& gpu_global_nls_, const GpuMat& gpu_curr_nls_, const GpuMat& gpu_curr_2d_, const GpuMat& gpu_curr_bv_, const GpuMat& gpu_curr_wg_,
	const int K_,
	GpuMat& gpu_credibility_, 
	vector<Mat>* ptr_v_selected_world_, vector<Mat>* ptr_v_nls_selected_world_, 
	vector<Mat>* ptr_v_selected_curr_, vector<Mat>* ptr_v_nls_selected_curr_, vector<Mat>* ptr_v_selected_2d_curr_, vector<Mat>* ptr_v_selected_bv_curr_, vector<Mat>* ptr_v_selected_wg_curr_,
	vector<Mat>* ptr_v_selected_inliers_)
{

	ptr_v_selected_world_->clear();
	ptr_v_nls_selected_world_->clear();
	ptr_v_selected_2d_curr_->clear();
	ptr_v_nls_selected_curr_->clear();
	ptr_v_selected_2d_curr_->clear();
	ptr_v_selected_bv_curr_->clear();
	ptr_v_selected_wg_curr_->clear();
	ptr_v_selected_inliers_->clear();
	//note that all input matrix and credability matrix should has the same magnitude.

	//1. Calc the adjacency matrix i.e. credability matrix
	/////////////////////////////////////////////////////////////////////////////////
	//Note that calc of the credibility matrix should be made parallelized

	btl::device::cuda_calc_adjacency_mt(gpu_global_pts_, gpu_curr_pts_, gpu_global_nls_, gpu_curr_nls_, gpu_curr_2d_, &gpu_credibility_);

	Mat credibility_; gpu_credibility_.download(credibility_);
	//2. harvest matches
	/////////////////////////////////////////////////////////////////////////////////
	Mat column_scores;
	calcScores(credibility_, &column_scores);
	Mat columIdx;
	sortIdx(column_scores, columIdx, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING); //

	const int ChainLength = _aMaxChainLength[_nResolution] * 10; //max correct chain out of candidate matches, inlier ratio times total # of matches
	const int TotalTrials = 10; //each trail may run par

	Mat node_numbers(1, TotalTrials, CV_32SC1);

	vector<Mat> v_selected_world, v_nls_selected_world;
	vector<Mat> v_selected_curr, v_nls_selected_curr;
	vector<Mat> v_selected_2d_curr;
	vector<Mat> v_selected_bv_curr;
	vector<Mat> v_selected_wg_curr;
	vector<Mat> v_selected_inliers;

	Mat selected_idx(1, ChainLength, CV_32SC1);

	Mat global_pts_; gpu_global_pts_.download(global_pts_);
	Mat global_nls_; gpu_global_nls_.download(global_nls_);
	Mat curr_pts_; gpu_curr_pts_.download(curr_pts_);
	Mat curr_nls_; gpu_curr_nls_.download(curr_nls_);
	Mat curr_2d_; gpu_curr_2d_.download(curr_2d_);
	Mat curr_bv_; gpu_curr_bv_.download(curr_bv_);
	Mat curr_wg_; gpu_curr_wg_.download(curr_wg_);

	_fExpThreshold = 0.05f;
	int nNode = 0;
	for (int nTrial = 0; nTrial < TotalTrials; nTrial++){
		int idx_0; //the idx of an anchor point
		nNode = 0;
		for (int i = nTrial; i < columIdx.cols; i++)
		// here we traverse all the pairs of correspondences from low hamming distance to high
		{
			int idx = columIdx.ptr<int>()[i]; //from highest to low
			//float hd = hamming_distance_.ptr<float>()[idx];//for debug

			if (nNode == 0) {
				assert(i == nTrial);
				selected_idx.ptr<int>()[nNode] = idx;
				idx_0 = idx;
				nNode++;
			}
			else {
				// calc the pair-wise distance	
				float fC = credibility_.ptr<float>(idx_0)[idx];
				if (fC > _fExpThreshold) {
					// store the point pair
					selected_idx.ptr<int>()[nNode] = idx;
					nNode++;
				}
			}

			if (nNode >= ChainLength) { break; }
		}//for
		//above code will produce selected_idx and nNode.
		//nNode is the number of selected feature pairs
		//selected_idx[0] stores the column selected, the column corresponds to a feature pair
		//from[1] to [nNode-1] stores the feature pairs consistent with the correspondig column.
		if (nNode < 4){
			v_selected_world.push_back(Mat());
			v_nls_selected_world.push_back(Mat());
			v_selected_curr.push_back(Mat());
			v_nls_selected_curr.push_back(Mat());
			v_selected_2d_curr.push_back(Mat());
			v_selected_bv_curr.push_back(Mat());
			v_selected_wg_curr.push_back(Mat());
			v_selected_inliers.push_back(Mat());
			node_numbers.at<int>(0, nTrial) = nNode;
		}
		else{
			//3. apply 1-to-1 constraint here
			/////////////////////////////////////////////////////////////////////////////////

			Mat one_2_one_idx = apply1to1Contraint(credibility_, column_scores, selected_idx.colRange(0, nNode));
			//collect matched pairs
			Mat selected_world(one_2_one_idx.size(), CV_32FC3), nls_selected_world(one_2_one_idx.size(), CV_32FC3),
				selected_curr(one_2_one_idx.size(), CV_32FC3), nls_selected_curr(one_2_one_idx.size(), CV_32FC3), 
				selected_2d_curr(one_2_one_idx.size(), CV_32FC2), selected_bv_curr(one_2_one_idx.size(), CV_32FC3), selected_wg_curr(3, one_2_one_idx.cols, CV_16SC1);

			for (int i = 0; i < one_2_one_idx.cols; i++){
				int idx = one_2_one_idx.ptr<int>()[i];
				selected_world.ptr<float3>()[i] = global_pts_.ptr<float3>()[idx];//get world 3d pts
				nls_selected_world.ptr<float3>()[i] = global_nls_.ptr<float3>()[idx];//get world nls
				selected_curr.ptr<float3>()[i] = curr_pts_.ptr<float3>()[idx];//get current 3d pts
				nls_selected_curr.ptr<float3>()[i] = curr_nls_.ptr<float3>()[idx];//get current nls
				selected_2d_curr.ptr<float2>()[i] = curr_2d_.ptr<float2>()[idx];//get 2d key point
				selected_bv_curr.ptr<float3>()[i] = curr_bv_.ptr<float3>()[idx];//get 2d key point
				selected_wg_curr.ptr<short>(0)[i] = curr_wg_.ptr<short>(0)[idx];//get 2d key point
				selected_wg_curr.ptr<short>(1)[i] = curr_wg_.ptr<short>(1)[idx];//get 2d key point
				selected_wg_curr.ptr<short>(2)[i] = curr_wg_.ptr<short>(2)[idx];//get 2d key point
			}

			v_selected_world.push_back(selected_world);
			v_nls_selected_world.push_back(nls_selected_world);
			v_selected_curr.push_back(selected_curr);
			v_nls_selected_curr.push_back(nls_selected_curr);
			v_selected_2d_curr.push_back(selected_2d_curr);
			v_selected_bv_curr.push_back(selected_bv_curr);
			v_selected_wg_curr.push_back(selected_wg_curr);
			v_selected_inliers.push_back(one_2_one_idx);
			node_numbers.at<int>(0, nTrial) = one_2_one_idx.cols;
		}
	}//several trial
	//4. chose the longest matches sequence
	/////////////////////////////////////////////////////////////////////////////////

	Mat trail_idx;
	sortIdx(node_numbers, trail_idx, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING); // rank the trails, the one with the most nodes rank higher
	for (int i = 0; i < K_; i++) {
		int nMaxIdx = trail_idx.at<int>(0, i);
		ptr_v_selected_world_->push_back(v_selected_world[nMaxIdx]);
		ptr_v_nls_selected_world_->push_back(v_nls_selected_world[nMaxIdx]);
		ptr_v_selected_curr_->push_back(v_selected_curr[nMaxIdx]);
		ptr_v_nls_selected_curr_->push_back(v_nls_selected_curr[nMaxIdx]);
		ptr_v_selected_2d_curr_->push_back(v_selected_2d_curr[nMaxIdx]);
		ptr_v_selected_bv_curr_->push_back(v_selected_bv_curr[nMaxIdx]);
		ptr_v_selected_wg_curr_->push_back(v_selected_wg_curr[nMaxIdx]);
		ptr_v_selected_inliers_->push_back(v_selected_inliers[nMaxIdx]);
	}

	return;
}

void CKinFuTracker::gpuExtractCorrespondencesMaxScore_binary(const GpuMat& hamming_distance_,
	const GpuMat& gpu_global_pts_, const GpuMat& gpu_curr_pts_,
	const GpuMat& gpu_global_nls_, const GpuMat& gpu_curr_nls_, const GpuMat& gpu_curr_2d_, const GpuMat& gpu_curr_bv_, const GpuMat& gpu_curr_wg_,
	const int K_,
	GpuMat& gpu_credibility_, 
	vector<Mat>* ptr_v_selected_world_, vector<Mat>* ptr_v_nls_selected_world_, 
	vector<Mat>* ptr_v_selected_curr_, vector<Mat>* ptr_v_nls_selected_curr_, vector<Mat>* ptr_v_selected_2d_curr_, vector<Mat>* ptr_v_selected_bv_curr_, vector<Mat>* ptr_v_selected_wg_curr_,
	vector<Mat>* ptr_v_selected_inliers_)
{
	ptr_v_selected_world_->clear();
	ptr_v_nls_selected_world_->clear();
	ptr_v_selected_2d_curr_->clear();
	ptr_v_nls_selected_curr_->clear();
	ptr_v_selected_2d_curr_->clear();
	ptr_v_selected_wg_curr_->clear();
	ptr_v_selected_inliers_->clear();
	//note that all input matrix and credability matrix should has the same magnitude.

	//1. Calc the adjacency matrix i.e. credability matrix
	/////////////////////////////////////////////////////////////////////////////////
	GpuMat gpu_geometry; gpu_geometry.create(gpu_credibility_.size(), CV_32FC4);
	btl::device::cuda_calc_adjacency_mt_binary(gpu_global_pts_, gpu_curr_pts_, gpu_global_nls_, gpu_curr_nls_, gpu_curr_2d_,  &gpu_credibility_, &gpu_geometry);

	//2. harvest matches
	/////////////////////////////////////////////////////////////////////////////////
	GpuMat gpu_column_scores;
	cv::cuda::reduce(gpu_credibility_, gpu_column_scores, 0, CV_REDUCE_SUM, CV_32F);
	GpuMat gpu_column_idx; btl::device::cuda_sort_column_idx(&gpu_column_scores, &gpu_column_idx); //NOTE that gpu_column_idx is float type due to the limitation of thrust.

	const int ChainLength = _aMaxChainLength[_nResolution] * 10; //max correct chain out of candidate matches, inlier ratio times total # of matches
	const int TotalTrials = gpu_credibility_.cols/2; //each trail may run par

	vector<Mat> v_selected_world, v_nls_selected_world;
	vector<Mat> v_selected_curr, v_nls_selected_curr;
	vector<Mat> v_selected_2d_curr;
	vector<Mat> v_selected_bv_curr;
	vector<Mat> v_selected_wg_curr;
	vector<Mat> v_selected_inliers;

	Mat node_numbers(TotalTrials, 1, CV_32SC1);
	Mat selected_idx(ChainLength, 1, CV_32SC1);

	Mat global_pts_; gpu_global_pts_.download(global_pts_);
	Mat global_nls_; gpu_global_nls_.download(global_nls_);
	Mat curr_pts_; gpu_curr_pts_.download(curr_pts_);
	Mat curr_nls_; gpu_curr_nls_.download(curr_nls_);
	Mat curr_2d_; gpu_curr_2d_.download(curr_2d_);
	Mat curr_bv_; gpu_curr_bv_.download(curr_bv_);
	Mat curr_wg_; gpu_curr_wg_.download(curr_wg_);

	_fExpThreshold = 0.05f;

	GpuMat inliers, gpu_node_numbers;  btl::device::cuda_select_inliers_from_am(gpu_credibility_, gpu_column_idx, _fExpThreshold, TotalTrials, ChainLength, &inliers, &gpu_node_numbers);
	//GpuMat inliers, gpu_node_numbers;  btl::device::cuda_select_inliers_from_am_2(gpu_credibility_, gpu_column_idx, _fExpThreshold, TotalTrials, ChainLength, &inliers, &gpu_node_numbers);
	gpu_node_numbers.download(node_numbers);
	for (int nTrial = 0; nTrial < TotalTrials; nTrial++){
		int nSelected = *node_numbers.ptr<int>(nTrial);
		nSelected = nSelected > ChainLength ? ChainLength : nSelected;
		if (nSelected < 4){
			v_selected_world.push_back(Mat());
			v_nls_selected_world.push_back(Mat());
			v_selected_curr.push_back(Mat());
			v_nls_selected_curr.push_back(Mat());
			v_selected_2d_curr.push_back(Mat());
			v_selected_bv_curr.push_back(Mat());
			v_selected_wg_curr.push_back(Mat());
			v_selected_inliers.push_back(Mat());
			continue;
		}

		Mat one_2_one_idx;  btl::device::cuda_apply_1to1_constraint_binary(gpu_credibility_, gpu_geometry, inliers.row(nTrial).colRange(0,nSelected), &one_2_one_idx);

		//collect matched pairs
		Mat selected_world(1, one_2_one_idx.rows, CV_32FC3), nls_selected_world(1, one_2_one_idx.rows, CV_32FC3),
			selected_curr(1, one_2_one_idx.rows, CV_32FC3), nls_selected_curr(1, one_2_one_idx.rows, CV_32FC3), 
			selected_2d_curr(1, one_2_one_idx.rows, CV_32FC2), selected_bv_curr(1, one_2_one_idx.rows, CV_32FC3), selected_wg_curr(3, one_2_one_idx.rows, CV_16SC1);

		for (int i = 0; i < one_2_one_idx.rows; i++){
			int idx = *one_2_one_idx.ptr<int>(i);
			selected_world.ptr<float3>()[i] = global_pts_.ptr<float3>()[idx];//get world 3d pts
			nls_selected_world.ptr<float3>()[i] = global_nls_.ptr<float3>()[idx];//get world nls
			selected_curr.ptr<float3>()[i] = curr_pts_.ptr<float3>()[idx];//get current 3d pts
			nls_selected_curr.ptr<float3>()[i] = curr_nls_.ptr<float3>()[idx];//get current nls
			selected_2d_curr.ptr<float2>()[i] = curr_2d_.ptr<float2>()[idx];//get 2d key point
			selected_bv_curr.ptr<float3>()[i] = curr_bv_.ptr<float3>()[idx];//get 2d key point
			selected_wg_curr.ptr<short>(0)[i] = curr_wg_.ptr<short>(0)[idx];//get weight
			selected_wg_curr.ptr<short>(1)[i] = curr_wg_.ptr<short>(1)[idx];
			selected_wg_curr.ptr<short>(2)[i] = curr_wg_.ptr<short>(2)[idx];
		}

		v_selected_world.push_back(selected_world);
		v_nls_selected_world.push_back(nls_selected_world);
		v_selected_curr.push_back(selected_curr);
		v_nls_selected_curr.push_back(nls_selected_curr);
		v_selected_2d_curr.push_back(selected_2d_curr);
		v_selected_bv_curr.push_back(selected_bv_curr);
		v_selected_wg_curr.push_back(selected_wg_curr);
		v_selected_inliers.push_back(one_2_one_idx.t());
		node_numbers.at<int>(nTrial, 0) = one_2_one_idx.rows;
	}

	//4. chose the longest matches sequence
	/////////////////////////////////////////////////////////////////////////////////
	Mat trail_idx;	sortIdx(node_numbers, trail_idx, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING); // rank the trails, the one with the most nodes rank higher
	for (int i = 0; i < K_; i++) {
		int nMaxIdx = trail_idx.at<int>(i, 0);
		ptr_v_selected_world_->push_back(v_selected_world[nMaxIdx]);
		ptr_v_nls_selected_world_->push_back(v_nls_selected_world[nMaxIdx]);
		ptr_v_selected_curr_->push_back(v_selected_curr[nMaxIdx]);
		ptr_v_nls_selected_curr_->push_back(v_nls_selected_curr[nMaxIdx]);
		ptr_v_selected_2d_curr_->push_back(v_selected_2d_curr[nMaxIdx]);
		ptr_v_selected_bv_curr_->push_back(v_selected_bv_curr[nMaxIdx]);
		ptr_v_selected_wg_curr_->push_back(v_selected_wg_curr[nMaxIdx]);
		ptr_v_selected_inliers_->push_back(v_selected_inliers[nMaxIdx]);
	}

	return;
}

Mat CKinFuTracker::apply1to1Contraint( const Mat& credibility_, const Mat& column_scores_, const Mat& selected_idx_){
	Mat one_2_one(selected_idx_.size(),CV_32SC1);
	int anchor_idx = selected_idx_.ptr<int>()[0];
	int nNode = 0;
	//assume anchor_idx is alway correct matches
	one_2_one.ptr<int>()[nNode++] = anchor_idx; 

	//need a new gpu code to apply 1-to-1 constraints

	for (int i = 1; i < selected_idx_.cols; i++) {
		int second_idx = selected_idx_.ptr<int>()[i];
		//check all other points to determine whether second_idx is engaged to M-to-1
		int j = i;
		for (; j < selected_idx_.cols; j++){
			int third_idx = selected_idx_.ptr<int>()[j];
			if ( i!=j && ( credibility_.at<float>(second_idx, third_idx) < 0.005f || credibility_.at<float>(third_idx, second_idx) )< 0.005f){ //indicate M-to-1 matches of i and j
				if (credibility_.at<float>(anchor_idx, third_idx) > credibility_.at<float>(anchor_idx, second_idx)){ // third is a better than second
					break;
				}
			}
		}
		if (j >= selected_idx_.cols) {//only if second_idx is the best match for the anchor point out of all other M-to-1 matches, it will be kept.
			one_2_one.ptr<int>()[nNode++] = second_idx;
		}
	}

	return one_2_one.colRange(0,nNode);
} 

void CKinFuTracker::calcScores(const Mat& credibility_, Mat* ptr_count_) const{

	cv::reduce(credibility_, *ptr_count_, 0, CV_REDUCE_SUM, CV_32F);
	return;
}

void CKinFuTracker::displayFeatures2D(int lvl_, CRGBDFrame::tp_ptr pCurFrame_) const{
		//if (_nMatchingMethod == FERNS || _nMatchingMethod == GEE) return;
		//if( lvl_>= _nFeatureScale || lvl_ < 0 ) return;
		//if( _gpu_keypoints_curr[lvl_].empty() ) return;
		//Mat keypoint; _gpu_keypoints_curr[lvl_].download(keypoint);
		//for(int i=0; i< keypoint.cols; i+=5 ){
		//	Point2f cen(keypoint.ptr<float>(0)[i],keypoint.ptr<float>(1)[i]);
		//	float radius = keypoint.ptr<float>(8)[i];
		//	if (radius < 20 ) continue;
		//	float angle = keypoint.ptr<float>(7)[i];//radian
		//	angle = angle / 180.f * float(M_PI);
		//	Point2f pt2 = cen + radius * Point2f(cos(angle),sin(angle));
		//	circle(*pCurFrame_->_acvmShrPtrPyrRGBs[0], cen, int(radius + .5f), Scalar(0, 0, 255), 1, LINE_4, 0);
		//	line(*pCurFrame_->_acvmShrPtrPyrRGBs[0], cen, pt2,Scalar(0,255,0));
		//}
		//pCurFrame_->_acvgmShrPtrPyrRGBs[0]->upload(*pCurFrame_->_acvmShrPtrPyrRGBs[0]);
		//imwrite("tmp.png",*pCurFrame_->_acvmShrPtrPyrRGBs[0]);
}
void CKinFuTracker::displayAllGlobalFeatures(int lvl_,bool bRenderSpheres_) const{
	// Generate the data
	if (_nMatchingMethod == ICP) return;

	_pCubicGrids->displayAllGlobalFeatures(lvl_, bRenderSpheres_);

	return;
}

void CKinFuTracker::displayTrackedKeyPoints()
{
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	GLboolean bLightIsOn;
	glGetBooleanv(GL_LIGHTING, &bLightIsOn);
	if (bLightIsOn){
		glDisable(GL_LIGHTING);
	}
	glPointSize(5.f);
	glLineWidth(1.f);

	Mat& Xrs = _v_selected_world[_best_k];
	Mat& Xls = _v_selected_curr[_best_k];
	int nn = _v_selected_world[_best_k].cols;
	Eigen::Affine3f T_cw;
	getCurrentProjectionMatrix(&T_cw);

	glColor3f(1.f, 0.f, 0.f);
	glBegin(GL_LINES);
	for (int i = 0; i < nn; i++)
	{
		Vector3f Xl = Vector3f(Xls.ptr<float3>()[i].x, Xls.ptr<float3>()[i].y, Xls.ptr<float3>()[i].z);
		Vector3f Xw = T_cw.linear().transpose()*Xl - T_cw.linear().transpose()*T_cw.translation();
		glVertex3fv(Xw.data());
		Xw = Vector3f(Xrs.ptr<float3>()[i].x, Xrs.ptr<float3>()[i].y, Xrs.ptr<float3>()[i].z);
		glVertex3fv(Xw.data());
	}
	glEnd();

	if (bLightIsOn){
		glEnable(GL_LIGHTING);
	}
	return;
}
	
void CKinFuTracker::displayGlobalRelocalization()
{
	return;
	if (_nMatchingMethod == ICP) return;
	glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
	GLboolean bLightIsOn;
	glGetBooleanv(GL_LIGHTING,&bLightIsOn);
	if (bLightIsOn){
		glDisable(GL_LIGHTING);
	}
	glPointSize(5.f);
	glLineWidth(1.f);
	Mat inliers_reloc;
	Mat pts_curr_world_reloc;
	Mat pts_global_reloc;
	Mat kp_2d_curr_reloc;

	if (_nStage == btl::Tracking_n_Mapping||_nStage == btl::Relocalisation_Only){
		if (_best_k < _v_selected_inliers_reloc.size()&&_best_k>=0) _v_selected_inliers_reloc[_best_k].copyTo(inliers_reloc);
		if (_nAppearanceMatched > 0 && !_gpu_pts_curr_reloc.empty()){
			Matrix3f Rw; Vector3f Tw; btl::utility::convertPrj2Rnt(_pose_feature_c_f_w, &Rw, &Tw);
			GpuMat pts = _gpu_pts_curr_reloc.clone(), nls = _gpu_nls_curr_reloc.clone(), mds = _gpu_mds_curr_reloc.clone();
			btl::device::cuda_transform_local2world(Rw.data(), Tw.data(), &pts, &nls, &mds);
			pts.colRange(0, _nAppearanceMatched).download(pts_curr_world_reloc);
			_gpu_pts_global_reloc.colRange(0, _nAppearanceMatched).download(pts_global_reloc);
			_gpu_2d_curr_reloc.colRange(0, _nAppearanceMatched).download(kp_2d_curr_reloc);
		}
	}
	else if (_nStage == btl::Tracking_NonStation){

	}
	else if (_nStage == btl::Mapping_Using_GT){
		if (_total_refined_features>0 && _nAppearanceMatched > 0){
			_gpu_relined_inliers.colRange(0, _total_refined_features).download(inliers_reloc);
			Matrix3f Rw; Vector3f Tw; btl::utility::convertPrj2Rnt(_pose_feature_c_f_w, &Rw, &Tw);
			GpuMat pts = _gpu_pts_curr_reloc.clone(), nls = _gpu_nls_curr_reloc.clone(), mds = _gpu_mds_curr_reloc.clone();
			btl::device::cuda_transform_local2world(Rw.data(), Tw.data(), &pts, &nls, &mds);
			pts.colRange(0, _nAppearanceMatched).download(pts_curr_world_reloc);
			_gpu_pts_global_reloc.colRange(0, _nAppearanceMatched).download(pts_global_reloc);
			_gpu_2d_curr_reloc.colRange(0, _nAppearanceMatched).download(kp_2d_curr_reloc);
		}
	}

	if( inliers_reloc.empty() ) return; 
	// Generate the data
	for (int i = 0; i < inliers_reloc.cols; i++){
		int nIdx = inliers_reloc.ptr<int>()[i];
		if (nIdx >= pts_global_reloc.cols){
			cout << ("Failure - inlier index incorrect.\n");
			continue;
		}
		const float3* pPt_global = pts_global_reloc.ptr<float3>() + nIdx;
		const float3* pPt_curr = pts_curr_world_reloc.ptr<float3>() + nIdx;
		//draw a line
		glBegin ( GL_LINES );
		glVertex3fv ( (float*) pPt_global );
		glVertex3fv ( (float*) pPt_curr );
		glEnd();

		/*glMatrixMode ( GL_MODELVIEW );
		glPushMatrix();
		glColor3f ( 0.f,0.f,1.f ); 
		glTranslatef ( pPt_global->x, pPt_global->y, pPt_global->z );
		gluSphere ( quadratic, 0.01f, 6, 6 );
		glPopMatrix();
		glPushMatrix();
		glColor3f ( 0.f,1.f,1.f ); 
		glTranslatef ( pPt_curr->x, pPt_curr->y, pPt_curr->z );
		gluSphere ( quadratic, 0.01f, 6, 6 );
		glPopMatrix();*/

		glBegin ( GL_POINTS );
		glColor3f ( 0.f,0.f,1.f ); 
		glVertex3fv ( (const float*)pPt_global );  
		glColor3f ( 0.f,1.f,1.f ); 
		glVertex3fv ( (const float*)pPt_curr );  
		glEnd();
		/*float fNX = cvmGlobalKeyPoint.ptr<float>(3)[nIdx];
		float fNY = cvmGlobalKeyPoint.ptr<float>(4)[nIdx];
		float fNZ = cvmGlobalKeyPoint.ptr<float>(5)[nIdx];
		glNormal3f ( fNX, fNY, fNZ );  */
	}

	if (bLightIsOn){
		glEnable(GL_LIGHTING);
	}
	return;
}
	
void CKinFuTracker::displayCameraPath() const{
	vector<Eigen::Affine3f>::const_iterator cit = _v_prj_c_f_w_training.begin(); //from world to camera
	
	glDisable(GL_LIGHTING);
	//glColor3i ( 255,0,0); 
	glColor3f ( 0.f,0.f,1.f); 
	glLineWidth(2.f);
	glBegin(GL_LINE_STRIP);
	for (; cit != _v_prj_c_f_w_training.end(); cit++ )
	{
		Matrix3f mR = cit->linear();
		Vector3f vT = cit->translation();
		Vector3f vC = -mR.transpose() * vT;
		vC = -mR.transpose()*vT;
		glVertex3fv( vC.data() );
	}
	glEnd();
	return;
}

void CKinFuTracker::displayCameraPathReloc() const{
	glDisable(GL_LIGHTING);
	glColor3f ( 1.f,0.f,0.f ); 
	glLineWidth(2.f);
	glBegin(GL_LINE_STRIP);
	for (vector<Eigen::Affine3f>::const_iterator cit = _v_relocalisation_p_matrices.begin(); cit != _v_relocalisation_p_matrices.end(); cit++) {
		Vector3f Cw = - cit->linear().transpose() * cit->translation();
		glVertex3fv( Cw.data() );
	}
	glEnd();
	return;
}

void CKinFuTracker::displayVisualRays(btl::gl_util::CGLUtil::tp_ptr pGL_,  const int lvl_, bool bRenderSphere_ ) const{

	Eigen::Affine3f prj_w_f_c = _pose_refined_c_f_w.inverse();
	Vector3f cw = prj_w_f_c.translation();//-_projection_c_f_w.linear().transpose()*_projection_c_f_w.translation();
	glPointSize(3.f);
	glLineWidth(2.f);

	glColor3f(1.f,0.f,1.f);
	if( lvl_ <0 || lvl_ >= _nFeatureScale) return;
	if( _gpu_pts_curr[lvl_].empty() ) return;
	Mat points;   _gpu_pts_curr[lvl_].download(points);
	Mat normal;   _gpu_nls_curr[lvl_].download(normal);
	Mat main_d;   _gpu_mds_curr[lvl_].download(main_d);
	Mat kp; _gpu_keypoints_curr[lvl_].download(kp);

	const float f = (_fx + _fy) / 2.f;
	//camera centre
	//render visual rays
	for ( int i=0; i < points.cols; i+=5 ) {
		float3 pt = points.ptr<float3>()[i];
		float3 nl = normal.ptr<float3>()[i];
		nl = nl/sqrt(nl.x*nl.x + nl.y*nl.y + nl.z*nl.z);
		float3 md = main_d.ptr<float3>()[i];
		float  sz = kp.ptr<float>(2)[i];// 0.01f;//
		float x_2d = kp.ptr<float>(0)[i];
		float y_2d = kp.ptr<float>(1)[i];
		float angle = kp.ptr<float>(7)[i];
		angle = angle / 180.f * float(M_PI);
		float radius = kp.ptr<float>(8)[i];
		if (radius < 20) continue;
		float o_x = radius*cos(angle) + x_2d;
		float o_y = radius*sin(angle) + y_2d;



		if (lvl_ == 0) {
			glColor4f(0.f,0.f,1.f,0.2f);
		}
		else if( lvl_== 1) {
			glColor4f(1.f,0.4f,.4f,0.2f);
		}
		else if( lvl_== 2){
			glColor4f(1.f,.5f,0.f,0.2f);
		}
		else if( lvl_== 3){
			glColor4f(1.f,0.69f,.4f,0.2f);
		}
		else {
			glColor4f(1.f,1.f,0.f,0.2f);
		}


		{
			//glDisable(GL_BLEND);
			//glEnable(GL_DEPTH_TEST);


			glDisable(GL_LIGHTING);
			//feature reference system
			glLineWidth(2.f);
			glBegin(GL_POINTS);
			glVertex3f ( pt.x, pt.y, pt.z );  
			glEnd();
			//normal
			glBegin(GL_LINES); 
			glColor4f(1.f, 0.f, 0.f, .8f);
			glVertex3f(pt.x, pt.y, pt.z);
			glVertex3f( pt.x + sz*nl.x, pt.y + sz*nl.y, pt.z + sz*nl.z );
			//draw main direction
			glColor4f(0.f,1.f,0.f,.8f);
			glVertex3f ( pt.x, pt.y, pt.z );  
			glVertex3f( pt.x + sz*md.x, pt.y + sz*md.y, pt.z + sz*md.z );
			glEnd();
			//render the third arrow
			Vector3f nn(nl.x, nl.y, nl.z);
			Vector3f v1(md.x, md.y, md.z);
			Vector3f v2 = nn.cross(v1);
			//glColor4f(0.f,0.f,1.f,.8f);
			//glBegin(GL_LINES); 
			//glVertex3f ( pt.x, pt.y, pt.z );  
			//glVertex3f( pt.x + sz*v2(0), pt.y + sz*v2(1), pt.z + sz*v2(2) );
			//glEnd();
			if (bRenderSphere_){//
				glMatrixMode(GL_MODELVIEW);
				glPushMatrix();
				glColor4f(0.f, 0.f, 1.f, 0.4f);
				glTranslated(pt.x, pt.y, pt.z);
				gluSphere(_quadratic, sz, 16, 16);
				glPopMatrix();
			}
			else{
				glEnable(GL_LIGHTING);
				// draw a rectangular
				glPushMatrix();
				glTranslated(pt.x, pt.y, pt.z);
				glScalef(sz, sz, sz);
				glBegin(GL_POLYGON);
				glColor4f(0.f, 0.f, 1.f, 0.4f);
				glNormal3fv(nn.data());
				glVertex3fv(v2.data());
				Vector3f tmp = v2 + v1;
				tmp.normalized(); tmp *= sz;
				glVertex3fv(tmp.data());
				glVertex3fv(v1.data());
				v2 *= -1.f;
				glVertex3fv(v2.data());
				v1 *= -1.f;
				glVertex3fv(v1.data());
				glEnd();
				glPopMatrix();
			}


			//visual ray
			glLineWidth(1.f);
			glBegin(GL_LINES);
			glVertex3f(pt.x, pt.y, pt.z);
			glVertex3fv(cw.data());
			glVertex3f(pt.x + sz*md.x, pt.y + sz*md.y, pt.z + sz*md.z);
			glVertex3fv(cw.data());
			glEnd();

			glMatrixMode(GL_MODELVIEW);
			glPushMatrix();
			{
				glMultMatrixf(prj_w_f_c.data());
				float x = x_2d - _cx; x /= f; x *= 0.2f;
				float y = y_2d - _cy; y /= f; y *= 0.2f;
				float x2 = o_x - _cx; x2 /= f; x2 *= 0.2f;
				float y2 = o_y - _cy; y2 /= f; y2 *= 0.2f;
				//draw orientation in 2d
				glColor4f(0.f, 1.f, 0.f, 1.f);
				glBegin(GL_LINES);
				glLineWidth(1.f);
				glVertex3f(x, y, 0.2f);
				glVertex3f(x2, y2, 0.2f);
				glEnd();
				//render the circle
				glBegin(GL_LINE_LOOP);
				float angle = float( M_PI / 16);
				glColor4f(0.f, 0.f, 1.f, 1.f);
				for (int i = 0; i < 32; i++){
					float o_x = radius*cos(angle*i) + x_2d;
					float o_y = radius*sin(angle*i) + y_2d;
					float x2 = o_x - _cx; x2 /= f; x2 *= 0.2f;
					float y2 = o_y - _cy; y2 /= f; y2 *= 0.2f;
					glVertex3f(x2, y2, 0.2f);
				}
				glEnd();
			}
			glPopMatrix();
		}

	}

	if( pGL_ && pGL_->_bEnableLighting )
		glEnable(GL_LIGHTING);
	return;
}

void CKinFuTracker::getPrevView( Eigen::Affine3f* pSystemPose_ ){
	*pSystemPose_ = _v_prj_c_f_w_training.back();
	cout << (_v_prj_c_f_w_training.back().matrix()) << endl;
	return;
}
void CKinFuTracker::getNextView( Eigen::Affine3f* pSystemPose_ ){
	*pSystemPose_ = _v_prj_c_f_w_training.front();
	cout << (_v_prj_c_f_w_training.front().matrix()) << endl;
	return;
}

void CKinFuTracker::storeGlobalFeaturesAndCameraPoses(const string& folder_)
{
	std::experimental::filesystem::path dir(folder_.c_str());
	if (std::experimental::filesystem::create_directories(dir)) {
		std::cout << "Success" << "\n";
	}

	//store features
	if (_nMatchingMethod == IDF || _nMatchingMethod == GM || _nMatchingMethod == GM_GPU ){
		_pCubicGrids->storeGlobalFeatures(folder_);
	}

	//store model scale (not to be load)
	{
		string global_features = folder_ + string("ModelScale.yml");
		cv::FileStorage storage(global_features, cv::FileStorage::WRITE);
		storage << "fVolumeSizeX" << _pCubicGrids->_fVolumeSizeM.x;
		storage << "fVolumeSizeY" << _pCubicGrids->_fVolumeSizeM.y;
		storage << "fVolumeSizeZ" << _pCubicGrids->_fVolumeSizeM.z;
		storage << "uResolution" << _nResolution;
		storage << "nFeatureScales" << _nFeatureScale;

		storage.release();  
	}

	//store camera poses
	vector<Mat> vPoses;
	for (unsigned int i=0; i< _v_prj_c_f_w_training.size(); i++) {
		Mat pos; convert(_v_prj_c_f_w_training[i], &pos);//convert vector<Eigen::Matrix4f> to vector<Mat>
		vPoses.push_back(pos);
	}
	string camera_poses = folder_ + string("Poses.yml");
	cv::FileStorage storagePose(camera_poses, cv::FileStorage::WRITE);
	storagePose << "Poses" << vPoses;
	
	storagePose.release();

	return;
}

void CKinFuTracker::loadGlobalFeaturesAndCameraPoses(const string& folder_)
{
	//lvl1
	if (_nMatchingMethod == IDF || _nMatchingMethod == GM || _nMatchingMethod == GM_GPU ){
		_pCubicGrids->loadGlobalFeatures(folder_);
	}
	
	//load camera poses
	vector<Mat> vPoses;
	string camera_poses = folder_ + string("Poses.yml");
	cv::FileStorage storagePose(camera_poses, cv::FileStorage::READ);
	storagePose["Poses"] >> vPoses;
	storagePose.release();
	//convert vector<Mat> to vector<Eigen::Matrix4f>  
	_v_prj_c_f_w_training.clear();
	for (unsigned int i = 0; i < vPoses.size(); i++) {
		Eigen::Affine3f eiPos;
		convert(vPoses[i], &eiPos);
		_v_prj_c_f_w_training.push_back(eiPos);
	}
	return;
}


}//geometry
}//btl
