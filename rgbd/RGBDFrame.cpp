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



#define EXPORT
#define INFO
//gl
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
//boost
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/math/special_functions/fpclassify.hpp> //isnan
#include <boost/lexical_cast.hpp>
//stl
#include <vector>
#include <iostream>
#include <fstream>
#include <list>
#include <math.h>
//openncv
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

#include "OtherUtil.hpp"
#include "Converters.hpp"
#include "EigenUtil.hpp"
#include "GLUtil.hpp"
#include <map>
#include "Camera.h"
#include "Kinect.h"
//#include "PlaneObj.h"
//#include "Histogram.h"

#include "RGBDFrame.h"
#include "CVUtil.hpp"
#include "Utility.hpp"
#include "CudaLib.cuh"
#include "pcl/internal.h"

using namespace Eigen;

btl::kinect::CRGBDFrame::CRGBDFrame( btl::image::SCamera::tp_ptr pRGBCamera_, ushort uResolution_, ushort uPyrLevel_, const Vector3f& eivCw_/*float fCwX_, float fCwY_, float fCwZ_*/ )
:_pRGBCamera(pRGBCamera_),_uResolution(uResolution_),_uPyrHeight(uPyrLevel_),_eivInitCw(eivCw_){
	allocate();
	initRT();
}
btl::kinect::CRGBDFrame::CRGBDFrame(const CRGBDFrame::tp_ptr pFrame_ )
{
	_pRGBCamera = pFrame_->_pRGBCamera;
	_uResolution = pFrame_->_uResolution;
	_uPyrHeight = pFrame_->_uPyrHeight;
	_eivInitCw = pFrame_->_eivInitCw;
	allocate();
	pFrame_->copyTo(this);
}

btl::kinect::CRGBDFrame::CRGBDFrame(const CRGBDFrame& Frame_)
{
	_pRGBCamera = Frame_._pRGBCamera;
	_uResolution = Frame_._uResolution;
	_uPyrHeight = Frame_._uPyrHeight;
	_eivInitCw = Frame_._eivInitCw;
	allocate();
	Frame_.copyTo(this);
}

btl::kinect::CRGBDFrame::~CRGBDFrame()
{

}
void btl::kinect::CRGBDFrame::allocate(){
	namespace btl_knt = btl::kinect;
	//_pBroxOpticalFlow.reset();
	//_pBroxOpticalFlow.reset(new cv::cuda::BroxOpticalFlow(80,100,0.5,3,10,5));
	
	_pcvgmPrev.reset();
	_pcvgmPrev.reset(new cv::cuda::GpuMat(btl_knt::__aDepthH[_uResolution],btl_knt::__aDepthW[_uResolution],CV_32FC1));
	_pcvgmCurr.reset();
	_pcvgmCurr.reset(new cv::cuda::GpuMat(btl_knt::__aDepthH[_uResolution],btl_knt::__aDepthW[_uResolution],CV_32FC1));
	_pcvgmU.reset();
	_pcvgmU.reset(new cv::cuda::GpuMat(btl_knt::__aDepthH[_uResolution],btl_knt::__aDepthW[_uResolution],CV_32FC1));
	_pcvgmV.reset();
	_pcvgmV.reset(new cv::cuda::GpuMat(btl_knt::__aDepthH[_uResolution],btl_knt::__aDepthW[_uResolution],CV_32FC1));

	//_sNormalHist.init(4);

	//disparity
	for(int i=0; i<_uPyrHeight; i++){
		//int nRows = _pRGBCamera->_sHeight >> i;
		//int nCols = _pRGBCamera->_sWidth >> i;//__aKinectW[_uResolution]>>i;
		int nRowsRGB = __aRGBH[_uResolution] >> i;
		int nColsRGB = __aRGBW[_uResolution] >> i;//__aKinectW[_uResolution]>>i;

		int nRowsDepth = __aDepthH[_uResolution]>>i;
		int nColsDepth = __aDepthW[_uResolution]>>i;

		//host
		_acvmShrPtrPyrPts[i] .reset();
		_acvmShrPtrPyrPts[i] .reset(new cv::Mat(nRowsDepth,nColsDepth,CV_32FC3));
		_acvmShrPtrPyrNls[i] .reset();
		_acvmShrPtrPyrNls[i] .reset(new cv::Mat(nRowsDepth,nColsDepth,CV_32FC3));
		_acvmShrPtrPyrReliability[i].reset();
		_acvmShrPtrPyrReliability[i].reset(new cv::Mat(nRowsDepth, nColsDepth, CV_32FC1));
		_acvmShrPtrPyrDepths[i]	 .reset();
		_acvmShrPtrPyrDepths[i]	 .reset(new cv::Mat(nRowsDepth,nColsDepth,CV_32FC1));

		_acvmShrPtrPyrRGBs[i].reset();
		_acvmShrPtrPyrRGBs[i].reset(new cv::Mat(nRowsRGB,nColsRGB,CV_8UC3));
		_acvmShrPtrPyrBWs[i] .reset();
		_acvmShrPtrPyrBWs[i] .reset(new cv::Mat(nRowsRGB,nColsRGB,CV_8UC1));
		//device
		_acvgmShrPtrPyrPts[i] .reset();
		_acvgmShrPtrPyrPts[i] .reset(new cv::cuda::GpuMat(nRowsDepth,nColsDepth,CV_32FC3));
		_acvgmShrPtrPyrNls[i] .reset();
		_acvgmShrPtrPyrNls[i] .reset(new cv::cuda::GpuMat(nRowsDepth,nColsDepth,CV_32FC3));
		_acvgmShrPtrPyrReliability[i].reset();
		_acvgmShrPtrPyrReliability[i].reset(new cv::cuda::GpuMat(nRowsDepth, nColsDepth, CV_32FC1));
		_acvgmShrPtrPyrDepths[i].reset();
		_acvgmShrPtrPyrDepths[i].reset(new cv::cuda::GpuMat(nRowsDepth,nColsDepth,CV_32FC1));

		_acvgmShrPtrPyrRGBs[i].reset();
		_acvgmShrPtrPyrRGBs[i].reset(new cv::cuda::GpuMat(nRowsRGB,nColsRGB,CV_8UC3));
		_acvgmShrPtrPyrBWs[i] .reset();
		_acvgmShrPtrPyrBWs[i] .reset(new cv::cuda::GpuMat(nRowsRGB,nColsRGB,CV_8UC1));

		_pry_mask[i].reset();
		_pry_mask[i].reset(new cv::cuda::GpuMat(nRowsRGB, nColsRGB, CV_8UC1));
		
		_acvgmShrPtrPyrDisparity[i].reset();
		_acvgmShrPtrPyrDisparity[i].reset(new cv::cuda::GpuMat(nRowsDepth,nColsDepth,CV_32FC1));
		_acvgmShrPtrPyr32FC1Tmp[i].reset();
		_acvgmShrPtrPyr32FC1Tmp[i].reset(new cv::cuda::GpuMat(nRowsDepth,nColsDepth,CV_32FC1));
	}

	_eConvention = btl::utility::BTL_CV;

	//rendering
	glPixelStorei ( GL_UNPACK_ALIGNMENT, 4 ); // 4
}

void btl::kinect::CRGBDFrame::setRTFromC(float fXA_, float fYA_, float fZA_, float fCwX_,float fCwY_,float fCwZ_){
	cv::Mat_<float> cvmR,cvmRVec(3,1);
	cvmRVec << fXA_,fYA_,fZA_;
	cv::Rodrigues(cvmRVec,cvmR);
	using namespace btl::utility;
	_Rw << cvmR;
	Vector3f eivC(fCwX_,fCwY_,fCwZ_); //camera location in the world cv-convention
	_Tw = -_Rw*eivC;
}

void btl::kinect::CRGBDFrame::setRTw(const Matrix3f& eimRotation_, const Vector3f& eivTw_){
	_Rw = eimRotation_;
	_Tw = eivTw_;
}

void btl::kinect::CRGBDFrame::setRTFromPrjWfC(const Eigen::Affine3f& prj_wfc_){
	_Rw = prj_wfc_.linear().transpose().eval();
	_Tw = -_Rw*prj_wfc_.translation();
}

void btl::kinect::CRGBDFrame::setRTFromPrjCfW(const Eigen::Affine3f& prj_cfw_){
	_Rw = prj_cfw_.linear();
	_Tw = prj_cfw_.translation();
}

void btl::kinect::CRGBDFrame::setRTFromC(const Matrix3f& eimRotation_, const Vector3f& eivCw_){
	_Rw = eimRotation_;
	_Tw = -_Rw*eivCw_;
}

void btl::kinect::CRGBDFrame::initRT(){
	_Rw << 1.f, 0.f, 0.f,
		      0.f, 1.f, 0.f,
			  0.f, 0.f, 1.f;
	_Tw = -_eivInitCw; 
}

void btl::kinect::CRGBDFrame::copyRTFrom(const CRGBDFrame& cFrame_ ){
	//assign rotation and translation 
	_Rw = cFrame_._Rw;
	_Tw = cFrame_._Tw;
}

void btl::kinect::CRGBDFrame::assignRTfromGL(){
	btl::gl_util::CGLUtil::getRTFromWorld2CamCV(&_Rw,&_Tw);
}

void btl::kinect::CRGBDFrame::copyTo( CRGBDFrame* pKF_, const short sLevel_ ) const{
	//host
	if( !_acvmShrPtrPyrPts[sLevel_]->empty()) _acvmShrPtrPyrPts[sLevel_]->copyTo(*pKF_->_acvmShrPtrPyrPts[sLevel_]);
	if( !_acvmShrPtrPyrNls[sLevel_]->empty()) _acvmShrPtrPyrNls[sLevel_]->copyTo(*pKF_->_acvmShrPtrPyrNls[sLevel_]);
	_acvmShrPtrPyrRGBs[sLevel_]->copyTo(*pKF_->_acvmShrPtrPyrRGBs[sLevel_]);
	_acvmShrPtrPyrBWs[sLevel_]->copyTo(*pKF_->_acvmShrPtrPyrBWs[sLevel_]);
	//device
	if( !_acvgmShrPtrPyrPts[sLevel_]->empty()) _acvgmShrPtrPyrPts[sLevel_]->copyTo(*pKF_->_acvgmShrPtrPyrPts[sLevel_]);
	if (!_acvgmShrPtrPyrNls[sLevel_]->empty()) _acvgmShrPtrPyrNls[sLevel_]->copyTo(*pKF_->_acvgmShrPtrPyrNls[sLevel_]);
	if (!_acvgmShrPtrPyrReliability[sLevel_]->empty()) _acvgmShrPtrPyrReliability[sLevel_]->copyTo(*pKF_->_acvgmShrPtrPyrReliability[sLevel_]);
	if (!_pry_mask[sLevel_]->empty()) _pry_mask[sLevel_]->copyTo(*pKF_->_pry_mask[sLevel_]);
	_acvgmShrPtrPyrRGBs[sLevel_]->copyTo(*pKF_->_acvgmShrPtrPyrRGBs[sLevel_]);
	_acvgmShrPtrPyrBWs[sLevel_]->copyTo(*pKF_->_acvgmShrPtrPyrBWs[sLevel_]);
	pKF_->_eConvention = _eConvention;
}

void btl::kinect::CRGBDFrame::copyTo( CRGBDFrame* pKF_ ) const{
	for(int i=0; i<_uPyrHeight; i++) {
		copyTo(pKF_,i);
	}
	_acvgmShrPtrPyrDepths[0]->copyTo(*pKF_->_acvgmShrPtrPyrDepths[0]);
	_acvmShrPtrPyrDepths[0]->copyTo(*pKF_->_acvmShrPtrPyrDepths[0]);
	//other
	pKF_->_Rw = _Rw;
	pKF_->_Tw = _Tw;
}

void btl::kinect::CRGBDFrame::copyImageTo( CRGBDFrame* pKF_ ) const{
	for(ushort sLevel_=0; sLevel_<_uPyrHeight; sLevel_++) {
		_acvmShrPtrPyrRGBs[sLevel_]->copyTo(*pKF_->_acvmShrPtrPyrRGBs[sLevel_]);
		_acvmShrPtrPyrBWs[sLevel_]->copyTo(*pKF_->_acvmShrPtrPyrBWs[sLevel_]);
		//device
		_acvgmShrPtrPyrRGBs[sLevel_]->copyTo(*pKF_->_acvgmShrPtrPyrRGBs[sLevel_]);
		_acvgmShrPtrPyrBWs[sLevel_]->copyTo(*pKF_->_acvgmShrPtrPyrBWs[sLevel_]);
	}
}
void btl::kinect::CRGBDFrame::exportYML(const std::string& strPath_, const std::string& strYMLName_){
	using namespace btl::utility;

	std::string strPathFileName = strPath_ + strYMLName_;
	cv::FileStorage cFSWrite( strPathFileName.c_str(), cv::FileStorage::WRITE );

	cFSWrite << "uPyrHeight" << _uPyrHeight;
	cFSWrite << "uResolution" << _uResolution;
	cv::Mat cvmRw; cvmRw << _Rw;
	cv::Mat cvmTw; cvmTw << _Tw;
	cFSWrite << "eimRw" << cvmRw;
	cFSWrite << "eivTw" << cvmTw;

	std::string strVariableName;
	for (int i = 0; i < _uPyrHeight; i++){
		strVariableName = "acvmShrPtrPyrPts";	strVariableName += boost::lexical_cast<std::string> ( i ); cFSWrite << strVariableName.c_str() << *_acvmShrPtrPyrPts[i];
		strVariableName = "acvmShrPtrPyrNls";	strVariableName += boost::lexical_cast<std::string> ( i ); cFSWrite << strVariableName.c_str() << *_acvmShrPtrPyrNls[i];
		strVariableName = "acvmShrPtrPyrRGBs";	strVariableName += boost::lexical_cast<std::string> ( i ); cFSWrite << strVariableName.c_str() << *_acvmShrPtrPyrRGBs[i];
		strVariableName = "acvmShrPtrPyrBWs";	strVariableName += boost::lexical_cast<std::string> ( i ); cFSWrite << strVariableName.c_str() << *_acvmShrPtrPyrBWs[i];
		strPathFileName = strPath_+strYMLName_; strPathFileName += boost::lexical_cast<std::string> ( i );	strPathFileName += ".bmp";	cv::imwrite(strPathFileName,*_acvmShrPtrPyrBWs[i]);
	}
	
	cFSWrite.release();
}

void btl::kinect::CRGBDFrame::importYML(const std::string& strPath_, const std::string& strYMLName_){
	using namespace btl::utility;
	std::string strPathFileName = strPath_ + strYMLName_;
	cv::FileStorage cFSRead( strPathFileName.c_str(), cv::FileStorage::READ );

	cFSRead["uPyrHeight"] >> _uPyrHeight;
	cFSRead["uResolution"] >> _uResolution;
	cv::Mat cvmRw; MatrixXf eimMat;
	cv::Mat cvmTw; VectorXf eimVec;
	cFSRead["eimRw"] >> cvmRw;
	cFSRead["eivTw"] >> cvmTw;
	_Rw = (eimMat << cvmRw);
	_Tw = (eimVec << cvmTw);

	std::string strVariableName;
	for (int i = 0; i < _uPyrHeight; i++){
		strVariableName = "acvmShrPtrPyrPts";  strVariableName += boost::lexical_cast<std::string> ( i ); cFSRead[strVariableName.c_str()] >> *_acvmShrPtrPyrPts[i];
		strVariableName = "acvmShrPtrPyrNls";  strVariableName += boost::lexical_cast<std::string> ( i ); cFSRead[strVariableName.c_str()] >> *_acvmShrPtrPyrNls[i];
		strVariableName = "acvmShrPtrPyrRGBs"; strVariableName += boost::lexical_cast<std::string> ( i ); cFSRead[strVariableName.c_str()] >> *_acvmShrPtrPyrRGBs[i];
		strVariableName = "acvmShrPtrPyrBWs";  strVariableName += boost::lexical_cast<std::string> ( i ); cFSRead[strVariableName.c_str()] >> *_acvmShrPtrPyrBWs[i];
	}

	cFSRead.release();
}

/*
double btl::kinect::CKeyFrame::gpuCalcRTBroxOpticalFlow ( const CKeyFrame& sPrevFrameWorld_, const double dDistanceThreshold_, unsigned short* pInliers_) {
	// - The reference frame must contain a calibrated Rw and Tw. 
	// - The point cloud in the reference (previous) frame must be transformed into the world coordinate system.
	// - The current frame's Rw and Tw must be initialized as the reference's Rw Tw. (This is for fDist = norm3<float>() ) 
	// - The point cloud in the current frame must be in the camera coordinate system.

	//establish the point correspondences using dense optical flow
	cv::cuda::GpuMat cvgmPrev,cvgmCurr,u,v;
	sPrevFrameWorld_._acvgmShrPtrPyrBWs[0]->convertTo(cvgmPrev,cv::DataType<float>::type);
	_acvgmShrPtrPyrBWs[0]->convertTo(cvgmCurr,cv::DataType<float>::type);
	(*_pBroxOpticalFlow)(cvgmPrev,cvgmCurr,u,v);

	cv::Mat cvmU,cvmV; u.download(cvmU); v.download(cvmV);
	return 0;
}

void btl::kinect::CKeyFrame::gpuBroxOpticalFlow (const CKeyFrame& sPrevFrameWorld_, cv::cuda::GpuMat* pcvgmColorGraph_){
	sPrevFrameWorld_._acvgmShrPtrPyrBWs[0]->convertTo(*_pcvgmPrev,cv::DataType<float>::type);
	_acvgmShrPtrPyrBWs[0]->convertTo(*_pcvgmCurr,cv::DataType<float>::type);
	//calc the brox optical flow
	(*_pBroxOpticalFlow)(*_pcvgmPrev,*_pcvgmCurr,*_pcvgmU,*_pcvgmV);
	//calc the color graph
	gpuConvert2ColorGraph( &*_pcvgmU, &*_pcvgmV, &*pcvgmColorGraph_ );
	//translate magnitude to range [0;1]
	return;
}*/

void btl::kinect::CRGBDFrame::gpuConvert2ColorGraph( cv::cuda::GpuMat* pcvgmU_, cv::cuda::GpuMat* pcvgmV_, cv::cuda::GpuMat* pcvgmColorGraph_ )
{
	//calc the color graph
	//initialize two GpuMat but reuse the input U and V;
	cv::cuda::GpuMat cvgmMag(pcvgmU_->size(),pcvgmU_->type(),pcvgmU_->data);
	cv::cuda::GpuMat cvgmAngle(pcvgmV_->size(),pcvgmV_->type(),pcvgmV_->data);
	//transform to polar 
	cv::cuda::cartToPolar(*pcvgmU_,*pcvgmV_,cvgmMag,cvgmAngle,true);

	//translate magnitude to range [0;1]
	double mag_max;
	//cv::cuda::minMaxLoc(cvgmMag, 0, &mag_max);
	cv::cuda::minMax(cvgmMag, 0, &mag_max);
	cvgmMag.convertTo(cvgmMag, -1, 1.0 / mag_max);

	cv::cuda::GpuMat cvgmHSV(pcvgmU_->size(),pcvgmU_->type(),_pcvgmPrev->data);
	cv::cuda::GpuMat cvgmOnes(pcvgmU_->size(),pcvgmU_->type(),_pcvgmCurr->data);
	cvgmOnes.setTo(1.f);

	//build hsv image
	std::vector<cv::cuda::GpuMat> vcvgmHSV;
	vcvgmHSV.push_back(cvgmAngle);
	vcvgmHSV.push_back(cvgmOnes);
	vcvgmHSV.push_back(cvgmMag);
	cv::cuda::merge(vcvgmHSV,cvgmHSV);
	//convert to BGR and show

	cv::cuda::cvtColor(cvgmHSV,cvgmOnes,cv::COLOR_HSV2BGR);//cvgmBGR is CV_32FC3 matrix
	cvgmOnes.convertTo(*pcvgmColorGraph_,CV_8UC3,255);
}

/*
double btl::kinect::CKeyFrame::calcRT ( const CKeyFrame& sPrevKF_, const unsigned short sLevel_ , const double dDistanceThreshold_, unsigned short* pInliers_) {
	// - The reference frame must contain a calibrated Rw and Tw. 
	// - The point cloud in the reference frame must be transformed into the world coordinate system.
	// - The current frame's Rw and Tw must be initialized as the reference's Rw Tw. (This is for fDist = norm3<float>() ) 
	// - The point cloud in the current frame must be in the camera coordinate system.
	//BTL_ASSERT(sPrevKF_._vKeyPoints.size()>10,"extractSurfFeatures() Too less SURF features detected in the reference frame");
	//matching from current to reference
	cv::cuda::BFMatcher_CUDA cBruteMatcher;
	cv::cuda::GpuMat cvgmTrainIdx, cvgmDistance;
	cBruteMatcher.matchSingle( this->_cvgmDescriptors,  sPrevKF_._cvgmDescriptors, cvgmTrainIdx, cvgmDistance);
	cv::cuda::BFMatcher_CUDA::matchDownload(cvgmTrainIdx, cvgmDistance, _vMatches);
	std::sort( _vMatches.begin(), _vMatches.end() );
	if (_vMatches.size()> 100) { _vMatches.erase( _vMatches.begin()+ 300, _vMatches.end() ); }
	//CHECK ( !_vMatches.empty(), "SKeyFrame::calcRT() _vMatches should not calculated." );
	//calculate the R and T
	return calcRTFromPair(sPrevKF_,dDistanceThreshold_,&*pInliers_);
}// calcRT*/

/*
float btl::kinect::CKeyFrame::calcRTFromPair(const CKeyFrame& sPrevKF_, const double dDistanceThreshold_, unsigned short* pInliers_){
	//calculate the R and T
	//search for pairs of correspondences with depth data available.
	const float*const  _pCurrPts = (const float*)         _acvmShrPtrPyrPts[0]->data;
	const float*const  _pPrevPts = (const float*)sPrevKF_._acvmShrPtrPyrPts[0]->data;
	std::vector< int > _vDepthIdxCur, _vDepthIdxRef, _vSelectedPairs;
	for ( std::vector< cv::DMatch >::const_iterator cit = _vMatches.begin(); cit != _vMatches.end(); cit++ ) {
		int nKeyPointIdxCur = cit->queryIdx;
		int nKeyPointIdxRef = cit->trainIdx;
		if (_vKeyPoints[nKeyPointIdxCur].response < 0.f || sPrevKF_._vKeyPoints[nKeyPointIdxRef].response < 0.f) continue;
		int nXCur = cvRound ( 		   _vKeyPoints[ nKeyPointIdxCur ].pt.x );
		int nYCur = cvRound ( 		   _vKeyPoints[ nKeyPointIdxCur ].pt.y );
		int nXRef = cvRound ( sPrevKF_._vKeyPoints[ nKeyPointIdxRef ].pt.x );
		int nYRef = cvRound ( sPrevKF_._vKeyPoints[ nKeyPointIdxRef ].pt.y );

		int nDepthIdxCur = nYCur * __aKinectW[_uResolution] * 3 + nXCur * 3;
		int nDepthIdxRef = nYRef * __aKinectW[_uResolution] * 3 + nXRef * 3;

		if ( !boost::math::isnan<float>( _pCurrPts[ nDepthIdxCur + 2 ] ) && !boost::math::isnan<float> (_pPrevPts[ nDepthIdxRef + 2 ]  ) ){
			float fDist = btl::utility::norm3<float>( _pCurrPts + nDepthIdxCur, _pPrevPts + nDepthIdxRef, _eimRw.data(), _eivTw.data() );
			if(  fDist < dDistanceThreshold_ ) {
				_vDepthIdxCur  .push_back ( nDepthIdxCur );
				_vDepthIdxRef  .push_back ( nDepthIdxRef );
				_vSelectedPairs.push_back ( nKeyPointIdxCur );
				_vSelectedPairs.push_back ( nKeyPointIdxRef );
			}//if(  fDist < dDistanceThreshold_ ) 
		}//if ( !boost::math::isnan<float>( _pCurrPts[ nDepthIdxCur + 2 ] ) && !boost::math::isnan<float> (_pPrevPts[ nDepthIdxRef + 2 ]  ) )
	}//for ( std::vector< cv::DMatch >::const_iterator cit = _vMatches.begin(); cit != _vMatches.end(); cit++ )

	int nSize = _vDepthIdxCur.size(); 
	*pInliers_ = nSize;
	PRINT(nSize);
	//if nSize smaller than a threshould, quit
	MatrixXf eimCurCam ( 3, nSize ), eimRefWorld ( 3, nSize );
	std::vector< int >::const_iterator cit_Cur = _vDepthIdxCur.begin();
	std::vector< int >::const_iterator cit_Ref = _vDepthIdxRef.begin();

	for ( int i = 0 ; cit_Cur != _vDepthIdxCur.end(); cit_Cur++, cit_Ref++ ){
		eimCurCam ( 0, i ) = _pCurrPts[ *cit_Cur     ];
		eimCurCam ( 1, i ) = _pCurrPts[ *cit_Cur + 1 ];
		eimCurCam ( 2, i ) = _pCurrPts[ *cit_Cur + 2 ];
		eimRefWorld ( 0, i ) = _pPrevPts[ *cit_Ref     ];
		eimRefWorld ( 1, i ) = _pPrevPts[ *cit_Ref + 1 ];
		eimRefWorld ( 2, i ) = _pPrevPts[ *cit_Ref + 2 ];
		i++;
	}

	float fS2;
	float dErrorBest = btl::utility::absoluteOrientation < float > ( eimRefWorld, eimCurCam ,0.1, 15., false, &_eimRw, &_eivTw, &fS2 ); // eimB_ = R * eimA_ + T;

	//PRINT ( dErrorBest );
	//PRINT ( _eimR );
	//PRINT ( _eivT );

	return dErrorBest;
}
*/

void btl::kinect::CRGBDFrame::render3DPtsInSurfel(btl::gl_util::CGLUtil::tp_ptr pGL_, const ushort usLevel_) const {
	if (usLevel_ >= _uPyrHeight) return;
	//////////////////////////////////
	float dNx, dNy, dNz;
	float dX, dY, dZ;
	const float* pPt = (const float*) _acvmShrPtrPyrPts[usLevel_]->data;
	const float* pNl = (const float*) _acvmShrPtrPyrNls[usLevel_]->data;
	const uchar* pRGB = (const uchar*) _acvmShrPtrPyrRGBs[usLevel_]->data;
	// Generate the data
	if( pGL_ && pGL_->_bEnableLighting ){
		glEnable(GL_LIGHTING);
		float shininess = 15.0f;
		float diffuseColor[3] = {0.8f, 0.8f, 0.8f};
		float specularColor[4] = {.2f, 0.2f, 0.2f, 1.0f};
		// set specular and shininess using glMaterial (gold-yellow)
		glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess); // range 0 ~ 128
		glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specularColor);
		// set ambient and diffuse color using glColorMaterial (gold-yellow)
		glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
		glColor3fv(diffuseColor);
	}
	else                            	{glDisable(GL_LIGHTING);}
	for( unsigned int i = 0; i < btl::kinect::__aDepthWxH[usLevel_]; i++,pRGB+=3,pNl+=3,pPt+=3){
		dNx = pNl[0];		dNy =-pNl[1];		dNz =-pNl[2];
		dX =  pPt[0];		dY = -pPt[1];		dZ = -pPt[2];
		if ( pGL_ )	{ pGL_->renderDisk<float>(dX,dY,dZ,dNx,dNy,dNz,pRGB,pGL_->_fSize*(usLevel_+1.f)*.5f,pGL_->_bRenderNormal); }
		else { glColor3ubv ( pRGB ); glVertex3f ( dX, dY, dZ );}
	}
	return;
} 

void btl::kinect::CRGBDFrame::render3DPts(btl::gl_util::CGLUtil::tp_ptr pGL_, const ushort usLevel_){
	if (usLevel_ >= _uPyrHeight) return;
	//////////////////////////////////
	const float* pPt = (const float*)_acvmShrPtrPyrPts[usLevel_]->data;
	const float* pNl = (const float*)_acvmShrPtrPyrNls[usLevel_]->data;
	const uchar* pRGB = (const uchar*)_acvmShrPtrPyrRGBs[usLevel_]->data;
	// Generate the data
	if (pGL_ && pGL_->_bEnableLighting){
		glEnable(GL_LIGHTING);
		float shininess = 15.0f;
		float diffuseColor[3] = { 0.8f, 0.8f, 0.8f };
		float specularColor[4] = { .2f, 0.2f, 0.2f, 1.0f };
		// set specular and shiniess using glMaterial (gold-yellow)
		glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess); // range 0 ~ 128
		glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specularColor);
		// set ambient and diffuse color using glColorMaterial (gold-yellow)
		glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
		glColor3fv(diffuseColor);
	}
	else {
		glDisable(GL_LIGHTING); 
	}
	glPointSize(0.1f*(usLevel_ + 1) * 20);
	glBegin(GL_POINTS);
	for (unsigned int uIdx = 0; uIdx < btl::kinect::__aDepthWxH[_uResolution + usLevel_]; uIdx++){
		glColor3ubv ( pRGB ); pRGB += 3;
		glVertex3fv(pPt);  pPt += 3;
		glNormal3fv(pNl);  pNl += 3;
	}
	glEnd();
	return;
}


void btl::kinect::CRGBDFrame::gpuRender3DPts(btl::gl_util::CGLUtil::tp_ptr pGL_,const ushort usPyrLevel_){
	//the 3-D points have been transformed in world already
	if( pGL_ && pGL_->_bEnableLighting ){
		glEnable(GL_LIGHTING); /* glEnable(GL_TEXTURE_2D);*/
		float shininess = 15.0f;
		float diffuseColor[3] = {0.8f, 0.8f, 0.8f};
		float specularColor[4] = {.2f, 0.2f, 0.2f, 1.0f};
		// set specular and shiniess using glMaterial (gold-yellow)
		glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess); // range 0 ~ 128
		glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specularColor);
		// set ambient and diffuse color using glColorMaterial (gold-yellow)
		glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
		glColor3fv(diffuseColor);
	}
	else                            	{glDisable(GL_LIGHTING);/* glEnable(GL_TEXTURE_2D);*/}
	glPointSize(0.1f*(usPyrLevel_+1+_uResolution)*20.f);
	if (usPyrLevel_ >= _uPyrHeight) return;
	//glEnableClientState(GL_VERTEX_ARRAY);
	//glEnableClientState(GL_NORMAL_ARRAY);
	//glEnableClientState(GL_COLOR_ARRAY);
	
	pGL_->gpuMapPtResources(*_acvgmShrPtrPyrPts[usPyrLevel_]);
	pGL_->gpuMapNlResources(*_acvgmShrPtrPyrNls[usPyrLevel_]);
	if(!pGL_->_bEnableLighting) pGL_->gpuMapRGBResources(*_acvgmShrPtrPyrRGBs[usPyrLevel_]);
	glDrawArrays(GL_POINTS, 0, btl::kinect::__aDepthWxH[usPyrLevel_] );
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	if(!pGL_->_bEnableLighting) glDisableClientState(GL_COLOR_ARRAY);
	glBindBuffer( GL_ARRAY_BUFFER, 0 );// it's crucially important for program correctness, it return the buffer to opengl rendering system.
	
}//gpuRenderVoxelInWorldCVGL()

void btl::kinect::CRGBDFrame::renderCameraInWorld(btl::gl_util::CGLUtil::tp_ptr pGL_, bool bRenderCoordinate_, float* color_, bool bRenderCamera_, const double& dPhysicalFocalLength_, const unsigned short uLevel_) {
	if (pGL_->_usPyrHeight != _uPyrHeight) return;
	glDisable(GL_LIGHTING);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();

	Eigen::Affine3f prj_c_f_w; prj_c_f_w.linear() = _Rw; prj_c_f_w.translation() = _Tw;
	Eigen::Affine3f prj_w_f_c = prj_c_f_w.inverse();
	glMultMatrixf(prj_w_f_c.data());//times with original model view matrix manipulated by mouse or keyboard

	glColor4f(1.f, 1.f, 1.f, .4f);
	glLineWidth(1);
#if USE_PBO
	//_pRGBCamera->renderCameraInLocal(*_pry_mask[pGL_->_usLevel], pGL_, bRenderCoordinate_, color_, dPhysicalFocalLength_, bRenderCamera_);//render small frame
	//_pRGBCamera->renderCameraInLocal(*_pry_mask[pGL_->_usLevel], pGL_, false, color_, 1.f, false);//render large frame
	_pRGBCamera->renderCameraInLocal(*_acvgmShrPtrPyrRGBs[pGL_->_usLevel], pGL_, bRenderCoordinate_, color_, dPhysicalFocalLength_, bRenderCamera_);//render small frame
	//_pRGBCamera->renderCameraInLocal(*_acvgmShrPtrPyrRGBs[pGL_->_usLevel], pGL_, false, color_, 1.f, false);//render large frame
#else 
	if (bRenderCamera_) _pRGBCamera->LoadTexture(*_acvmShrPtrPyrRGBs[uLevel_], &pGL_->_auTexture[pGL_->_usLevel]);
	_pRGBCamera->renderCameraInGLLocal(pGL_->_auTexture[pGL_->_usLevel], pGL_, bRenderCoordinate_, color_, dPhysicalFocalLength_, bRenderCamera_);
#endif	
	glPopMatrix();

	return;
}

void btl::kinect::CRGBDFrame::gpuTransformToWorld(const ushort usLevel_){
	if (usLevel_>=_uPyrHeight) return;
	GpuMat tmp;
	btl::device::cuda_transform_local2world(_Rw.data(),_Tw.data(),&*_acvgmShrPtrPyrPts[usLevel_],&*_acvgmShrPtrPyrNls[usLevel_],&tmp);
	_acvgmShrPtrPyrPts[usLevel_]->download(*_acvmShrPtrPyrPts[usLevel_]);
	_acvgmShrPtrPyrNls[usLevel_]->download(*_acvmShrPtrPyrNls[usLevel_]);
}//gpuTransformToWorldCVCV()

void btl::kinect::CRGBDFrame::gpuTransformToWorld(){
	for (ushort usI=0;usI<_uPyrHeight;usI++) {
		gpuTransformToWorld(usI);
	}
}//gpuTransformToWorldCVCV()

void btl::kinect::CRGBDFrame::applyRelativePose( const CRGBDFrame& sReferenceKF_ ){
	//1.when the Rw and Tw is: Rw * Cam_Ref + Tw = Cam_Cur
	//_eivTw = _eimRw*sReferenceKF_._eivTw + _eivTw;//1.order matters 
	//_eimRw = _eimRw*sReferenceKF_._eimRw;//2.
	//2.when the Rw and Tw is: Rw * World_Ref + Tw = World_cur
	_Tw = sReferenceKF_._Tw + sReferenceKF_._Rw*_Tw;//1.order matters 
	_Rw = sReferenceKF_._Rw*_Rw;
}

bool btl::kinect::CRGBDFrame::isMovedwrtReferencInRadiusM(const CRGBDFrame* const pRefFrame_, double dRotAngleThreshold_, double dTranslationThreshold_){
	using namespace btl::utility; //for operator <<
	//rotation angle
	cv::Mat_<float> cvmRRef,cvmRCur;
	cvmRRef << pRefFrame_->_Rw;
	cvmRCur << _Rw;
	cv::Mat_<float> cvmRVecRef,cvmRVecCur;
	cv::Rodrigues(cvmRRef,cvmRVecRef);
	cv::Rodrigues(cvmRCur,cvmRVecCur);
	cvmRVecCur -= cvmRVecRef;
	//get translation vector
	Vector3f eivCRef,eivCCur;
	eivCRef = - pRefFrame_->_Rw * pRefFrame_->_Tw;
	eivCCur = -             _Rw *             _Tw;
	eivCCur -= eivCRef;
	double dRot = cv::norm( cvmRVecCur, cv::NORM_L2 );
	double dTrn = eivCCur.norm();
	return ( dRot > dRotAngleThreshold_ || dTrn > dTranslationThreshold_);
}


