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


#define INFO
#define _USE_MATH_DEFINES
//gl
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include <memory>
//opencv
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
//nifti
#include <nifti1_io.h>   /* directly include I/O library functions */
#include <experimental/filesystem>
//eigen
#include <Eigen/Core>
#include <sophus/se3.hpp>
//openni
#include <OpenNI.h>
//self
#include "Utility.hpp"
#include "GLUtil.hpp"
#include <map>
#include "Camera.h"
#include "Kinect.h"
#include "RGBDFrame.h"
#include "VideoSourceKinect.hpp"
#include "CudaLib.cuh"
#include <iostream>
#include <string>
#include <limits>


#ifndef CHECK_RC_
#define CHECK_RC_(rc, what)	\
	BTL_ASSERT(rc == STATUS_OK, (what) ) //std::string(xnGetStatusString(rc)
#endif

using namespace btl::utility;
using namespace openni;
using namespace std;

namespace btl{ namespace kinect
{  

bool CVideoSourceKinect::_bIsSequenceEnds = false;
CVideoSourceKinect::CVideoSourceKinect (ushort uResolution_, ushort uPyrHeight_, bool bUseNIRegistration_,const Eigen::Vector3f& eivCw_, const string& cam_param_path_ )
:_uResolution(uResolution_),_uPyrHeight(uPyrHeight_),_cam_param_file(cam_param_path_)
{
	_eivCw = eivCw_;

	_nRawDataProcessingMethod = CVideoSourceKinect::tp_raw_data_processing_methods::BIFILTER_DYNAMIC;
	//other
	//definition of parameters
	_fThresholdDepthInMeter = 0.05f;
	_fSigmaSpace = 4;
	_fSigmaDisparity = 1.f/1.f - 1.f/(1.f+_fThresholdDepthInMeter);

	_bIsSequenceEnds = false;
	_bFast = true;

	_fScaleRGB = 1.f;
	_fScaleDepth = 1.f;

	_fCutOffDistance = 10.f;
	_bMapUndistortionOn = true;

	_fMtr2Depth = 1000.f;
	std::cout << " Done. " << std::endl;
}
CVideoSourceKinect::~CVideoSourceKinect()
{
	_color.stop();
	_color.destroy();
	_depth.stop();
	_depth.destroy();
	_device.close();

	openni::OpenNI::shutdown();
}

void CVideoSourceKinect::init()
{
	_nMode = SIMPLE_CAPTURING; 
	//allocate

	cout << ("Allocate buffers...") << endl;
	_cvmRGB		   .create( __aRGBH[_uResolution], __aRGBW[_uResolution], CV_8UC3 );
	_undist_rgb	   .create( __aRGBH[_uResolution], __aRGBW[_uResolution], CV_8UC3 );
	_depth_float	   .create( __aDepthH[_uResolution], __aDepthW[_uResolution], CV_32FC1);
	_undist_depth	   .create( __aDepthH[_uResolution], __aDepthW[_uResolution], CV_32FC1);

	// allocate memory for later use ( registrate the depth with rgb image
	// refreshed for every frame
	// pre-allocate cvgm to increase the speed
	_gpu_undist_depth    .create(__aDepthH[_uResolution], __aDepthW[_uResolution],CV_32FC1);

	//import camera parameters
	_pCurrFrame.reset();_pRGBCamera.reset();_pIRCamera.reset();
	cout << "1. VideoSourceKinect::init() " <<endl;
	_pRGBCamera.reset(new btl::image::SCamera(_cam_param_file+"RGB.yml"/*btl_knt::SCamera::CAMERA_RGB*/,_uResolution));

	if (_uResolution == 6)
	{
		_pIRCamera .reset(new btl::image::SCamera(_cam_param_file+"IR.yml"/*btl_knt::SCamera::CAMERA_IR*/, 0));
	}
	else
	{
		cout << _cam_param_file+"IR.yml" <<endl;
		_pIRCamera .reset(new btl::image::SCamera(_cam_param_file+"IR.yml"/*btl_knt::SCamera::CAMERA_IR*/, _uResolution));
	}

	importYML();
	cout << "2. VideoSourceKinect::init() " <<endl;
	_pCurrFrame.reset(new CRGBDFrame(_pRGBCamera.get(),_uResolution,_uPyrHeight,_eivCw));
	cout << "end of VideoSourceKinect::init()" <<endl;
}

void CVideoSourceKinect::initKinect()
{
	init();
	_nMode = SIMPLE_CAPTURING;
	cout << ("Initialize RGBD camera...") << endl;
	//inizialization 
	Status nRetVal = openni::OpenNI::initialize();
	printf("After initialization:\n%s\n", openni::OpenNI::getExtendedError());
	nRetVal = _device.open(openni::ANY_DEVICE);			//CHECK_RC_(nRetVal, "Initialize _cContext"); 
	
	nRetVal = _depth.create(_device, openni::SENSOR_DEPTH); //CHECK_RC_(nRetVal, "Initialize _cContext");
	_depth.setMirroringEnabled(false);
	nRetVal = _color.create(_device, openni::SENSOR_COLOR); // CHECK_RC_(nRetVal, "Initialize _cContext"); 
	_color.setMirroringEnabled(false);
	_colorSensorInfo = _device.getSensorInfo(openni::SENSOR_COLOR);

	if( setVideoMode(_uResolution) == STATUS_OK )
	{
		nRetVal = _depth.start(); //CHECK_RC_(nRetVal, "Create depth video stream fail");
		nRetVal = _color.start(); //CHECK_RC_(nRetVal, "Create color video stream fail"); 
	}

	if (_depth.isValid() && _color.isValid())
	{
		VideoMode depthVideoMode = _depth.getVideoMode();
		VideoMode colorVideoMode = _color.getVideoMode();

		int depthWidth = depthVideoMode.getResolutionX();
		int depthHeight = depthVideoMode.getResolutionY();
		int colorWidth = colorVideoMode.getResolutionX();
		int colorHeight = colorVideoMode.getResolutionY();

		if (depthWidth != colorWidth || depthHeight != colorHeight)
		{
			printf("Warning - expect color and depth to be in same resolution: D: %dx%d, C: %dx%d\n",
					depthWidth, depthHeight,
					colorWidth, colorHeight);
			//return ;
		}
	}

	_streams = new VideoStream*[2];
	_streams[0] = &_depth;
	_streams[1] = &_color;

	// set as the highest resolution 0 for 480x640 
	
	char _serial[100];
	int size = sizeof(_serial);
	_device.getProperty(openni::DEVICE_PROPERTY_SERIAL_NUMBER, &_serial, &size);
	_serial_number = string(_serial);
	cout << _serial_number << endl;

	Mat cpuClibXYxZ0, cpuMask0;
	if (!loadLocation(&_vRegularLocations))
	{
		_bMapUndistortionOn = false;
		return;
	}
	loadCoefficient(0, _vRegularLocations.size(), &cpuClibXYxZ0, &cpuMask0);
	_calibXYxZ[0].upload(cpuClibXYxZ0);
	_mask[0].upload(cpuMask0);
	_fCutOffDistance = _vRegularLocations.back();

	if (_uResolution == 1) {
		btl::device::cuda_resize_coeff(_calibXYxZ[0], &_calibXYxZ[1]);
		cuda::resize(_mask[0], _mask[1], Size(_mask[0].cols / 2, _mask[0].rows / 2));
	}


	cout << (" Done.\n") ;

	return;
}
void CVideoSourceKinect::initRecorder(std::string& strPath_){
	initKinect();
	_nMode = RECORDING;
	cout << ("Initialize RGBD data recorder...\n");
	_recorder.create(strPath_.c_str());
	_recorder.attach( _depth );
	_recorder.attach( _color );

	_recorder.start();
	cout << (" Done.\n");
}
void CVideoSourceKinect::initPlayer(std::string& strPathFileName_){
	init();
	_nMode = PLAYING_BACK;

	cout << ("Initialize OpenNI Player...\n");
	//inizialization 
	Status nRetVal = openni::OpenNI::initialize();
	printf("After initialization:\n%s\n", openni::OpenNI::getExtendedError());
	nRetVal = _device.open(strPathFileName_.c_str());		//CHECK_RC_(nRetVal, "Open oni file");
	nRetVal = _depth.create(_device, openni::SENSOR_DEPTH); //CHECK_RC_(nRetVal, "Initialize _cContext"); 
	nRetVal = _color.create(_device, openni::SENSOR_COLOR); //CHECK_RC_(nRetVal, "Initialize _cContext"); 

	nRetVal = _depth.start(); //CHECK_RC_(nRetVal, "Create depth video stream fail");
	nRetVal = _color.start(); //CHECK_RC_(nRetVal, "Create color video stream fail"); 

	if (_depth.isValid() && _color.isValid())
	{
		VideoMode depthVideoMode = _depth.getVideoMode();
		VideoMode colorVideoMode = _color.getVideoMode();

		int depthWidth = depthVideoMode.getResolutionX();
		int depthHeight = depthVideoMode.getResolutionY();
		int colorWidth = colorVideoMode.getResolutionX();
		int colorHeight = colorVideoMode.getResolutionY();

		if (depthWidth != colorWidth || depthHeight != colorHeight)
		{
			printf("Warning - expect color and depth to be in same resolution: D: %dx%d, C: %dx%d\n",
				depthWidth, depthHeight,
				colorWidth, colorHeight);
			//return ;
		}
	}

	_streams = new VideoStream*[2];
	_streams[0] = &_depth;
	_streams[1] = &_color;

	// set as the highest resolution 0 for 480x640 
	//register the depth generator with the image generator
	{
		nRetVal = _device.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);

		//nRetVal = _depth.GetAlternativeViewPointCap().getGLModelViewMatrixPoint ( _color );	CHECK_RC_ ( nRetVal, "Getting and setting AlternativeViewPoint failed: " ); 
		// Set Hole Filter
		_device.setDepthColorSyncEnabled(TRUE);
	}//if (_bUseNIRegistration)
	cout << (" Done.\n");

	return;
}//initPlayer()

void CVideoSourceKinect::importYML()
{
	// create and open a character archive for output
#if __linux__
	cv::FileStorage cFSRead( "../data/xtion_intrinsics.yml", cv::FileStorage::READ );
#elif _WIN32
	cv::FileStorage cFSRead ( "..\\data\\xtion_intrinsics.yml", cv::FileStorage::READ );
#endif
	cv::Mat cvmRelativeRotation,cvmRelativeTranslation;
	cFSRead ["cvmRelativeRotation"] >> cvmRelativeRotation;
	cFSRead ["cvmRelativeTranslation"] >> cvmRelativeTranslation;
	cFSRead.release();

	//prepare camera parameters
	cv::Mat  mRTrans = cvmRelativeRotation.t();
	cv::Mat vRT = mRTrans * cvmRelativeTranslation;

	_aR[0] = (float)mRTrans.at<double> ( 0, 0 );
	_aR[1] = (float)mRTrans.at<double> ( 0, 1 );
	_aR[2] = (float)mRTrans.at<double> ( 0, 2 );
	_aR[3] = (float)mRTrans.at<double> ( 1, 0 );
	_aR[4] = (float)mRTrans.at<double> ( 1, 1 );
	_aR[5] = (float)mRTrans.at<double> ( 1, 2 );
	_aR[6] = (float)mRTrans.at<double> ( 2, 0 );
	_aR[7] = (float)mRTrans.at<double> ( 2, 1 );
	_aR[8] = (float)mRTrans.at<double> ( 2, 2 );

	_aRT[0] = (float)vRT.at<double> ( 0 );
	_aRT[1] = (float)vRT.at<double> ( 1 );
	_aRT[2] = (float)vRT.at<double> ( 2 );
}

bool CVideoSourceKinect::getNextFrame(int* pnStatus_){
	if(_bIsSequenceEnds) { *pnStatus_ = PAUSE; _bIsSequenceEnds = false; }
	Status nRetVal = STATUS_OK;

	openni::VideoStream* streams[] = {&_depth, &_color};

	int changedIndex = -1;

	while (nRetVal == STATUS_OK ) //if any one of the frames are not loaded properly, then loop to try to load them
	{
		nRetVal = openni::OpenNI::waitForAnyStream(streams, 2, &changedIndex, 0);
		if ( nRetVal == openni::STATUS_OK)
		{
			switch (changedIndex)
			{
			case 0:
				_depth.readFrame(&_depthFrame); break;
			case 1:
				_color.readFrame(&_colorFrame); break;
			default:
				printf("Error in wait\n");
			}
		}
	}

	//load color image to 
	if( !_colorFrame.isValid() || !_depthFrame.isValid() ) return false;

#ifdef EXPLICIT_LOAD_FRAMES
		const openni::RGB888Pixel* pImageRow = (const openni::RGB888Pixel*)_colorFrame.getData();
		int rowSizeImage = _colorFrame.getStrideInBytes() / sizeof(openni::RGB888Pixel);
		const openni::DepthPixel* pDepthRow = (const openni::DepthPixel*)_depthFrame.getData();
		int rowSizeDepth = _depthFrame.getStrideInBytes() / sizeof(openni::DepthPixel);

		for (int y = 0; y < _colorFrame.getHeight(); ++y) {
			const openni::RGB888Pixel* pImage = pImageRow;
			uchar3* pColorDst = _cvmRGB.ptr<uchar3>(y);
			const openni::DepthPixel* pDepth = pDepthRow;
			float* pDepthDst = _depth_float.ptr<float>(y);

			for ( int x = 0; x < _colorFrame.getWidth(); ++x, ++pImage, ++pDepth ) {
				*pColorDst = *((uchar3*)pImage); pColorDst++;
				*pDepthDst = *((short*)pDepth);  pDepthDst++;
			}
			pImageRow += rowSizeImage;
			pDepthRow += rowSizeDepth;
		}
#else
	cv::Mat cvmRGB(__aRGBH[_uResolution],__aRGBW[_uResolution],CV_8UC3, (unsigned char*)	_colorFrame._getFrame()->data );
	cv::Mat cvmDep(__aDepthH[_uResolution],__aDepthW[_uResolution],CV_16UC1,(unsigned short*) _depthFrame._getFrame()->data );
	_gpu_rgb.upload(cvmRGB);
	_gpu_depth_float.upload(cvmDep);
	_gpu_depth_float.convertTo(_gpu_depth,CV_32FC1, 0.001f);
	
#endif
	switch(_nRawDataProcessingMethod){
	case BIFILTER_IN_ORIGINAL:
		gpuBuildPyramidUseNICVmBiFilteringInOriginalDepth();
		break;
	case BIFILTER_IN_DISPARITY:
		gpuBuildPyramidUseNICVm();
		break;
	case BIFILTER_DYNAMIC:
		gpu_build_pyramid_dynamic_bilatera();
		break;
	default:
		gpuBuildPyramidUseNICVm();
		break;
	}
	_pCurrFrame->initRT();
		
    return true;
}
 
void CVideoSourceKinect::gpuBuildPyramidUseNICVmBiFilteringInOriginalDepth( ){

	if (_bMapUndistortionOn) {
		btl::device::undistortion_depth(_calibXYxZ[_uResolution], _mask[_uResolution], _vRegularLocations[0],
										_vRegularLocations[1] - _vRegularLocations[0], &_gpu_depth);
	}
	//resize the rgbd video source
	if ( fabs( _fScaleRGB-1.f ) > 0.00005f ) { cuda::resize( _gpu_rgb, _gpu_rgb, Size(), _fScaleRGB, _fScaleRGB, INTER_LINEAR ); }
	if ( fabs( _fScaleDepth-1.f ) > 0.00005f ){	cuda::resize( _gpu_depth, _gpu_depth, Size(), _fScaleDepth, _fScaleDepth, INTER_LINEAR ); }
	//undistort
	_pRGBCamera->gpuUndistort(_gpu_rgb, &*_pCurrFrame->_acvgmShrPtrPyrRGBs[0]);//undistort rgb if linear interpolation is used, this may introduce some noises
	_pRGBCamera->gpuUndistort(_gpu_depth, &_gpu_undist_depth);//undistort depth
	//get bw
	cv::cuda::cvtColor(*_pCurrFrame->_acvgmShrPtrPyrRGBs[0],*_pCurrFrame->_acvgmShrPtrPyrBWs[0],cv::COLOR_RGB2GRAY);
	//bilateral filtering (comments off the following three lines to get raw depth map image of kinect)
	btl::device::cuda_bilateral_filtering(_gpu_undist_depth,_fSigmaSpace,_fThresholdDepthInMeter,&*_pCurrFrame->_acvgmShrPtrPyrDepths[0]);
	//_gpu_depth.copyTo(*_pCurrFrame->_acvgmShrPtrPyrDepths[0]);
	//get pts and nls
	btl::device::cuda_unproject_rgb(*_pCurrFrame->_acvgmShrPtrPyrDepths[0],_pRGBCamera->_fFx,_pRGBCamera->_fFy,_pRGBCamera->_u,_pRGBCamera->_v, 0,&*_pCurrFrame->_acvgmShrPtrPyrPts[0] );
	if (_bFast) {
		btl::device::cuda_fast_normal_estimation(*_pCurrFrame->_acvgmShrPtrPyrPts[0], &*_pCurrFrame->_acvgmShrPtrPyrNls[0]);//_vcvgmPyrNls[0]);
	}
	else {
		btl::device::cuda_estimate_normals(*_pCurrFrame->_acvgmShrPtrPyrPts[0], &*_pCurrFrame->_acvgmShrPtrPyrNls[0], &*_pCurrFrame->_acvgmShrPtrPyrReliability[0]);
	}

	//down-sampling
	for( unsigned int i=1; i<_uPyrHeight; i++ )	{
		cv::cuda::pyrDown(*_pCurrFrame->_acvgmShrPtrPyrRGBs[i-1],*_pCurrFrame->_acvgmShrPtrPyrRGBs[i]); //down-sampling rgb
		cv::cuda::cvtColor(*_pCurrFrame->_acvgmShrPtrPyrRGBs[i],*_pCurrFrame->_acvgmShrPtrPyrBWs[i],cv::COLOR_RGB2GRAY); //convert 2 bw
		cv::cuda::resize(*_pCurrFrame->_acvgmShrPtrPyrDepths[i-1],*_pCurrFrame->_acvgmShrPtrPyr32FC1Tmp[i],_pCurrFrame->_acvgmShrPtrPyr32FC1Tmp[i]->size(),0,0,cv::INTER_LINEAR);//down-sampling depth
		//bilateral filtering (comments off the following three lines to get raw depth map image of kinect)
		btl::device::cuda_bilateral_filtering(*_pCurrFrame->_acvgmShrPtrPyr32FC1Tmp[i],_fSigmaSpace,_fThresholdDepthInMeter,&*_pCurrFrame->_acvgmShrPtrPyrDepths[i]);
		//get pts and nls
		btl::device::cuda_unproject_rgb(*_pCurrFrame->_acvgmShrPtrPyrDepths[i],_pRGBCamera->_fFx,_pRGBCamera->_fFy,_pRGBCamera->_u,_pRGBCamera->_v, i,&*_pCurrFrame->_acvgmShrPtrPyrPts[i] );
		if (i!=2) {
			btl::device::cuda_fast_normal_estimation(*_pCurrFrame->_acvgmShrPtrPyrPts[i], &*_pCurrFrame->_acvgmShrPtrPyrNls[i]);
		}
		else {
			btl::device::cuda_estimate_normals(*_pCurrFrame->_acvgmShrPtrPyrPts[i], &*_pCurrFrame->_acvgmShrPtrPyrNls[i], &*_pCurrFrame->_acvgmShrPtrPyrReliability[i]);
		}
	}	

	for( unsigned int i=0; i<_uPyrHeight; i++ )	{
		_pCurrFrame->_acvgmShrPtrPyrRGBs[i]->download(*_pCurrFrame->_acvmShrPtrPyrRGBs[i]);
		_pCurrFrame->_acvgmShrPtrPyrBWs[i]->download(*_pCurrFrame->_acvmShrPtrPyrBWs[i]);
		//_pCurrFrame->_acvgmShrPtrPyrPts[i]->download(*_pCurrFrame->_acvmShrPtrPyrPts[i]);
		//_pCurrFrame->_acvgmShrPtrPyrNls[i]->download(*_pCurrFrame->_acvmShrPtrPyrNls[i]);
		//scale the depth map
		btl::device::cuda_scale_depth(i,_pRGBCamera->_fFx,_pRGBCamera->_fFy,_pRGBCamera->_u,_pRGBCamera->_v,&*_pCurrFrame->_acvgmShrPtrPyrDepths[i]);
		//_pCurrFrame->_acvgmShrPtrPyrDepths[i]->download(*_pCurrFrame->_acvmShrPtrPyrDepths[i]);
	}
	return;
} 
void CVideoSourceKinect::gpuBuildPyramidUseNICVm( ){
	
	if (_bMapUndistortionOn) {
		btl::device::undistortion_depth(_calibXYxZ[_uResolution], _mask[_uResolution], _vRegularLocations[0],
			_vRegularLocations[1] - _vRegularLocations[0], &_gpu_depth);
	}
	
	if ( fabs( _fScaleRGB-1.f ) > 0.00005f ) { cuda::resize( _gpu_rgb, _gpu_rgb, Size(), _fScaleRGB, _fScaleRGB, INTER_LINEAR ); }
	if ( fabs( _fScaleDepth-1.f ) > 0.00005f ){ cuda::resize( _gpu_depth, _gpu_depth, Size(), _fScaleDepth, _fScaleDepth, INTER_LINEAR ); }

	_pRGBCamera->gpuUndistort(_gpu_rgb, &*_pCurrFrame->_acvgmShrPtrPyrRGBs[0]);
	_pRGBCamera->gpuUndistort(_gpu_depth, &_gpu_undist_depth);

	//bilateral filtering (comments off the following three lines to get raw depth map image of kinect)
 	btl::device::cuda_depth2disparity2(_gpu_undist_depth,_fCutOffDistance, &*_pCurrFrame->_acvgmShrPtrPyr32FC1Tmp[0],1.f);//convert depth from mm to m
	btl::device::cuda_bilateral_filtering(*_pCurrFrame->_acvgmShrPtrPyr32FC1Tmp[0],_fSigmaSpace,_fSigmaDisparity,&*_pCurrFrame->_acvgmShrPtrPyrDisparity[0]);
	btl::device::cuda_disparity2depth(*_pCurrFrame->_acvgmShrPtrPyrDisparity[0],&*_pCurrFrame->_acvgmShrPtrPyrDepths[0]);
	//get pts and nls
	btl::device::cuda_unproject_rgb(*_pCurrFrame->_acvgmShrPtrPyrDepths[0],_pRGBCamera->_fFx,_pRGBCamera->_fFy,_pRGBCamera->_u,_pRGBCamera->_v, 0,&*_pCurrFrame->_acvgmShrPtrPyrPts[0] );
	if (_bFast) {
		btl::device::cuda_fast_normal_estimation(*_pCurrFrame->_acvgmShrPtrPyrPts[0], &*_pCurrFrame->_acvgmShrPtrPyrNls[0]);//_vcvgmPyrNls[0]);
	}
	else {
		btl::device::cuda_estimate_normals(*_pCurrFrame->_acvgmShrPtrPyrPts[0], &*_pCurrFrame->_acvgmShrPtrPyrNls[0], &*_pCurrFrame->_acvgmShrPtrPyrReliability[0]);
	}
	//generate black and white
	cv::cuda::cvtColor(*_pCurrFrame->_acvgmShrPtrPyrRGBs[0],*_pCurrFrame->_acvgmShrPtrPyrBWs[0],cv::COLOR_RGB2GRAY);

	//down-sampling
	for( unsigned int i=1; i<_uPyrHeight; i++ )	{
		_pCurrFrame->_acvgmShrPtrPyrRGBs[i]->setTo(0);
		cv::cuda::pyrDown(*_pCurrFrame->_acvgmShrPtrPyrRGBs[i-1],*_pCurrFrame->_acvgmShrPtrPyrRGBs[i]);
		cv::cuda::cvtColor(*_pCurrFrame->_acvgmShrPtrPyrRGBs[i],*_pCurrFrame->_acvgmShrPtrPyrBWs[i],cv::COLOR_RGB2GRAY);
		cv::cuda::resize(*_pCurrFrame->_acvgmShrPtrPyrDisparity[i-1],*_pCurrFrame->_acvgmShrPtrPyr32FC1Tmp[i],_pCurrFrame->_acvgmShrPtrPyr32FC1Tmp[i]->size(),0,0,cv::INTER_LINEAR);
		btl::device::cuda_bilateral_filtering(*_pCurrFrame->_acvgmShrPtrPyr32FC1Tmp[i],_fSigmaSpace,_fSigmaDisparity,&*_pCurrFrame->_acvgmShrPtrPyrDisparity[i]);
		btl::device::cuda_disparity2depth(*_pCurrFrame->_acvgmShrPtrPyrDisparity[i],&*_pCurrFrame->_acvgmShrPtrPyrDepths[i]);
		btl::device::cuda_unproject_rgb(*_pCurrFrame->_acvgmShrPtrPyrDepths[i],_pRGBCamera->_fFx,_pRGBCamera->_fFy,_pRGBCamera->_u,_pRGBCamera->_v, i,&*_pCurrFrame->_acvgmShrPtrPyrPts[i] );
		if (i!=2) {
			btl::device::cuda_fast_normal_estimation(*_pCurrFrame->_acvgmShrPtrPyrPts[i], &*_pCurrFrame->_acvgmShrPtrPyrNls[i]);
		}
		else {
			btl::device::cuda_estimate_normals(*_pCurrFrame->_acvgmShrPtrPyrPts[i], &*_pCurrFrame->_acvgmShrPtrPyrNls[i], &*_pCurrFrame->_acvgmShrPtrPyrReliability[i]);
		}
	}

	for( unsigned int i=0; i<_uPyrHeight; i++ )	{
		_pCurrFrame->_acvgmShrPtrPyrRGBs[i]->download(*_pCurrFrame->_acvmShrPtrPyrRGBs[i]);
		_pCurrFrame->_acvgmShrPtrPyrBWs[i]->download(*_pCurrFrame->_acvmShrPtrPyrBWs[i]);
		//scale the depth map
		btl::device::cuda_scale_depth(i,_pRGBCamera->_fFx,_pRGBCamera->_fFy,_pRGBCamera->_u,_pRGBCamera->_v,&*_pCurrFrame->_acvgmShrPtrPyrDepths[i]);
	}
	return;
}

void CVideoSourceKinect::gpu_build_pyramid_dynamic_bilatera() {

	if (_bMapUndistortionOn) {
		btl::device::undistortion_depth(_calibXYxZ[_uResolution], _mask[_uResolution], _vRegularLocations[0], 
			                            _vRegularLocations[1] - _vRegularLocations[0], &_gpu_depth);
	}

	if (fabs(_fScaleRGB - 1.f) > 0.00005f) { cuda::resize(_gpu_rgb, _gpu_rgb, Size(), _fScaleRGB, _fScaleRGB, INTER_LINEAR); }
	if (fabs(_fScaleDepth - 1.f) > 0.00005f) { cuda::resize(_gpu_depth, _gpu_depth, Size(), _fScaleDepth, _fScaleDepth, INTER_LINEAR); }

	_pRGBCamera->gpuUndistort(_gpu_rgb, &*_pCurrFrame->_acvgmShrPtrPyrRGBs[0]);
	_pRGBCamera->gpuUndistort(_gpu_depth, &_gpu_undist_depth);
	//bilateral filtering (comments off the following three lines to get raw depth map image of kinect)
	btl::device::cuda_bilateral_filtering(_gpu_undist_depth, _fSigmaSpace, &*_pCurrFrame->_acvgmShrPtrPyrDepths[0]);
	//get pts and nls
	btl::device::cuda_unproject_rgb(*_pCurrFrame->_acvgmShrPtrPyrDepths[0], _pRGBCamera->_fFx, _pRGBCamera->_fFy, _pRGBCamera->_u, _pRGBCamera->_v, 0, &*_pCurrFrame->_acvgmShrPtrPyrPts[0]);
	if (_bFast) {
		btl::device::cuda_fast_normal_estimation(*_pCurrFrame->_acvgmShrPtrPyrPts[0], &*_pCurrFrame->_acvgmShrPtrPyrNls[0]);//_vcvgmPyrNls[0]);
	}
	else {
		btl::device::cuda_estimate_normals(*_pCurrFrame->_acvgmShrPtrPyrPts[0], &*_pCurrFrame->_acvgmShrPtrPyrNls[0], &*_pCurrFrame->_acvgmShrPtrPyrReliability[0]);
	}
	//generate black and white
	cv::cuda::cvtColor(*_pCurrFrame->_acvgmShrPtrPyrRGBs[0], *_pCurrFrame->_acvgmShrPtrPyrBWs[0], cv::COLOR_RGB2GRAY);

	//down-sampling
	for (unsigned int i = 1; i < _uPyrHeight; i++) {
		cv::cuda::pyrDown(*_pCurrFrame->_acvgmShrPtrPyrRGBs[i - 1], *_pCurrFrame->_acvgmShrPtrPyrRGBs[i]);
		cv::cuda::cvtColor(*_pCurrFrame->_acvgmShrPtrPyrRGBs[i], *_pCurrFrame->_acvgmShrPtrPyrBWs[i], cv::COLOR_RGB2GRAY);
		cv::cuda::resize(*_pCurrFrame->_acvgmShrPtrPyrDepths[i-1], *_pCurrFrame->_acvgmShrPtrPyr32FC1Tmp[i], _pCurrFrame->_acvgmShrPtrPyr32FC1Tmp[i]->size(), 0, 0, cv::INTER_LINEAR);
		btl::device::cuda_bilateral_filtering(*_pCurrFrame->_acvgmShrPtrPyr32FC1Tmp[i], _fSigmaSpace, &*_pCurrFrame->_acvgmShrPtrPyrDepths[i]);
		btl::device::cuda_unproject_rgb(*_pCurrFrame->_acvgmShrPtrPyrDepths[i], _pRGBCamera->_fFx, _pRGBCamera->_fFy, _pRGBCamera->_u, _pRGBCamera->_v, i, &*_pCurrFrame->_acvgmShrPtrPyrPts[i]);
		if (i != 2) {
			btl::device::cuda_fast_normal_estimation(*_pCurrFrame->_acvgmShrPtrPyrPts[i], &*_pCurrFrame->_acvgmShrPtrPyrNls[i]);
		}
		else {
			btl::device::cuda_estimate_normals(*_pCurrFrame->_acvgmShrPtrPyrPts[i], &*_pCurrFrame->_acvgmShrPtrPyrNls[i], &*_pCurrFrame->_acvgmShrPtrPyrReliability[i]);
		}
	}

	for (unsigned int i = 0; i < _uPyrHeight; i++) {
		_pCurrFrame->_acvgmShrPtrPyrRGBs[i]->download(*_pCurrFrame->_acvmShrPtrPyrRGBs[i]);
		_pCurrFrame->_acvgmShrPtrPyrBWs[i]->download(*_pCurrFrame->_acvmShrPtrPyrBWs[i]);
		//_pCurrFrame->_acvgmShrPtrPyrPts[i]->download(*_pCurrFrame->_acvmShrPtrPyrPts[i]);
		//_pCurrFrame->_acvgmShrPtrPyrNls[i]->download(*_pCurrFrame->_acvmShrPtrPyrNls[i]);
		
		//scale the depth map
		btl::device::cuda_scale_depth(i, _pRGBCamera->_fFx, _pRGBCamera->_fFy, _pRGBCamera->_u, _pRGBCamera->_v, &*_pCurrFrame->_acvgmShrPtrPyrDepths[i]);
		//_pCurrFrame->_acvgmShrPtrPyrDepths[i]->download(*_pCurrFrame->_acvmShrPtrPyrDepths[i]);
	}

	if (false)
	{
		Mat rel, disp; 
		_pCurrFrame->_acvgmShrPtrPyrReliability[2]->download(rel);
		rel.convertTo(disp, CV_8UC1, 255);
		applyColorMap(disp, disp, ColormapTypes::COLORMAP_JET); //img0 can only be grayscale or rgb image
		imwrite("reliability1.png", disp);
	}

	return;
}

Status CVideoSourceKinect::setVideoMode(ushort uResolutionLevel_){
	_uResolution = uResolutionLevel_;
	Status nRetVal = STATUS_OK;
	
#ifdef PRINT_MODE 
	//print supported sensor format
	const openni::Array<openni::VideoMode>& color_modes = _device.getSensorInfo( openni::SENSOR_COLOR )->getSupportedVideoModes();
	cout << " Color" << endl; 
	for (int i=0; i<color_modes.getSize();i++) {
		cout<< "FPS: " << color_modes[i].getFps() << " Pixel format: " << color_modes[i].getPixelFormat() << " X resolution: " << color_modes[i].getResolutionX() << " Y resolution: " << color_modes[i].getResolutionY() << endl;
	}
	const openni::Array<openni::VideoMode>& depth_modes = _device.getSensorInfo( openni::SENSOR_DEPTH )->getSupportedVideoModes();
	cout << " Depth" << endl; 
	for (int i=0; i<depth_modes.getSize();i++) {
		cout<< "FPS: " << depth_modes[i].getFps() << " Pixel format: " << depth_modes[i].getPixelFormat() << " X resolution: " << depth_modes[i].getResolutionX() << " Y resolution: " << depth_modes[i].getResolutionY() << endl;
	}
#endif

	openni::VideoMode depthMode = _depth.getVideoMode();
	openni::VideoMode colorMode = _color.getVideoMode();

	depthMode.setFps(30);
	depthMode.setPixelFormat(openni::PIXEL_FORMAT_DEPTH_1_MM);
	colorMode.setFps(30);
	colorMode.setPixelFormat(openni::PIXEL_FORMAT_RGB888);

	switch(_uResolution){
	case 3:
		depthMode.setResolution(80,60);
		colorMode.setResolution(80,60);
		depthMode.setFps(30);
		colorMode.setFps(30);
		break;
	case 2:
		depthMode.setResolution(160,120);
		colorMode.setResolution(160,120);
		depthMode.setFps(30);
		colorMode.setFps(30);
		nRetVal = _color.setVideoMode(colorMode);
		if ( nRetVal != STATUS_OK)
		{
			printf("SimpleViewer: Couldn't start color stream:\n%s\n", openni::OpenNI::getExtendedError());
			_color.destroy();
			return nRetVal;
		}
		//register the depth generator with the image generator
		{
			nRetVal = _device.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);

			//nRetVal = _depth.GetAlternativeViewPointCap().SetViewPoint ( _color );	CHECK_RC_ ( nRetVal, "Getting and setting AlternativeViewPoint failed: " ); 
			// Set Hole Filter
			nRetVal = _device.setDepthColorSyncEnabled(TRUE);
		}//if (_bUseNIRegistration)
		break;
	case 1:
		depthMode.setResolution(320,240);
		colorMode.setResolution(320,240);
		depthMode.setFps(30);
		colorMode.setFps(30);
		nRetVal = _color.setVideoMode(colorMode);
		if ( nRetVal != STATUS_OK)
		{
			printf("SimpleViewer: Couldn't start color stream:\n%s\n", openni::OpenNI::getExtendedError());
			_color.destroy();
			return nRetVal;
		}
		//register the depth generator with the image generator
		{
			nRetVal = _device.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);

			//nRetVal = _depth.GetAlternativeViewPointCap().SetViewPoint ( _color );	CHECK_RC_ ( nRetVal, "Getting and setting AlternativeViewPoint failed: " ); 
			// Set Hole Filter
			nRetVal = _device.setDepthColorSyncEnabled(TRUE);
		}//if (_bUseNIRegistration)
		break;
	case 0:
		depthMode.setResolution(640,480);
		colorMode.setResolution(640,480);
		depthMode.setFps(30);
		colorMode.setFps(30);
		nRetVal = _color.setVideoMode(colorMode);
		if ( nRetVal != STATUS_OK)
		{
			printf("SimpleViewer: Couldn't start color stream:\n%s\n", openni::OpenNI::getExtendedError());
			_color.destroy();
			return nRetVal;
		}
		//register the depth generator with the image generator
		{
			nRetVal = _device.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);

			//nRetVal = _depth.GetAlternativeViewPointCap().SetViewPoint ( _color );	CHECK_RC_ ( nRetVal, "Getting and setting AlternativeViewPoint failed: " ); 
			// Set Hole Filter
			nRetVal = _device.setDepthColorSyncEnabled(TRUE);
		}//if (_bUseNIRegistration)
		break;
	default:
		depthMode.setResolution(640,480);
		nRetVal = _color.setVideoMode(_colorSensorInfo->getSupportedVideoModes()[10]);
		if ( nRetVal != STATUS_OK)
		{
			printf("SimpleViewer: Couldn't start color stream:\n%s\n", openni::OpenNI::getExtendedError());
			_color.destroy();
			return nRetVal;
		}
		//register the depth generator with the image generator
		{	
		
			nRetVal = _device.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);

			//nRetVal = _depth.GetAlternativeViewPointCap().SetViewPoint ( _color );	CHECK_RC_ ( nRetVal, "Getting and setting AlternativeViewPoint failed: " ); 
			// Set Hole Filter
			nRetVal = _device.setDepthColorSyncEnabled(FALSE);
		}//if (_bUseNIRegistration)
		//colorMode.setResolution(1280,1024);
		//depthMode.setFps(15);
		//colorMode.setFps(30);
		break;
	}

	nRetVal = _depth.setVideoMode(depthMode); 
	if ( nRetVal != STATUS_OK)
	{
		printf("SimpleViewer: Couldn't start depth stream:\n%s\n", openni::OpenNI::getExtendedError());
		_depth.destroy();
		return nRetVal;
	}
	
	return nRetVal;
}


bool CVideoSourceKinect::loadCoefficient(int nResolution_, int size_, Mat* coeff_, Mat* mask_)
{
	{
		ostringstream directory;
		directory << "..//data//" << _serial_number << "//mask_0.png";
		*mask_ = imread(directory.str().c_str(), IMREAD_GRAYSCALE);
	}
	int nRows, nCols;
	if (nResolution_ == 0) {
		nRows = 480;
		nCols = 640;
	}
	else {
		nRows = 240;
		nCols = 320;
	}

	coeff_->create(nCols * nRows, size_, CV_32FC1);
	ostringstream volume_name;

	volume_name << "..//data//" << _serial_number << "//coefficient_" << nResolution_ << ".nii.gz";
		
	nifti_image* ptr_nim = nifti_image_read(volume_name.str().c_str(), 1);

	if (coeff_->type() == CV_32FC1) {
		memcpy((void*)coeff_->data, ptr_nim->data, sizeof(float) * nCols * nRows * size_);
	}

	nifti_image_free(ptr_nim);	
	return true;
}

bool CVideoSourceKinect::loadLocation(vector<float>* pvLocations_)
{
	pvLocations_->clear();
	ostringstream locName;
	locName << "..//data//" << _serial_number << "//RegularLocations.yml";
	if (!std::experimental::filesystem::exists(locName.str().c_str()))
	{
		return false;
	}
	cv::FileStorage storage(locName.str().c_str(), cv::FileStorage::READ);
	storage["locations"] >> *pvLocations_;
	storage.release();

	return true;
}

} //namespace kinect
} //namespace btl
