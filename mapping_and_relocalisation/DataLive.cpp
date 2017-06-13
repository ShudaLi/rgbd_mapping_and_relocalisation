#define TIMER
#define _USE_MATH_DEFINES
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#ifdef __gnu_linux__
#include <sys/types.h>
#include <sys/stat.h>
#elif _WIN32
#include <direct.h>
#else 
#error "OS not supported!"
#endif
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <memory>

//#include <opencv2/core/utility.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>

#include "Utility.hpp"

#include <opencv2/cudaarithm.hpp>
#include <OpenNI.h>
#include "Kinect.h"
#include <sophus/se3.hpp>
#include "EigenUtil.hpp"
#include "GLUtil.hpp"
#include "pcl/internal.h"
#include <map>
#include "Camera.h"
#include "RGBDFrame.h"
#include "VideoSourceKinect.hpp"
#include "CubicGrids.h"
#include "KinfuTracker.h"

//Qt
#include <QGLViewer/qglviewer.h>
#include "Data4Viewer.h"
#include "DataLive.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace qglviewer;

CDataLive::CDataLive()
:CData4Viewer(){
	_bUseNIRegistration = true;
	_fVolumeSize = 3.f;
	_nMode = 3;//PLAYING_BACK
	_oniFileName = std::string("x.oni"); // the openni file 
	_bRepeat = false;// repeatedly play the sequence 
	_nStatus = 01;//1 restart; 2 //recording continue 3://pause 4://dump
	_bDisplayImage = false;
	_bLightOn = false;
	_bCapture = false;
	_bTrackOnly = false;
	_bShowSurfaces = false;
	_bIsCameraPathOn = false;
	_bRenderSphere = false;
	_nSequenceIdx = 0;
	_nMethodIdx = 0;
	_bInitialized = false;
	_nVoxelLevel = 0;
}

void CDataLive::loadFromYml(){
	//cout << "DataLive::loadFromYml()"<< endl;
#ifdef __gnu_linux__
	cv::FileStorage cFSRead ( "../mapping_and_relocalisation/MappingAndRelocalisationControlUbuntu.yml", cv::FileStorage::READ );
#elif _WIN32
	cv::FileStorage cFSRead ( "..\\mapping_and_relocalisation\\MappingAndRelocalisationControl.yml", cv::FileStorage::READ );
#endif 
	if (!cFSRead.isOpened()) {
		cout << "Load MappingAndRelocalisationControl failed." <<endl;
		return;
	}
	cFSRead["uResolution"] >> _uResolution;
	cFSRead["uPyrHeight"] >> _uPyrHeight;
	cFSRead["bUseNIRegistration"] >> _bUseNIRegistration;
	cFSRead["uCubicGridResolution"] >> _vXYZCubicResolution;

	cFSRead["fVolumeSize"] >> _fVolumeSize;
	cFSRead["InitialCamPos"] >> _vInitial_Cam_Pos;
	_eivCw = Eigen::Vector3f(_vInitial_Cam_Pos[0],_vInitial_Cam_Pos[1],_vInitial_Cam_Pos[2]);
	cFSRead["CamParamPathFileName"] >> _cam_param_path;
	cFSRead["Feature_scales"] >> _nFeatureScale;
	cFSRead["Stage"] >> _strStage;
	cFSRead["Result_folder"] >> _result_folder; 
	cFSRead["Global_folder"] >> _global_model_folder;
	//rendering
	cFSRead["bDisplayImage"] >> _bDisplayImage;
	cFSRead["bLightOn"] >> _bLightOn;
	cFSRead["nMode"] >> _nMode;//1 kinect; 2 recorder; 3 player
	cFSRead["bRepeat"] >> _bRepeat;
	cFSRead["nStatus"] >> _nStatus;
	cFSRead["Pose_Estimation"] >> _vstrPoseEstimationMethod;
	cFSRead["Matching_Method"] >> _vstrMatchingMethod;
	cFSRead["ICP_Method"] >> _vstrICPMethod;
	cFSRead["Feature_Name"] >> _vstrFeatureName;

	cFSRead["bCameraPathOn"] >> _bIsCameraPathOn;
	cFSRead["oniFile"] >> _oniFileName;
	cFSRead["LoadVolume"] >> _bLoadVolume;
	cFSRead["KNN_tracking"] >> _KNN_tracking;
	cFSRead["KNN_relocalisation"] >> _KNN_relocalisation;
	cFSRead["Ransac_iterations_tracking"] >> _nRansacIterationasTracking;
	cFSRead["Ransac_iterations_relocalisation"] >> _nRansacIterationasRelocalisation;

	cFSRead.release();
}

void CDataLive::reset(){
	//for testing different oni files
	if (_nMethodIdx <  _vstrPoseEstimationMethod.size() )	{
		_strPoseEstimationMethod = _vstrPoseEstimationMethod[_nMethodIdx];
		_strICPMethod = _vstrICPMethod[_nMethodIdx];
		_strMatchingMethod = _vstrMatchingMethod[_nMethodIdx];
		_strFeatureName = _vstrFeatureName[_nMethodIdx];
	}

	_nStatus = 9;
	//store viewing status
	if(_pGL.get()){
		_eimModelViewGL = _pGL->_eimModelViewGL;
		_dZoom = _pGL->_dZoom;
		_dXAngle = _pGL->_dXAngle;
		_dYAngle = _pGL->_dYAngle;
	}
	_pGL.reset();
	_pGL.reset( new btl::gl_util::CGLUtil(_uResolution,_uPyrHeight) );

	_pGL->_bDisplayCamera = _bDisplayImage;
	_pGL->_bEnableLighting = _bLightOn;
	_pGL->_bRenderReference = _bRenderReference;
	//_pGL->clearColorDepth();
	_pGL->init();
	_pGL->constructVBOsPBOs();
	{
		//have to be after _pGL->init()
		//recover viewing status
		_pGL->_eimModelViewGL = _eimModelViewGL ;
		_pGL->_dZoom = _dZoom;
		_pGL->_dXAngle = _dXAngle;
		_pGL->_dYAngle = _dYAngle;	
	}
	//reset shared pointer, notice that the relationship makes a lot of sense
	_pTracker.reset();	_pCubicGrids.reset();	_pDepth.reset(); _pVirtualGlobalView.reset(); _pVirtualCameraView.reset(); _pKinect.reset();

	_pKinect.reset( new btl::kinect::CVideoSourceKinect(_uResolution,_uPyrHeight,_bUseNIRegistration,_eivCw,_cam_param_path) );
	switch(_nMode)
	{
		using namespace btl::kinect;
	case CVideoSourceKinect::SIMPLE_CAPTURING: //the simple capturing mode of the rgbd camera
		_pKinect->initKinect();
		break;
	case CVideoSourceKinect::RECORDING: //record the captured sequence from the camera
		_pKinect->setDumpFileName(_oniFileName);
		_pKinect->initRecorder(_oniFileName);
		break;
	case CVideoSourceKinect::PLAYING_BACK: //replay from files
		_pKinect->initPlayer(_oniFileName);
		break;
	}

	_pDepth            .reset(new btl::kinect::CRGBDFrame(_pKinect->_pRGBCamera.get(),_uResolution,_uPyrHeight,_eivCw));
	_pVirtualGlobalView.reset(new btl::kinect::CRGBDFrame(_pKinect->_pRGBCamera.get(),0,_uPyrHeight,_eivCw));
	_pVirtualCameraView.reset(new btl::kinect::CRGBDFrame(_pKinect->_pRGBCamera.get(),_uResolution,_uPyrHeight,_eivCw));
	_pCameraView2.reset(new btl::kinect::CRGBDFrame(_pKinect->_pRGBCamera.get(), _uResolution, _uPyrHeight, _eivCw));

	_pKinect->getNextFrame(&_nStatus);

	//initialize the cubic grids
	_pCubicGrids.reset(new btl::geometry::CCubicGrids(_vXYZCubicResolution[0], _vXYZCubicResolution[1], _vXYZCubicResolution[2], _fVolumeSize, _nFeatureScale, _strFeatureName)); //_uCubicGridResolution by default is 512, it is loaded from Control.yml
	//initialize the tracker
	_pTracker.reset(new btl::geometry::CKinFuTracker(_pKinect->_pCurrFrame.get(), _pCubicGrids, _uResolution, _uPyrHeight, _strPoseEstimationMethod, _strICPMethod,_strMatchingMethod));

	_pTracker->_bLoadVolume = _bLoadVolume;
	_pTracker->_bRelocWRTVolume = _bLoadVolume;
	if (!_strStage.compare("Tracking_n_Mapping")){
		_pTracker->_bLoadVolume = _bLoadVolume;
		_pTracker->_bRelocWRTVolume = _bLoadVolume;
		_pTracker->_nStage = btl::Tracking_n_Mapping;
	}
	else if (!_strStage.compare("Relocalisation_Only")){
		_pTracker->_bLoadVolume = true;
		_pTracker->_bRelocWRTVolume = true;
		_pTracker->_nStage = btl::Relocalisation_Only;
	}

	_pTracker->setModelFolder(_global_model_folder);
	_pTracker->_fWeight = _fWeight;
	_pTracker->_nRansacIterationasTracking = _nRansacIterationasTracking;
	_pTracker->_nRansacIterationasRelocalisation = _nRansacIterationasRelocalisation;
	_pTracker->_K_tracking = _KNN_tracking;
	_pTracker->_K_relocalisation = _KNN_relocalisation;
	bool bIsSuccessful = _pTracker->init(_pKinect->_pCurrFrame.get());
	if (!_strStage.compare("Tracking_n_Mapping") ){
		while (!bIsSuccessful){
			_pKinect->getNextFrame(&_nStatus);
			bIsSuccessful = _pTracker->init(_pKinect->_pCurrFrame.get());
		}
	}

	_bCapture = true;
	return;
}

void CDataLive::updatePF(){
	//int64 A = getTickCount();
	using btl::kinect::CVideoSourceKinect;

	if ( _bCapture ) {
		//load data from video source and model
		if( !_pKinect->getNextFrame(&_nStatus) ) return;
		_pKinect->_pCurrFrame->copyTo( &*_pDepth );
		_pKinect->_pCurrFrame->copyTo(&*_pDepth);
		_pKinect->_pCurrFrame->convert2NormalMap();

		cuda::minMax(*_pKinect->_pCurrFrame->_acvgmShrPtrPyrDepths[0], &_mi, &_ma);
		_pKinect->_pCurrFrame->convertDepth2Gray(_ma);

		if (!_strStage.compare("Tracking_n_Mapping")){
			_pTracker->_nStage = btl::Tracking_n_Mapping;
			_pTracker->_bTrackingOnly = _bTrackOnly;
			_pTracker->tracking(&*_pKinect->_pCurrFrame);
			//if (true) storeResultPose();
		}
	}//if( _bCapture )
	return;
}
















