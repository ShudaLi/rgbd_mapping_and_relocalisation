#define INFO
#define TIMER
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

#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>

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

//camera calibration from a sequence of images
#include <opencv2/cudaarithm.hpp>
#include <OpenNI.h>
#include "Kinect.h"
#include "EigenUtil.hpp"
#include "GLUtil.hpp"
#include <map>
#include "Camera.h"
#include "RGBDFrame.h"
#include "VideoSourceKinect.hpp"
#include "pcl/internal.h"
#include "CubicGrids.h"
#include "KinfuTracker.h"

//Qt
#include <QGLViewer/qglviewer.h>
#include "Data4Viewer.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace qglviewer;

void convert(const Eigen::Affine3f& eiM_, Mat* pM_){
	pM_->create(4, 4, CV_32FC1);
	for (int r = 0; r < 4; r++)
		for (int c = 0; c < 4; c++)
		{
		pM_->at<float>(r, c) = eiM_(r, c);
		}

	return;
}

void convert(const Eigen::Matrix4f& eiM_, Mat* pM_){
	pM_->create(4, 4, CV_32FC1);
	for (int r = 0; r < 4; r++)
		for (int c = 0; c < 4; c++)
		{
		pM_->at<float>(r, c) = eiM_(r, c);
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

void convert(const Mat& M_, Eigen::Matrix4f* peiM_){

	for (int r = 0; r < 4; r++)
		for (int c = 0; c < 4; c++)
		{
		(*peiM_)(r, c) = M_.at<float>(r, c);
		}

	return;
}


CData4Viewer::CData4Viewer(){
	_uResolution = 0;
	_uPyrHeight = 3;
	_eivCw = Eigen::Vector3f(0.f,0.f,0.f);
	_bUseNIRegistration = true;
	_vXYZCubicResolution.push_back(512); _vXYZCubicResolution.push_back(512); _vXYZCubicResolution.push_back(512);
	_bRenderReference = true;
	_fVolumeSize = 3.f;
	_nMode = 3;//PLAYING_BACK
	_bRepeat = false;// repeatedly play the sequence 
	_nStatus = 01;//1 restart; 2 //recording continue 3://pause 4://dump
	_bDisplayImage = false;
	_bLightOn = false;
	_bCapture = false;
	_bContinuous = false;
	_bTrackOnly = false;
	_bShowSurfaces = false;
        _bShowCamera = false;
        _bShowVoxels = false;
        _bShowMarkers = false;
        _bShowVisualRays = false;
        _bIsCurrFrameOn = false;
	_nSequenceIdx = 0;
	_nMethodIdx = 0;
	_nRound = 0;
	_bRenderSphere = true;
	_bInitialized = false;
	_bLoadVolume = false;
	_strStage = string("Tracking_n_Mapping");
	_nVoxelLevel = 0;
	_bCameraFollow = true;
	_prj_c_f_w.setIdentity();
}
CData4Viewer::~CData4Viewer()
{
	//_pGL->destroyVBOsPBOs();
}

void CData4Viewer::init()
{
	if( _bInitialized ) return;
	GLenum eError = glewInit(); 
	if (GLEW_OK != eError){
		PRINTSTR("glewInit() error.");
		PRINT( glewGetErrorString(eError) );
	}
	btl::gl_util::CGLUtil::initCuda();
	btl::gl_util::CGLUtil::setCudaDeviceForGLInteroperation();//initialize before using any cuda component
	loadFromYml();
	boost::filesystem::remove_all(_result_folder.c_str());
	boost::filesystem::path dir(_result_folder.c_str());
	if(boost::filesystem::create_directories(dir)) { 
		std::cout << "Success" << "\n";
	}
	reset();

	_bInitialized = true;
	_vError.clear();
	return;
}//init()


void CData4Viewer::loadFromYml(){
	cv::FileStorage cFSRead("C:\\csxsl\\src\\opencv-old-old\\btl\\rgbd_fusion_qglviewer\\RGBDFusionQGlviewerControl.yml", cv::FileStorage::READ );
	//cv::FileStorage cFSRead("C:\\csxsl\\src\\opencv-old-old\\btl\\rgbd_fusion_baselin_qglviewer\\RGBDFusionBaselineQGlviewerControl.yml", cv::FileStorage::READ);
	if (!cFSRead.isOpened()) {
		std::cout << "Load RGBDFusionQGlviewerControl failed.";
		return;
	}
	cFSRead["uResolution"] >> _uResolution;
	cFSRead["uPyrHeight"] >> _uPyrHeight;
	cFSRead["bUseNIRegistration"] >> _bUseNIRegistration;
	cFSRead["uCubicGridResolution"] >> _vXYZCubicResolution;
	cFSRead["Global_folder"] >> _global_model_folder;
	cFSRead["Ransac_iterations_tracking"] >> _nRansacIterationasTracking;
	cFSRead["Ransac_iterations_relocalisation"] >> _nRansacIterationasRelocalisation;

	cFSRead["fVolumeSize"] >> _fVolumeSize;
	cFSRead["InitialCamPos"] >> _vInitial_Cam_Pos;
	_eivCw = Eigen::Vector3f(_vInitial_Cam_Pos[0],_vInitial_Cam_Pos[1],_vInitial_Cam_Pos[2]);
	cFSRead["TrainingDataPath"] >> _v_training_data_path;
	cFSRead["CamParamPathFileName"] >> _cam_param_path;
	cFSRead["Feature_scales"] >> _nFeatureScale;
	cFSRead["SyncOffset"] >> _nOffset;
	cFSRead["StartIdx"] >> _nStartIdx;
	cFSRead["Step"] >> _nStep;
	string _strStagePrev = _strStage;
	cFSRead["Stage"] >> _strStage; 
	if (_strStage.compare(_strStagePrev))//if  _strStage != _strStagePrev
		_nRound = 0;
	cFSRead["Result_folder"] >> _result_folder;

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
	cFSRead["Weight"] >> _fWeight;
	cFSRead["bCameraPathOn"] >> _bIsCameraPathOn;
	cFSRead["LoadVolume"] >> _bLoadVolume;
	cFSRead["KNN_tracking"] >> _KNN_tracking;
	cFSRead["KNN_relocalisation"] >> _KNN_relocalisation;

	cFSRead.release();

	return;
}

void CData4Viewer::reset(){
	_nStatus = 9;
	//store viewing status
	if(_pGL.get()){
		_eimModelViewGL = _pGL->_eimModelViewGL;
		_dZoom = _pGL->_dZoom;
		_dXAngle = _pGL->_dXAngle;
		_dYAngle = _pGL->_dYAngle;
	}
	_pGL.reset();
	_pGL.reset( new btl::gl_util::CGLUtil(_uResolution,_uPyrHeight,btl::utility::BTL_GL) );

	_pGL->_bDisplayCamera = _bDisplayImage;
	_pGL->_bEnableLighting = _bLightOn;
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

	return;
}

void CData4Viewer::recordTrackerCamPairs(){

	return;
}

void CData4Viewer::storeResultPose()
{
	if (_bStorePoses) {
		Eigen::Affine3f prj_c_f_w;
		_pTracker->getCurrentProjectionMatrix(&prj_c_f_w);
		Eigen::Affine3f prj_w_f_c = prj_c_f_w.inverse();
		Eigen::Affine3f prj_pov_f_c = _prj_pov_f_w * prj_w_f_c;

		ostringstream convert;   // stream used for the conversion
		convert << setfill('0') << setw(6) << _nIdx;      // insert the textual representation of 'Number' in the characters in the stream
		//convert n to string
		string path = _result_folder + "\\frame-" + convert.str() + ".pose.txt";
		//store prj
		ofstream out;
		out.open(path.c_str());
		out << fixed << std::setprecision(std::numeric_limits<float>::digits10 + 1);
		for (int r = 0; r < 4; r++){
			out << prj_pov_f_c(r, 0) << "\t" << prj_pov_f_c(r, 1) << "\t" << prj_pov_f_c(r, 2) << "\t" << prj_pov_f_c(r, 3) << endl;
		}
		out.close();
	}
}

void CData4Viewer::drawGlobalView()
{
	if (_bShowVisualRays){
		glEnable(GL_BLEND);
		glDisable(GL_DEPTH_TEST);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
	else{
		glDisable(GL_BLEND);
		glEnable(GL_DEPTH_TEST);
	}

	_pKinect->_pRGBCamera->setGLProjectionMatrix(0.1f, 100.f);
	if (_bCameraFollow){
		_pTracker->getCurrentProjectionMatrix(&_prj_c_f_w);
	}

	glMatrixMode(GL_MODELVIEW);
	glMultMatrixf(_prj_c_f_w.data());//times with manipulation matrix

	if (_bShowSurfaces)	{
		_pVirtualGlobalView->assignRTfromGL();
		_pCubicGrids->rayCast(&*_pVirtualGlobalView,true,_bCapture); //if capturing is on, fineCast is off
		_pVirtualGlobalView->gpuRender3DPts(_pGL.get(), 0);
	}
	else{
		//_pVirtualGlobalView->assignRTfromGL();
		//_pCubicGridsMoved->gpuRaycast(&*_pVirtualGlobalView, true, _bCapture); //if capturing is on, fineCast is off
		//_pVirtualGlobalView->gpuRender3DPts(_pGL.get(), 0);
		//PRINT(*_pVirtualGlobalView->_acvmShrPtrPyrPts[0]);
	}
	
	if (_bIsCurrFrameOn){
		_pKinect->_pCurrFrame->gpuRender3DPts(_pGL.get(), _pGL->_usLevel);
	}

	if(_bShowVoxels) {
		_pCubicGrids->renderOccupiedVoxels(_pGL.get(),_nVoxelLevel);
	}
	if(_bShowMarkers) {
		_pTracker->displayAllGlobalFeatures(_nVoxelLevel, _bRenderSphere);
	}

	if (_bRenderReference)	{
		_pGL->renderAxisGL();
		_pGL->renderPatternGL(.1f,20,20);
		_pGL->renderPatternGL(1.f,10,10);
		_pCubicGrids->renderBoxGL();
		//_pGL->renderVoxelGL(_fVolumeSize);
	}
	if(_bShowCamera) {
		_pKinect->_pCurrFrame->copyTo( &*_pCameraView2 );
		_pTracker->displayFeatures2D(_nVoxelLevel, _pCameraView2.get());
		
		if (_bShowVisualRays){
			_pTracker->displayVisualRays(_pGL.get(), _nVoxelLevel, _bRenderSphere); //, _pCameraView2.get(), .2f, _pGL->_usLevel 
		}
		else{
			_pTracker->displayGlobalRelocalization();
		}

		float aColor[3] = { 0.f, 0.f, 1.f };
		Eigen::Affine3f prj_wtc; _pTracker->getCurrentProjectionMatrix(&prj_wtc);
		_pCameraView2->setRTFromPrjCfW(prj_wtc);
		_pCameraView2->renderCameraInWorld(_pGL.get(), true, aColor, _pGL->_bDisplayCamera, .2f, _pGL->_usLevel);//refine pose
	}
	
	
	if( _bIsCameraPathOn ){
		_pTracker->displayCameraPath();
		_pTracker->displayCameraPathReloc();
	}
	_pTracker->displayTrackedKeyPoints();
	//PRINTSTR("drawGlobalView");
	return;
}

void CData4Viewer::drawCameraView(qglviewer::Camera* pCamera_)
{
	_pKinect->_pRGBCamera->setGLProjectionMatrix( 0.1f,100.f);
	
	glMatrixMode(GL_MODELVIEW);
	Eigen::Affine3f prj_w_t_c; _pTracker->getCurrentProjectionMatrix(&prj_w_t_c);
	Eigen::Affine3f init; init.setIdentity(); init(1, 1) = -1.f; init(2, 2) = -1.f;// rotate the default opengl camera orientation to make it facing positive z
	glLoadMatrixf(init.data());
	glMultMatrixf(prj_w_t_c.data());

	//if(_bShowCamera) {
	//	_pTracker->displayGlobalRelocalization();
	//}
	//if(_bShowMarkers) {
	//	_pTracker->displayAllGlobalFeatures(_nVoxelLevel,_bRenderSphere);
	//}

	_pVirtualCameraView->assignRTfromGL();
	_pCubicGrids->rayCast(&*_pVirtualCameraView,true,_bCapture); //get virtual frame
	bool bLightingStatus = _pGL->_bEnableLighting;
	_pGL->_bEnableLighting = true;
	_pVirtualCameraView->gpuRender3DPts(_pGL.get(),_pGL->_usLevel);
	_pGL->_bEnableLighting = bLightingStatus;
	//PRINTSTR("drawCameraView");
	return;	
}

void CData4Viewer::drawRGBView()
{
	_pKinect->_pRGBCamera->setGLProjectionMatrix( 0.1f,100.f);

	glMatrixMode ( GL_MODELVIEW );
	Eigen::Affine3f tmp; tmp.setIdentity();
	Matrix4f mv = btl::utility::setModelViewGLfromPrj(tmp); //mv transform X_m to X_w i.e. model coordinate to world coordinate
	glLoadMatrixf( mv.data() );
	_pKinect->_pRGBCamera->renderCameraInLocal(*_pKinect->_pCurrFrame->_acvgmShrPtrPyrRGBs[_pGL->_usLevel],  _pGL.get(),false, NULL, 0.2f, true ); //render in model coordinate
	//PRINTSTR("drawRGBView");
    return;
}

void CData4Viewer::drawDepthView(qglviewer::Camera* pCamera_)
{
	_pKinect->_pRGBCamera->setGLProjectionMatrix( 0.01f,20.f);

	_pDepth->setRTw( Matrix3f::Identity(),Vector3f(0,0,0) );
	//_pDepth->gpuTransformToWorldCVCV();

	Matrix4f mModelView;
	_pDepth->getGLModelViewMatrix(&mModelView);
	Matrix4d mTmp = mModelView.cast<double>(); 
	pCamera_->setFromModelViewMatrix( mTmp.data() );

	_pDepth->gpuRender3DPts(_pGL.get(), _pGL->_usLevel);
	//PRINTSTR("drawDepthView");
	return;
}

void CData4Viewer::setAsPrevPos()
{
	_pTracker->getPrevView(&_pGL->_eimModelViewGL);
}

void CData4Viewer::exportGlobalModel()
{
	PRINTSTR("Export global model...")
	boost::filesystem::path dir(_global_model_folder.c_str());
	if(boost::filesystem::create_directories(dir)) {
		std::cout << "Success" << "\n";
	}

	//mkdir(_global_model_folder.c_str());
	_pCubicGrids->storeNIFTI(_global_model_folder);
	_pTracker->storeGlobalFeaturesAndCameraPoses(_global_model_folder);
	exportRelativeGTworld2Userworld();
	PRINTSTR("Done.");
	return;
}

void CData4Viewer::exportRelativeGTworld2Userworld(){
	boost::filesystem::path dir(_global_model_folder.c_str());
	if (!boost::filesystem::create_directories(dir)) {
		std::cout << "Failure - create folder fails. " << _global_model_folder << " \n";
	}
	//store relative motion between GT and self-defined system.
	_prj_w_f_pov;
	_prj_pov_f_w;

	vector<Mat> vPoses;
	{
		Mat pos; convert(_prj_w_f_pov, &pos);
		vPoses.push_back(pos);
	}
	{
		Mat pos; convert(_prj_pov_f_w, &pos);
		vPoses.push_back(pos);
	}
	string camera_poses = _global_model_folder + string("Relative.yml");
	cv::FileStorage storagePose(camera_poses, cv::FileStorage::WRITE);
	storagePose << "Relative" << vPoses;

	storagePose.release();
}

void CData4Viewer::importRelativeGTworld2Userworld(){
	//load relative motion between GT and self-defined system.
	vector<Mat> vPoses;
	string camera_poses = _global_model_folder + string("Relative.yml");
	cv::FileStorage storagePose(camera_poses, cv::FileStorage::READ);
	storagePose["Relative"] >> vPoses;
	storagePose.release();
	{
		convert(vPoses[0], &_prj_w_f_pov);
	}
	{
		convert(vPoses[1], &_prj_pov_f_w);
	}
	return;
}


void CData4Viewer::importGlobalModel()
{
	PRINTSTR("Loading a global model...")
	_pCubicGrids->loadNIFTI(_global_model_folder);
	_pTracker->loadGlobalFeaturesAndCameraPoses(_global_model_folder);
}

















