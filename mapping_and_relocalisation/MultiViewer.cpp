/**
* @file MultiViewer.cpp
* @brief gui multiple viewer class
* @author Shuda Li<lishuda1980@gmail.com>
* @version 1.0
* @date 2015-07-15
*/
#define _USE_MATH_DEFINES
#define INFO
#define TIMER
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <string>
#include <vector>
#include <memory>

#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "Utility.hpp"

#include "QGLViewer/manipulatedCameraFrame.h"

//camera calibration from a sequence of images
#include <opencv2/cudaarithm.hpp>
#include <OpenNI.h>
#include "Kinect.h"
#include "EigenUtil.hpp"
#include <sophus/se3.hpp>
#include "GLUtil.hpp"
#include <map>
#include "Camera.h"
#include "RGBDFrame.h"
#include "VideoSourceKinect.hpp"
#include "pcl/internal.h"
#include "CubicGrids.h"
#include "KinfuTracker.h"


//Qt
#include <QResizeEvent>
#include <QGLViewer/qglviewer.h>
#include "Data4Viewer.h"
#include "DataLive.h"
#include "MultiViewer.h"
#include <QCoreApplication>
using namespace qglviewer;
using namespace std;

CMultiViewer::CMultiViewer(string strName_, CDataLive::tp_shared_ptr pData, QWidget* parent, const QGLWidget* shareWidget)
:QGLViewer(parent, shareWidget)
{
	_pData = pData;
	_strViewer = strName_;
	_bShowText = true;

	// Forbid rotation
	if(!_strViewer.compare("global_view") ){
		setAxisIsDrawn(false);
		setFPSIsDisplayed();
		setGridIsDrawn(false);
	}
	else
	{
		setAxisIsDrawn(false);
		setGridIsDrawn(false);
		WorldConstraint* constraint = new WorldConstraint();
		constraint->setRotationConstraintType(AxisPlaneConstraint::FORBIDDEN);
		constraint->setTranslationConstraintType(AxisPlaneConstraint::FORBIDDEN);
		camera()->frame()->setConstraint(constraint);
	}
}
CMultiViewer::~CMultiViewer()
{
}

void CMultiViewer::draw()
{
	if(!_strViewer.compare("global_view"))	{
		_pData->updatePF();
		_pData->drawGlobalView();
		if(_bShowText){
			float aColor[4] = {0.f,0.f,1.f,1.f};	glColor4fv(aColor);
			renderText(5, 40, QString("global view"), QFont("Arial", 13, QFont::Bold));
	
			if (_pData->_pKinect->_bMapUndistortionOn) {
				glColor3f(1.f, 0.f, 0.f);
				renderText(330, 40, QString("Undistortion ON"), QFont("Arial", 13, QFont::Bold));
			}

			if( _pData->isCapturing() )	{
				if (!_pData->_bTrackOnly){
					float aColor[4] = {1.f,0.f,0.f,1.f}; glColor4fv(aColor);
					renderText(230,20, QString("mapping"), QFont("Arial", 13, QFont::Normal));
				}
				else{
					float aColor[4] = {1.f,0.f,0.f,1.f}; glColor4fv(aColor);
					renderText(230,20, QString("tracking"), QFont("Arial", 13, QFont::Normal));
				}
			}
		}
	}
	else if(!_strViewer.compare("camera_view"))	{
		_pData->drawCameraView(camera());
		float aColor[4] = { 0.f, 0.f, 1.f, 1.f };	glColor4fv(aColor);
		renderText(5, 40, QString("predicted depth"), QFont("Arial", 13, QFont::Bold));
	}
	else if(!_strViewer.compare("rgb_view"))	{
		_pData->drawRGBView();
		float aColor[4] = { 0.f, 0.f, 1.f, 1.f };	glColor4fv(aColor);
		renderText(5, 40, QString("RGB"), QFont("Arial", 13, QFont::Bold));
	}
	else if(!_strViewer.compare("depth_view"))	{
		_pData->drawDepthView(camera());
		float aColor[4] = { 0.f, 0.f, 1.f, 1.f };	glColor4fv(aColor);
		//renderText(5, 40, QString("raw depth"), QFont("Arial", 13, QFont::Bold));
	}
	return;
}

void CMultiViewer::init()
{
	// Restore previous viewer state.
	//restoreStateFromFile();
	// Opens help window
	//help();
	startAnimation();
	//
	_pData->init();
	//reset();
}//init()

QString CMultiViewer::helpString() const
{
	QString text("<h2>S i m p l e V i e w e r</h2>");
	text += "Use the mouse to move the camera around the object. ";
	text += "You can respectively revolve around, zoom and translate with the three mouse buttons. ";
	text += "Left and middle buttons pressed together rotate around the camera view direction axis<br><br>";
	text += "Pressing <b>Alt</b> and one of the function keys (<b>F1</b>..<b>F12</b>) defines a camera keyFrame. ";
	text += "Simply press the function key again to restore it. Several keyFrames define a ";
	text += "camera path. Paths are saved when you quit the application and restored at next start.<br><br>";
	text += "Press <b>F</b> to display the frame rate, <b>A</b> for the world axis, ";
	text += "<b>Alt+Return</b> for full screen mode and <b>Control+S</b> to save a snapshot. ";
	text += "See the <b>Keyboard</b> tab in this window for a complete shortcut list.<br><br>";
	text += "Double clicks automates single click actions: A left button double click aligns the closer axis with the camera (if close enough). ";
	text += "A middle button double click fits the zoom of the camera and the right button re-centers the scene.<br><br>";
	text += "A left button double click while holding right button pressed defines the camera <i>Revolve Around Point</i>. ";
	text += "See the <b>Mouse</b> tab and the documentation web pages for details.<br><br>";
	text += "Press <b>Escape</b> to exit the viewer.";
	return text;
}

void CMultiViewer::keyPressEvent(QKeyEvent *pEvent_)
{
	using namespace btl::kinect;
	using namespace Eigen;
	// Defines the Alt+R shortcut.
	if (pEvent_->key() == Qt::Key_0)
	{
		_pData->_bCameraFollow = !_pData->_bCameraFollow;
		camera()->setPosition(qglviewer::Vec(0, 0, 0));
		camera()->setUpVector(qglviewer::Vec(0, -1, 0));
		camera()->setViewDirection(qglviewer::Vec(0, 0, 1));
		//updateGL(); // Refres

#ifdef INACCURATE_METHOD_0
		Affine3f prj_w_t_c; _pData->_pTracker->getCurrentProjectionMatrix(&prj_w_t_c);
		Vector3f pos = -prj_w_t_c.linear().transpose() * prj_w_t_c.translation();
		Vector3f upv = prj_w_t_c.linear()*Vector3f(0, -1, 0); //up vector
		Vector3f vdr = prj_w_t_c.linear()*Vector3f(0, 0, 1); //viewing direction

		camera()->setPosition(qglviewer::Vec(pos(0), pos(1), pos(2)));
		camera()->setUpVector(qglviewer::Vec(upv(0), upv(1), upv(2)));
		camera()->setViewDirection(qglviewer::Vec(vdr(0), vdr(1), vdr(2)));
		//updateGL(); // Refresh display
#endif

#ifdef INACCURATE_METHOD_1
		glMatrixMode(GL_MODELVIEW);
		Affine3f init; init.setIdentity(); init(1, 1) = -1.f; init(2, 2) = -1.f;// rotate the default opengl camera orientation to make it facing positive z
		Affine3f prj_w_t_c; _pData->_pTracker->getCurrentProjectionMatrix(&prj_w_t_c);
		prj_w_t_c = init * prj_w_t_c; //the order matters
		glLoadMatrixf(prj_w_t_c.data());//times with manipulation matrix

		GLdouble mvm[16];
		glGetDoublev(GL_MODELVIEW_MATRIX, mvm);
		camera()->setFromModelViewMatrix(mvm);
		//updateGL(); // Refresh display
#endif
	}
	else if (pEvent_->key() == Qt::Key_BracketLeft)
	{
		//_idx_view--;
		//Affine3f prj_w_t_c = _pData->getTrackerPtr()->getView(&_idx_view);
		//Affine3f init; init.setIdentity(); init(1, 1) = -1.f; init(2, 2) = -1.f;// rotate the default opengl camera orientation to make it facing positive z
		//prj_w_t_c = init * prj_w_t_c; //the order matters
		//Affine3d mTmp = prj_w_t_c.cast<double>();
		//mTmp.linear().transposeInPlace();
		//camera()->setFromModelViewMatrix(mTmp.data());
		camera()->setPosition(qglviewer::Vec(0, 0, 0));
		camera()->setUpVector(qglviewer::Vec(0, -1, 0));
		camera()->setViewDirection(qglviewer::Vec(0, 0, 1));
		//updateGL(); // Refresh display
	}
	else if (pEvent_->key() == Qt::Key_BracketRight)
	{
		//_idx_view++;
		//Affine3f prj_c_f_w = _pData->getTrackerPtr()->getView(&_idx_view);
		//Affine3f init; init.setIdentity(); init(1, 1) = -1.f; init(2, 2) = -1.f;// rotate the default opengl camera orientation to make it facing positive z
		//prj_c_f_w = init * prj_c_f_w; //the order matters
		//Affine3d mTmp = prj_c_f_w.cast<double>();
		//mTmp.linear().transposeInPlace();
		//camera()->setFromModelViewMatrix(mTmp.data());
		//camera()->setPosition(qglviewer::Vec(0, 0, 0));
		//camera()->setUpVector(qglviewer::Vec(0, -1, 0));
		//camera()->setViewDirection(qglviewer::Vec(0, 0, 1));
		//updateGL(); // Refresh display
	}
	else if (pEvent_->key() == Qt::Key_1)
	{
		_bShowText = !_bShowText;
		//updateGL(); // Refresh display
	}
	else if (pEvent_->key() == Qt::Key_2)
	{
		//Matrix4f mModelView;
		//Affine3f prj;
		//_pData->getTrackerPtr()->getCurrentFeatureProjectionMatrix(&prj);
		//mModelView = btl::utility::setModelViewGLfromPrj(prj);
		//Matrix4d mTmp = mModelView.cast<double>(); 
		//mTmp.linear().transposeInPlace();
		//camera()->setFromModelViewMatrix(mTmp.data());
		////updateGL(); // Refresh display
	}
	else if (pEvent_->key() == Qt::Key_4 && !(pEvent_->modifiers() & Qt::ShiftModifier)){
		_pData->switchShowTexts();
		//updateGL(); // Refresh display
	}
	else if (pEvent_->key() == Qt::Key_5) {
		_pData->switchShowMarkers();
		//updateGL(); // Refresh display
	}
	else if (pEvent_->key() == Qt::Key_6) {
		//_pData->switchShowMatchedFeaturesForRelocalisation();
		//updateGL(); // Refresh display
	}
	else if (pEvent_->key() == Qt::Key_7) {
		_pData->switchShowSurfaces();
		//updateGL(); // Refresh display
	}
	else if (pEvent_->key() == Qt::Key_8) {
		_pData->switchShowVoxels();
		//updateGL(); // Refresh display
	}
	else if (pEvent_->key() == Qt::Key_9) {
		//_pData->switchPyramid();
		_pData->_nNormalMap++;
		_pData->_nNormalMap %= 3;
		//updateGL(); // Refresh display
	}
	else if (pEvent_->key() == Qt::Key_L && !(pEvent_->modifiers() & Qt::ShiftModifier)) {
		_pData->switchLighting();
		//updateGL(); // Refresh display
	}
	else if (pEvent_->key() == Qt::Key_F2) {
		_pData->switchImgPlane();
		//updateGL();
	}
	else if (pEvent_->key() == Qt::Key_F3) {
		_pData->switchReferenceFrame();
		//updateGL();
	}
	else if (pEvent_->key() == Qt::Key_F4){
		_pData->switchCurrentFrame();
		//updateGL();
	}
	else if (pEvent_->key() == Qt::Key_F5){
		_pData->switchCameraPath();
		//updateGL();
	}
	else if (pEvent_->key() == Qt::Key_F6){
		_pData->switchShowCamera();
		//updateGL();
	}
	else if (pEvent_->key() == Qt::Key_F7){
		_pData->switchShowVisualRay();
		//updateGL();
	}
	else if (pEvent_->key() == Qt::Key_F8){
		_pData->_pKinect->_bFast = !_pData->_pKinect->_bFast;
		_pData->switchVoxelLevel();
		//updateGL();
	}
	else if (pEvent_->key() == Qt::Key_F9){
		//_pData->switchSphere();
		_pData->_pKinect->_nRawDataProcessingMethod++;
		_pData->_pKinect->_nRawDataProcessingMethod%=3;
		//updateGL();
	}
	else if (pEvent_->key() == Qt::Key_R && !(pEvent_->modifiers() & Qt::ShiftModifier)){
		//if (_pData->_strStage.compare("Relocalisation_Only")){//if Relocalisation_Only no need to exportGlobalModel()
		//	if (!_pData->_strStage.compare("Mapping_Using_GT") && _pData->_nRound >= _pData->_v_training_data_path.size())
		//		_pData->exportGlobalModel();
		//	if (!_pData->_strStage.compare("Tracking_n_Mapping"))
		//		_pData->exportGlobalModel();
		//}
		//_pData->exportRelativeGTworld2Userworld();
		_pData->loadFromYml();
		_pData->reset();
		//updateGL();
	}
	else if (pEvent_->key() == Qt::Key_C && !(pEvent_->modifiers() & Qt::ShiftModifier)){
		//lower c
		_pData->exportGlobalModel();
		//updateGL();
	}
	else if (pEvent_->key() == Qt::Key_C && (pEvent_->modifiers() & Qt::ShiftModifier)){
		//upper C
		_pData->importGlobalModel();
		//_pData->importVolume();
		//_pData->switchCapturing();
		//updateGL();
	}
	else if (pEvent_->key() == Qt::Key_S && !(pEvent_->modifiers() & Qt::ShiftModifier)){
		_pData->switchCapturing();
		//_nStatus = (_nStatus&(~VideoSourceKinect::MASK_RECORDER))|VideoSourceKinect::DUMP_RECORDING;
		//updateGL();
	}
	else if (pEvent_->key() == Qt::Key_T && !(pEvent_->modifiers() & Qt::ShiftModifier)){
		_pData->_pTracker->_bTrackingOnly = !_pData->_pTracker->_bTrackingOnly;
		//updateGL();
	}
	else if (pEvent_->key() == Qt::Key_P && !(pEvent_->modifiers() & Qt::ShiftModifier)){
		/*if ((_nStatus&VideoSourceKinect::MASK1) == VideoSourceKinect::PAUSE){
		_nStatus = (_nStatus&(~VideoSourceKinect::MASK1))|VideoSourceKinect::CONTINUE;
		}else if ((_nStatus&VideoSourceKinect::MASK1) == VideoSourceKinect::CONTINUE){
		_nStatus = (_nStatus&(~VideoSourceKinect::MASK1))|VideoSourceKinect::PAUSE;
		}*/
		_pData->switchCapturing();
		//updateGL();
	}
	else if (pEvent_->key() == Qt::Key_P && (pEvent_->modifiers() & Qt::ShiftModifier)){
		/*if ((_nStatus&VideoSourceKinect::MASK1) == VideoSourceKinect::PAUSE){
		_nStatus = (_nStatus&(~VideoSourceKinect::MASK1))|VideoSourceKinect::CONTINUE;
		}else if ((_nStatus&VideoSourceKinect::MASK1) == VideoSourceKinect::CONTINUE){
		_nStatus = (_nStatus&(~VideoSourceKinect::MASK1))|VideoSourceKinect::PAUSE;
		}*/
		_pData->switchContinuous();
		//updateGL();
	}
	else if (pEvent_->key() == Qt::Key_M && (pEvent_->modifiers() & Qt::ShiftModifier)) {
		/*if ((_nStatus&VideoSourceKinect::MASK1) == VideoSourceKinect::PAUSE){
		_nStatus = (_nStatus&(~VideoSourceKinect::MASK1))|VideoSourceKinect::CONTINUE;
		}else if ((_nStatus&VideoSourceKinect::MASK1) == VideoSourceKinect::CONTINUE){
		_nStatus = (_nStatus&(~VideoSourceKinect::MASK1))|VideoSourceKinect::PAUSE;
		}*/
		_pData->_pCubicGrids->gpuMarchingCubes();
		//updateGL();
	}
	else if (pEvent_->key() == Qt::Key_Escape && !(pEvent_->modifiers() & Qt::ShiftModifier)){
		QCoreApplication::instance()->quit();
	}
	else if (pEvent_->key() == Qt::Key_F && (pEvent_->modifiers() & Qt::ShiftModifier)){
		if (!isFullScreen()){
			toggleFullScreen();
		}
		else{
			setFullScreen(false);
			resize(1280, 480);
		}
	}
	QGLViewer::keyPressEvent(pEvent_);
}






