/* Copyright (c) 2015 by Shuda Li [lishuda1980@gmail.com]
* Computer Science Department, University of Bristol
* 
* Mapping and Relocalisation is licensed under the GPLv3 license.
* Details can be found in Licenses.txt
*
* For using the code or comparing to it in your research, you are
* expected to cite:
* Li, Shuda, & Calway, (2015) Andrew. RGBD Relocalisation Using
* Pairwise Geometry and Concise Key Point Sets.
* In Intl. Conf. on Robotics and Automatiro (ICRA) 2015.
*
* Permission is hereby granted, free of charge, to any person
* obtaining a copy of this software and associated documentation
* files (the "Software"), to deal in the Software without
* restriction, including without limitation the rights to use,
* copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following
* conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
* OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
* HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef _DATA4VIEWER_H_
#define _DATA4VIEWER_H_

using namespace btl::gl_util;
using namespace btl::kinect;
using namespace btl::geometry;
using namespace std;
using namespace Eigen;

void convert(const Eigen::Affine3f& eiM_, Mat* pM_);
void convert(const Eigen::Matrix4f& eiM_, Mat* pM_);
void convert(const Mat& M_, Eigen::Affine3f* peiM_);
void convert(const Mat& M_, Eigen::Matrix4f* peiM_);


class CData4Viewer
{
public:
	typedef boost::shared_ptr<CData4Viewer> tp_shared_ptr;

	CData4Viewer();
	virtual ~CData4Viewer();
	virtual void init();
	virtual void loadFromYml();
	virtual void reset();
	virtual void updatePF(){;}

	virtual void drawGlobalView();
	virtual void drawCameraView(qglviewer::Camera* pCamera_);
	virtual void drawRGBView();
	virtual void drawDepthView(qglviewer::Camera* pCamera_);

	virtual void setAsPrevPos();
	virtual void switchCapturing() { _bCapture = !_bCapture; }
	virtual void switchViewPosLock() { _bViewLocked = !_bViewLocked; }
	virtual void switchShowTexts() { _bShowText = !_bShowText; }
	virtual void switchShowMarkers() { _bShowMarkers = !_bShowMarkers; }
	virtual void switchShowVoxels() { _bShowVoxels = !_bShowVoxels; }
	virtual void switchShowSurfaces() { _bShowSurfaces = !_bShowSurfaces; }
	virtual void switchPyramid() { _pGL->_usLevel = ++_pGL->_usLevel%_pGL->_usPyrHeight; }
	virtual void switchLighting() { _pGL->_bEnableLighting = !_pGL->_bEnableLighting; }
	virtual void switchImgPlane() { _pGL->_bDisplayCamera = !_pGL->_bDisplayCamera; }
	virtual void switchReferenceFrame() { _bRenderReference = !_bRenderReference; }
	virtual void switchTrackOnly() { _bTrackOnly = !_bTrackOnly; }
	virtual void switchCameraPath() { _bIsCameraPathOn = !_bIsCameraPathOn; }
	virtual void switchCurrentFrame() { _bIsCurrFrameOn = !_bIsCurrFrameOn; }
	virtual void switchContinuous() { _bContinuous = !_bContinuous; }
	virtual void switchShowCamera() { _bShowCamera = !_bShowCamera; }
	virtual void switchShowVisualRay() { _bShowVisualRays = !_bShowVisualRays; }
	virtual void switchSphere() { _bRenderSphere = !_bRenderSphere; }
	virtual void switchVoxelLevel() {
		if(_nVoxelLevel < _nFeatureScale) 
			_nVoxelLevel++; 
		else 
			_nVoxelLevel = -1;
	}
	virtual const string& poseEstimationMethodName() { return _strPoseEstimationMethod; }
	virtual const string& icpMethodName() { return _strICPMethod; }
	virtual const string& matchingMethodName() { return _strMatchingMethod; }
	virtual const string& featureName() { return _strFeatureName; }
	virtual const CKinFuTracker::tp_shared_ptr getTrackerPtr() { return _pTracker; }
	virtual const bool isCapturing() const { return _bCapture; }
	virtual const bool isTrackOnly() const { return _bTrackOnly; }

	virtual void exportGlobalModel();
	virtual void importGlobalModel();
	virtual void initGroundTruth(){;}
	virtual void recordTrackerCamPairs();
	virtual void storeResultPose();//store current poses into reslt.txt files
	virtual void exportRelativeGTworld2Userworld();
	virtual void importRelativeGTworld2Userworld();
	virtual void make_statistics_on_estimation(){;}

	btl::kinect::CVideoSourceKinect::tp_shared_ptr _pKinect;
	btl::gl_util::CGLUtil::tp_shared_ptr _pGL;

	btl_img::SCamera::tp_scoped_ptr _pVirtualCamera;
	btl::kinect::CRGBDFrame::tp_scoped_ptr _pDepth;
	btl::kinect::CRGBDFrame::tp_scoped_ptr _pVirtualGlobalView;
	btl::kinect::CRGBDFrame::tp_scoped_ptr _pVirtualCameraView;
	btl::kinect::CRGBDFrame::tp_scoped_ptr _pCameraView2;

	btl::geometry::CCubicGrids::tp_shared_ptr _pCubicGrids;
	btl::geometry::CCubicGrids::tp_shared_ptr _pCubicGridsMoved;
	GLuint _uTexture;

	std::string _strPoseEstimationMethod;
	std::string _strMatchingMethod;
	std::string _strICPMethod;
	std::string _strFeatureName;
	ushort _uResolution;
	ushort _uPyrHeight;
	Eigen::Vector3f _eivCw;
	bool _bUseNIRegistration;
	vector<short> _vXYZCubicResolution;
	float _fVolumeSize;
	int _nRansacIterationasTracking;
	int _nRansacIterationasRelocalisation;
	int _nMode;//btl::kinect::VideoSourceKinect::PLAYING_BACK
	bool _bRepeat;// repeatedly play the sequence 
	float _fTimeLeft;
	int _nStatus;//1 restart; 2 //recording continue 3://pause 4://dump

	int _nFeatureScale;
	bool _bDisplayImage;
	bool _bLightOn;
	bool _bRenderReference;
	bool _bCapture; // controled by c
	bool _bContinuous; 
	bool _bTrackOnly;
	bool _bViewLocked; // controlled by 2
	//bool _bShowRelocalizaitonFeatures;
	bool _bShowVoxels;
	bool _bShowSurfaces;
	bool _bShowMarkers;
	bool _bShowCamera;
	bool _bShowText;
	bool _bIsCameraPathOn; 
	bool _bIsCurrFrameOn; 
	bool _bShowVisualRays;
	bool _bRenderSphere;
	bool _bLoadVolume;
	bool _bStorePoses;

	cv::String _global_model_folder;
	std::vector<cv::String> _vstrPoseEstimationMethod;
	std::vector<cv::String> _vstrICPMethod;
	std::vector<cv::String> _vstrMatchingMethod;
	std::vector<cv::String> _vstrFeatureName;

	int _nSequenceIdx;
	int _nMethodIdx;

	int _nVoxelLevel;

	double _dZoom;
	double _dXAngle;
	double _dYAngle;
	Eigen::Affine3f _eimModelViewGL;

	vector< Vector2f > _vError;

	Eigen::Affine3f _r_w2tw1;//a 4x4 matrix transforming a 3d point from w2 to w1
	Eigen::Affine3f _prj_c1tc2;//a 4x4 matrix transforming a 3d point from w1 to w2
	Eigen::Affine3f _prj_pov_f_w; 
	Eigen::Affine3f _prj_w_f_pov;

	bool _bInitialized;
	string _cam_param_path;
	vector<float> _vInitial_Cam_Pos; //intial camera position
	int _nOffset; // used for sync the rgb frame with depth frame
	int _nIdx;
	int _nStartIdx;
	int _nStep;
	float _fWeight; // the weight of similarity score and pair-wise score

	int _nError;
	int _nLost;
public:
	vector<String> _v_training_data_path;
	int _nRound;
	int _KNN_tracking;
	int _KNN_relocalisation;
	string _strStage; // 
	MatrixXf _cam_w1, _cam_w2;
	MatrixXf _viewing_dir_w1, _viewing_dir_w2;
	Eigen::Affine3f _prj_c_f_w;
	int _nIdxPairs; 
	bool _bCameraFollow;
	String _result_folder;
	vector<int> vEasy;
	vector<int> vHard;
	btl::geometry::CKinFuTracker::tp_shared_ptr _pTracker;
};


#endif
