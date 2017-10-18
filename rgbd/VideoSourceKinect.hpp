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


#ifndef BTL_EXTRA_VIDEOSOURCE_VIDEOSOURCEKINECT
#define BTL_EXTRA_VIDEOSOURCE_VIDEOSOURCEKINECT

#include "DllExportDef.h"

using namespace openni;

namespace btl{
namespace kinect{

namespace btl_img = btl::image;
namespace btl_knt = btl::kinect;

class  CVideoSourceKinect 
{
public:
	//type
	typedef std::shared_ptr<CVideoSourceKinect> tp_shared_ptr;
	enum tp_mode { SIMPLE_CAPTURING = 1, RECORDING = 2, PLAYING_BACK = 3};
	enum tp_status { CONTINUE=01, PAUSE=02, MASK1 =07, START_RECORDING=010, STOP_RECORDING=020, CONTINUE_RECORDING=030, DUMP_RECORDING=040, MASK_RECORDER = 070 };
	enum tp_raw_data_processing_methods {
		BIFILTER_IN_ORIGINAL = 0, BIFILTER_IN_DISPARITY, BIFILTER_DYNAMIC
	};

	//constructor
    CVideoSourceKinect(ushort uResolution_, ushort uPyrHeight_, bool bUseNIRegistration_,const Eigen::Vector3f& eivCw_,const string& cam_param_path_ );
    virtual ~CVideoSourceKinect();
	void initKinect();
	void initRecorder(std::string& strPath_);
	virtual void initPlayer(std::string& strPathFileName_);
	// 1. need to call getNextFrame() before hand
	// 2. RGB color channel (rather than BGR as used by cv::imread())
	virtual bool getNextFrame(int* pnStatus_);

	// 0 VGA
	// 1 QVGA
	Status setVideoMode(ushort uLevel_);
	void setDumpFileName( const std::string& strFileName_ ){_strDumpFileName = strFileName_;}

protected:
	virtual void importYML();
	// convert the depth map/ir camera to be aligned with the rgb camera

	virtual void init();
	void gpuBuildPyramidUseNICVmBiFilteringInOriginalDepth( );
	void gpu_build_pyramid_dynamic_bilatera();
	virtual void gpuBuildPyramidUseNICVm();
	bool loadCoefficient(int nResolution_, int size_, Mat* coeff_, Mat* mask_);
	bool loadLocation(vector<float>* pvLocations_);

public:
	//depth calibration parameters
	GpuMat _calibXYxZ[2];
	GpuMat _mask[2];
	vector<float> _vRegularLocations;
	string _serial_number;
	bool _bMapUndistortionOn;
	//parameters
	float _fThresholdDepthInMeter; //threshold for filtering depth
	float _fSigmaSpace; //degree of blur for the bilateral filter
	float _fSigmaDisparity; 
	unsigned int _uPyrHeight;//the height of pyramid
	ushort _uResolution;//0 640x480; 1 320x240; 2 160x120 3 80x60
	float _fScaleRGB;//scale the input video source to standard resolution 0,1,2,.. __aKinectW[]
	float _fScaleDepth;// __aKinectW[] / ( # of columns of input video )
	float _fMtr2Depth; // 100
	int _nRawDataProcessingMethod;
	bool _bFast;

	//cameras
	btl_img::SCamera::tp_shared_ptr _pRGBCamera;
	btl_img::SCamera::tp_shared_ptr _pIRCamera;
	btl_knt::CRGBDFrame::tp_shared_ptr _pCurrFrame;
	//rgb
	cv::Mat			_cvmRGB;
	Mat				_cvmDep;

protected:
	//openni
    Device _device;
    VideoStream _color;
    VideoStream _depth;
	VideoStream** _streams;//array of pointers
	Recorder _recorder;

	VideoFrameRef _depthFrame;
	VideoFrameRef _colorFrame;

	const openni::SensorInfo* _depthSensorInfo;
	const openni::SensorInfo* _colorSensorInfo;
	const openni::SensorInfo* _irSensorInfo;

	cv::cuda::GpuMat _gpu_rgb;
	cv::Mat         _undist_rgb;
	cv::cuda::GpuMat _gpu_undist_rgb;
	//depth
    cv::Mat          _depth_float;
	cv::cuda::GpuMat _gpu_depth;
	cv::Mat          _undist_depth;
	cv::cuda::GpuMat _gpu_undist_depth;
	cv::cuda::GpuMat _gpu_depth_float;
	// duplicated camera parameters for speed up the VideoSourceKinect::align() in . because Eigen and cv matrix class is very slow.
	// initialized in constructor after load of the _cCalibKinect.
	float _aR[9];	// Relative rotation transpose
	float _aRT[3]; // aRT =_aR * T, the relative translation

	// Create and initialize the cyclic buffer

	//controlling flag
	static bool _bIsSequenceEnds;
	std::string _strDumpFileName;
	int _nMode; 

	float _fCutOffDistance;
	// (opencv-default camera reference system convention)
	Eigen::Vector3f _eivCw;
	string _cam_param_file;
	string _cam_param;
};//class VideoSourceKinect

} //namespace kinect
} //namespace btl



#endif //BTL_EXTRA_VIDEOSOURCE_VIDEOSOURCEKINECT
