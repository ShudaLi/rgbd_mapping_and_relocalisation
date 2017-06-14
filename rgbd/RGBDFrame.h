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



#ifndef BTL_KEYFRAME
#define BTL_KEYFRAME
#define USE_PBO 1
#include "DllExportDef.h"

namespace btl { namespace kinect {

using namespace Eigen;
using namespace cv;
using namespace cv::cuda;

class  CRGBDFrame {
	//type
public:
	typedef std::shared_ptr< CRGBDFrame > tp_shared_ptr;
	typedef CRGBDFrame* tp_ptr;
	enum tp_cluster { NORMAL_CLUSTER, DISTANCE_CLUSTER};


public:
    CRGBDFrame( btl::image::SCamera::tp_ptr pRGBCamera_, ushort uResolution_, ushort uPyrLevel_, const Eigen::Vector3f& eivCw_ );
	CRGBDFrame(const CRGBDFrame::tp_ptr pFrame_);
	CRGBDFrame(const CRGBDFrame& Frame_);
	~CRGBDFrame();
	
	//accumulate the relative R T to the global RT
	void applyRelativePose( const CRGBDFrame& sReferenceKF_ ); 
	bool isMovedwrtReferencInRadiusM(const CRGBDFrame* const pRefFrame_, double dRotAngleThreshold_, double dTranslationThreshold_);

	// set the opengl modelview matrix to align with the current view
	void getGLModelViewMatrix(Matrix4f* pModelViewGL_) const {
		*pModelViewGL_ = btl::utility::setModelViewGLfromRTCV ( _Rw, _Tw );
		return;
	}
	Matrix4f getGLModelViewMatrix( ) const {
		return btl::utility::setModelViewGLfromRTCV ( _Rw, _Tw );
	}
	void getRnt(Matrix3f* pR_, Vector3f* pt_) const {
		*pR_ = _Rw;
		*pt_ = _Tw;
		return;
	}
	const Matrix3f& getR() const {return _Rw;}
	const Vector3f& getT() const {return _Tw;}
	void getPrjCfW(Eigen::Affine3f* ptr_proj_cfw) {
		ptr_proj_cfw->setIdentity();
		ptr_proj_cfw->translation() = _Tw;
		ptr_proj_cfw->linear() = _Rw;
		return;
	}
	Eigen::Affine3f getPrjCfW() const {
		Eigen::Affine3f prj; prj.setIdentity();
		prj.linear() = _Rw;
		prj.translation() = _Tw;
		return prj;
	}

	void setRTFromC(float fXA_, float fYA_, float fZA_, float fCwX_,float fCwY_,float fCwZ_);
	void setRTFromC(const Matrix3f& eimRotation_, const Vector3f& eivCw_);
	void setRTw(const Matrix3f& eimRotation_, const Vector3f& eivTw_);
	void setRTFromPrjWfC(const Eigen::Affine3f& prj_wfc_);
	void setRTFromPrjCfW(const Eigen::Affine3f& prj_cfw_);
	void initRT();
	void copyRTFrom(const CRGBDFrame& cFrame_ );
	void assignRTfromGL();

	void gpuTransformToWorld(const ushort usPyrLevel_);
	void gpuTransformToWorld();

	// render the camera location in the GL world
	void renderCameraInWorld(btl::gl_util::CGLUtil::tp_ptr pGL_, bool bRenderCoordinate_, float* color_, bool bRenderCamera_, const double& dPhysicalFocalLength_, const unsigned short uLevel_);
	// render the depth in the GL world 
	void render3DPtsInSurfel(btl::gl_util::CGLUtil::tp_ptr pGL_, const unsigned short uLevel_) const;
	void render3DPts(btl::gl_util::CGLUtil::tp_ptr pGL_,const ushort usLevel_);
	void gpuRender3DPts(btl::gl_util::CGLUtil::tp_ptr pGL_, const ushort usPyrLevel_);

	// detect the correspondences 
	double gpuCalcRTBroxOpticalFlow ( const CRGBDFrame& sPrevFrameWorld_, const double dDistanceThreshold_, unsigned short* pInliers_);
	void gpuBroxOpticalFlow (const CRGBDFrame& sPrevFrameWorld_, GpuMat* pcvgmColorGraph_);

	// copy the content to another keyframe at 
	void copyTo( CRGBDFrame* pKF_, const short sLevel_ ) const;
	void copyTo( CRGBDFrame* pKF_ ) const;
	void copyImageTo( CRGBDFrame* pKF_ ) const;

	void convert2NormalMap();
	void convertDepth2Gray(float max_);
	void exportNormalMap(const string& file_name_) const;
	void exportYML(const std::string& strPath_, const std::string& strYMLName_);
	void importYML(const std::string& strPath_, const std::string& strYMLName_);

	ushort pyrHeight() {return _uPyrHeight;}

private:
	//surf keyframe matching
	void allocate();
	void gpuConvert2ColorGraph( GpuMat* pcvgmU_, GpuMat* pcvgmV_, GpuMat* pcvgmColorGraph_ );
	float calcRTFromPair(const CRGBDFrame& sPrevKF_, const double dDistanceThreshold_, unsigned short* pInliers_);
public:
	btl::image::SCamera::tp_ptr _pRGBCamera; //share the content of the RGBCamera with those from VideoKinectSource
	//host
	std::shared_ptr<cv::Mat> _acvmShrPtrPyrDepths[4];
	std::shared_ptr<cv::Mat> _acvmShrPtrPyrPts[4]; //using pointer array is because the vector<cv::Mat> has problem when using it &vMat[0] in calling a function
	std::shared_ptr<cv::Mat> _acvmShrPtrPyrNls[4]; //CV_32FC3 type
	std::shared_ptr<cv::Mat> _acvmShrPtrPyrReliability[4]; //cpu version 
	std::shared_ptr<cv::Mat> _acvmShrPtrPyrRGBs[4];
	std::shared_ptr<cv::Mat> _acvmShrPtrPyrBWs[4];
	//device
	std::shared_ptr<GpuMat> _acvgmShrPtrPyrDepths[4];
	std::shared_ptr<GpuMat> _acvgmShrPtrPyrPts[4]; //using pointer array is because the vector<cv::Mat> has problem when using it &vMat[0] in calling a function
	std::shared_ptr<GpuMat> _acvgmShrPtrPyrNls[4]; //CV_32FC3
	std::shared_ptr<GpuMat> _acvgmShrPtrPyrReliability[4]; //CV_32FC1 ratio = largest / smallest eigen value
	std::shared_ptr<GpuMat> _acvgmShrPtrPyrRGBs[4];
	std::shared_ptr<GpuMat> _acvgmShrPtrPyrBWs[4];

	std::shared_ptr<GpuMat> _acvgmShrPtrPyrDisparity[4];
	std::shared_ptr<GpuMat> _acvgmShrPtrPyr32FC1Tmp[4];
	std::shared_ptr<GpuMat> _pcvgmPrev,_pcvgmCurr,_pcvgmU,_pcvgmV;

	std::shared_ptr<GpuMat> _pry_mask[4];
	//clusters
	/*static*/ std::shared_ptr<cv::Mat> _acvmShrPtrAA[4];//for rendering
		
	GpuMat _normal_map;
	GpuMat _depth_gray;
	//pose
	//R & T is the relative pose w.r.t. the coordinate defined in previous camera system.
	//R & T is defined using CV convention
	//R & T X_curr = R* X_prev + T;
	//after applying void applyRelativePose() R, T -> R_w, T_w
	//X_c = R_w * X_w + T_w 
	//where _w defined in world reference system
	//      _c defined in camera reference system (local reference system) 
	Matrix3f _Rw; 
	Vector3f _Tw; 
	Vector3f _eivInitCw;
	Sophus::SE3<float> _T_cw;
	//GL ModelView Matrix
	//render context
	//btl::gl_util::CGLUtil::tp_ptr _pGL;
	bool _bGPURender;


	tp_cluster _eClusterType;

private:
	//for surf matching
	//host
	public:

	ushort _uPyrHeight;
	ushort _uResolution;
	
};//end of class



}//utility
}//btl

#endif
