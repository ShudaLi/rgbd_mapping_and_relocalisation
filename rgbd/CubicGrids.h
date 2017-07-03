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


#ifndef BTL_GEOMETRY_CUBIC_GRIDS
#define BTL_GEOMETRY_CUBIC_GRIDS
#include "DllExportDef.h"
namespace btl{ namespace geometry
{
	using namespace cv::cuda;
	using namespace btl::kinect;
	using namespace std;
	using namespace pcl::device;
	class  CCubicGrids
	{
		//type
	public:
		typedef std::shared_ptr<CCubicGrids> tp_shared_ptr;
		enum {_X = 1, _Y = 2, _Z = 3};

		enum
		{ 
			DEFAULT_OCCUPIED_VOXEL_BUFFER_SIZE = 2 * 1000 * 1000      
		};

	private:
		void releaseVBOPBO();
		//methods
	public:
		CCubicGrids(ushort usVolumeResolutionX_, ushort usVolumeResolutionY_, ushort usVolumeResolutionZ_, float fVolumeSizeMX_, int nFeatureScale_, const string& featureName_);
		~CCubicGrids();
		void setFeatureName(const string& FeatureName_);


		void integrateDepth(const CRGBDFrame& cFrame_, const Intr& intr_);
		void integrateFeatures(const pcl::device::Intr& cCam_, const Matrix3f& Rw_, const Vector3f& Tw_, int nFeatureScale_, GpuMat& pts_w_, GpuMat& nls_w_,
											GpuMat& gpu_key_points_curr_, const GpuMat& gpu_descriptor_curr_, const GpuMat& gpu_distance_curr_, const int nEffectiveKeypoints_, GpuMat& gpu_inliers_, const int nTotalInliers_);
		int totalFeatures() const{
			return std::accumulate(_vTotalGlobal.begin(), _vTotalGlobal.end(), 0);
		}
		void rayCast(CRGBDFrame* pVirtualFrame_, bool bForDisplay_ = false, bool bFineCast_ = true, int lvl = 0) const;

		void storeNIFTI(const string& strPath_) const;
		void loadNIFTI(const string& strPath_);

		void storeGlobalFeatures(const string& strPath_) const;
		void loadGlobalFeatures(const string& strPath_);

		//ICP approach
		//SURF Pnp
		//Rw is the rotation matrix defined in world coordinate  
		//Tw is the translation vector defined in world coordinate
		//Xc = Rw * Xw + Tw;
		//Xc is a point defined in camera system
		//Xw is a point defined in world
		double icpFrm2Frm(const CRGBDFrame::tp_ptr pCurFrame_, const CRGBDFrame::tp_ptr pPrevFrame_, const short asICPIterations_[], Eigen::Matrix3f* peimRw_, Eigen::Vector3f* peivTw_, Eigen::Vector4i* pActualIter_) const;
		void extractRTFromBuffer(const GpuMat& cvgmSumBuf, Eigen::Matrix3f* peimRw_, Eigen::Vector3f* peivTw_) const;
		double verifyPoseHypothesesAndRefine(CRGBDFrame::tp_ptr pCurFrame_, CRGBDFrame::tp_ptr pPrevFrame_, vector<Eigen::Affine3f>& v_k_hypothese_poses, int nICPMethod_, int nStage_,
													   Matrix3f* pRw_, Vector3f* pTw_, int* p_best_idx_);

		void renderBoxGL() const;
		void renderOccupiedVoxels(btl::gl_util::CGLUtil::tp_ptr pGL_, int lvl_);
		void displayAllGlobalFeatures(int lvl_, bool bRenderSpheres_) const;
		void displayTriangles() const;
		void gpuMarchingCubes();
		void exportCSV() const;
	public:
		float _fx, _fy, _cx, _cy;
		pcl::device::Intr _intrinsics;

		vector<GpuMat> _gpu_global_3d_key_points; //12 x nfeatures
		// first 0,1,2 rows store the 3-D coordinate of the key point in world
		// the 3,4,5 rows store the 3-D surface normal of the key point
		// the 6 store the hessian score which describe the sharpness of the keypoint
		// the 7 is unimportant
		// the 8 stores the cosine angle between the surface normal and viewing direction of the camera
		// the 9, 10, 11 store the main direction which is under developing.
		// more details can be found from the kernel function insert_features_into_volume2() in FeatureVolume.cu
		vector<GpuMat> _gpu_global_descriptors; //nfeatures x 64/32
		vector<int> _vTotalGlobal;
		int _nFeatureScale;
		int _nKeyPointDimension;
		int _nFeatureName;

		//data
		//volume data
		//the front top left of the volume defines the origin of the world coordinate
		//and follows the right-hand cv-convention
		//physical size of the volume
		float3 _fVolumeSizeM;//in meter
		float _fVoxelSizeM; //in meter
		short3 _VolumeResolution; //in meter
		unsigned int _uVolumeLevelXY;
		unsigned int _uVolumeTotal;
		//truncated distance in meter
		//must be larger than 2*voxelsize 
		float _fTruncateDistanceM;
		//host
		cv::Mat _YXxZ_volumetric_content; //y*z,x,CV_32FC1,x-first
		//device
		cv::cuda::GpuMat _gpu_YXxZ_tsdf;

		float _fFeatureVoxelSizeM[5];
		vector<float> _vFeatureVoxelSizeM;
		vector<short3> _vFeatureResolution; //128,64,32,16,8

		GpuMat _gpu_YXxZ_vg_idx[5];
		GpuMat _gpu_feature_volume_coordinate;
		GpuMat _gpu_counter;

		//render context
		btl::gl_util::CGLUtil::tp_ptr _pGL;
		GLuint _uVBO;
		cudaGraphicsResource* _pResourceVBO;
		GLuint _uPBO;
		cudaGraphicsResource* _pResourcePBO;
		GLuint _uTexture;

		GpuMat _gpu_pts[5];
		GpuMat _gpu_feature_idx[5];
		vector<int> _vOccupied;
		GLUquadricObj*   _quadratic;	// Storage For Our Quadratic Objects

		Mat _triangles;
		Mat _normals;
	};




}//geometry
}//btl
#endif

