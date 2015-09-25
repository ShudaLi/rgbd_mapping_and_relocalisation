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
//
//You agree to acknowledge use of the Software in any reports or publications of
//results obtained with the Software and make reference to the following publication :
//Li, Shuda, &Calway, Andrew(2015).RGBD Relocalisation Using Pairwise Geometry
//and Concise Key Point Sets.In Intl Conf.Robotics and Automation.


#ifndef BTL_GEOMETRY_KINFU_TRACKER
#define BTL_GEOMETRY_KINFU_TRACKER

#include "DllExportDef.h"

namespace btl{ namespace geometry
{
	using namespace std;
	using namespace cv;
	using namespace cv::cuda;
	using namespace btl::kinect;
	using namespace Eigen;

	class DLL_EXPORT CKinFuTracker
	{
	public:
		//type
		typedef boost::shared_ptr<CKinFuTracker> tp_shared_ptr;

		enum{ RANSACPnP, AO, AONn2d, AORansac, AONRansac, AONn2dRansac, AONn2dRansac2, AONMDn2dRansac }; //pose estimation approaches
		enum{ IDF = 0, GM, GM_GPU, BUILD_GT, ICP }; //matching approaches
	public:
		//both pKeyFrame_ and pCubicGrids_ must be allocated before hand
		CKinFuTracker(CRGBDFrame::tp_ptr pKeyFrame_, CCubicGrids::tp_shared_ptr pCubicGrids_, int nResolution_/*=0*/, int nPyrHeight_/*=3*/, string& trackingMathod_, string& ICPMethod_, string& matchingMethod_);
		~CKinFuTracker(){ ; }

		void reset();

		virtual bool init(CRGBDFrame::tp_ptr pKeyFrame_);
		virtual void tracking(CRGBDFrame::tp_ptr pCurFrame_);
		virtual bool relocalise(CRGBDFrame::tp_ptr pCurFrame_);

		void setPoseEstimationMethod(int nMethod_){ _nPoseEstimationMethod = nMethod_;}
		void setICPMethod(int nMethod_){ _nICPMethod = nMethod_; }
		void setMatchingMethod(int nMethod_){ _nMatchingMethod = nMethod_; }
		void setKinFuTracker(string& PoseEstimationMethodName_, string& ICPMethods_, string& MatchingMethodName_);
		void setModelFolder(const string& global_model_folder_){ _path_to_global_model = global_model_folder_; }
		void setResultFolder(const string& result_folder_){ _path_to_result = result_folder_; }

		void getCurrentProjectionMatrix(Eigen::Affine3f* pProjection_){ *pProjection_ = _pose_refined_c_f_w; return; }
		void getCurrentFeatureProjectionMatrix(Eigen::Affine3f* pProjection_){ *pProjection_ = _pose_feature_c_f_w; return; }
		void getNextView( Eigen::Affine3f* pSystemPose_ );
		void getPrevView( Eigen::Affine3f* pSystemPose_ );

		void displayVisualRays(btl::gl_util::CGLUtil::tp_ptr pGL_, const int lvl_, bool bRenderSphere_ ) const;
		void displayAllGlobalFeatures(int lvl_, bool bRenderSpheres_) const;
		void displayGlobalRelocalization();
		void displayCameraPath() const;
		void displayCameraPathReloc() const;
		void displayFeatures2D(int lvl_, CRGBDFrame::tp_ptr pCurFrame_ ) const;
		void displayTrackedKeyPoints();

		void storeGlobalFeaturesAndCameraPoses(const string& folder_);
		void loadGlobalFeaturesAndCameraPoses(const string& folder_);

	protected:
		void storeCurrFrame(CRGBDFrame::tp_ptr pCurFrame_);
		void extractFeatures(const CRGBDFrame::tp_ptr pFrame_, int nFeatureType_);

		void calcScores(const Mat& credibility_, Mat* ptr_count_) const;
		float calcSimilarity(const float& hamming_distance_) const {
			return _fWeight * exp( /*512.f -*/ -1.f* hamming_distance_ / 128.f );
		}
		void findCorrespondences ( const vector<GpuMat>& gpu_descriptors_training_, const vector<GpuMat>& gpu_pts_training_, const vector<int> vTotal_, const int K_,
									vector<Mat>* ptr_v_selected_world_,vector<Mat>* ptr_v_nls_selected_world_, vector<Mat>* ptr_v_mds_selected_world_,
									vector<Mat>* ptr_v_selected_curr_, vector<Mat>* ptr_v_nls_selected_curr_,  vector<Mat>* ptr_v_mds_selected_curr_,
									vector<Mat>* ptr_v_selected_2d_curr_,  vector<Mat>* ptr_v_selected_bv_curr_, vector<Mat>* ptr_v_selected_weights_, vector<Mat>* ptr_v_selected_inliers_);

		void gpuExtractCorrespondencesMaxScore( const GpuMat& hamming_distance_, const GpuMat& global_pts_, const GpuMat& curr_pts_, const GpuMat& global_nls_, const GpuMat& curr_nls_, const GpuMat& curr_2d_, const GpuMat& curr_bv_, const GpuMat& gpu_curr_wg_,
												const int K_, GpuMat& credibility_, 
												vector<Mat>* ptr_v_selected_world_,vector<Mat>* ptr_v_nls_selected_world_, 
												vector<Mat>* ptr_v_selected_curr_, vector<Mat>* ptr_v_nls_selected_curr_, vector<Mat>* ptr_v_selected_2d_curr_, vector<Mat>* ptr_v_selected_bv_curr_, vector<Mat>* ptr_v_selected_wg_curr_,
												vector<Mat>* ptr_v_selected_inliers_ );
		void gpuExtractCorrespondencesMaxScore_binary(const GpuMat& hamming_distance_, const GpuMat& gpu_global_pts_, const GpuMat& gpu_curr_pts_, const GpuMat& gpu_global_nls_, const GpuMat& gpu_curr_nls_, const GpuMat& gpu_curr_2d_, const GpuMat& gpu_curr_bv_, const GpuMat& gpu_curr_wg_,
													  const int K_, GpuMat& gpu_credibility_, 
													  vector<Mat>* ptr_v_selected_world_, vector<Mat>* ptr_v_nls_selected_world_, 
													  vector<Mat>* ptr_v_selected_curr_, vector<Mat>* ptr_v_nls_selected_curr_, vector<Mat>* ptr_v_selected_2d_curr_, vector<Mat>* ptr_v_selected_bv_curr_, vector<Mat>* ptr_v_selected_wg_curr_,
													  vector<Mat>* ptr_v_selected_inliers_);
		void sparsePoseEstimation(const Mat& pts_World_, const Mat& nls_World_, const Mat& mds_World_,
				 						   const Mat& pts_Cam_,   const Mat& nls_Cam_,   const Mat& mds_Cam_, 
										   const Mat& m2D_, const Mat& bv_, const Mat& wg_, Matrix3f* pR_, Vector3f* pT_, Mat* ptr_inliers_);

		//given current frame, find K nearest neighbours of all tiny frames learnt before. 
		Mat apply1to1Contraint( const Mat& credibility_, const Mat& column_scores_, const Mat& selected_idx_);
		void initialPoseEstimation(CRGBDFrame::tp_ptr pCurFrame_, const vector<GpuMat>& gpu_descriptors_, const vector<GpuMat>& gpu_3d_key_points_, const vector<int> vTotal_, vector<Eigen::Affine3f>* ptr_v_k_hypothese_poses);
		string _path_to_global_model;
		string _path_to_result;
		Ptr<TemplateMatching> _pGpuTM_bw, _pGpuTM_dp;
		CCubicGrids::tp_shared_ptr _pCubicGrids; //volumetric data
		CRGBDFrame::tp_scoped_ptr _pPrevFrameWorld;

		btl::image::SCamera::tp_ptr _pRGBCamera; //share the content of the RGBCamera with those from VideoKinectSource
		Eigen::Affine3f _pose_refined_c_f_w;//projection matrix world to cam
		Eigen::Affine3f _pose_feature_c_f_w;//projection matrix world to cam

		//camera matrix
		Mat _cvmA;
		float _fx, _fy, _cx, _cy;
		pcl::device::Intr _intrinsics;
		int _nCols, _nRows;
		float _visual_angle_threshold;
		float _normal_angle_threshold;
		float _distance_threshold;
		unsigned int _uViewNO;

		//vector<Eigen::Vector3i> _veivIters;
		vector<Eigen::Affine3f> _v_prj_c_f_w_training;//a vector of projection matrices, where the projection matrices transform points defined in world to camera system.
		vector<Eigen::Affine3f> _v_relocalisation_p_matrices;//a vector of projection matrices, where the projection matrices transform points defined in world to camera system.
		int _nPoseEstimationMethod;
		int _nICPMethod;
		int _nMatchingMethod;
		int _nFeatureName;

		Ptr<cv::xfeatures2d::SURF> _pSurf;
		boost::scoped_ptr<cuda::ORB> _pORBGpu;
		Ptr<BRISK> _pBRISK;

		vector<GpuMat> _gpu_key_points_prev;
		vector<GpuMat> _gpu_descriptor_prev;
		vector<GpuMat> _gpu_pts_prev; //3d points with feature attached. Note that _gpu_selected_pts_curr is further refined from this
		vector<GpuMat> _gpu_nls_prev;
		vector<GpuMat> _gpu_mds_prev;

		vector<GpuMat> _gpu_pts_curr;
		vector<GpuMat> _gpu_nls_curr;
		vector<GpuMat> _gpu_mds_curr;
		vector<GpuMat> _gpu_2d_curr;//float2 format
		vector<GpuMat> _gpu_bv_curr;//float2 format
		vector<GpuMat> _gpu_weights;//float2 format
		vector<GpuMat> _gpu_keypoints_curr;
		vector<GpuMat> _gpu_descriptor_curr;

		int _nGlobalKeyPointDimension;

		GpuMat _gpu_key_array_2d;//flattened key point idx for inserting features into global feature set
		GpuMat _gpu_relined_inliers;
		
		GpuMat _gpu_keypoints_all_curr;
		GpuMat _gpu_descriptor_all_curr;
		GpuMat _gpu_pts_all_curr;
		GpuMat _gpu_nls_all_curr;
		GpuMat _gpu_mds_all_curr;
		GpuMat _gpu_mask_all_curr; //mask for counting 3D-3D feature pairs
		GpuMat _gpu_mask_all_2d_curr; //mask for counting 2D-3D feature pairs
		GpuMat _gpu_mask_all_2d_prev;

		GpuMat _gpu_keypoints_all_prev;
		GpuMat _gpu_descriptor_all_prev;
		GpuMat _gpu_pts_all_prev;
		GpuMat _gpu_nls_all_prev;
		GpuMat _gpu_mds_all_prev;
		GpuMat _gpu_mask_all_prev;
		GpuMat _gpu_c2p_idx;
		GpuMat _gpu_mask_counting;

		vector<cv::DMatch> _vMatches;
		vector<vector<cv::DMatch> > _vvMatches;

		GpuMat _match_tmp;

		int _n_detected_features_curr;
		int _n_detected_features_prev;

		//boost::scoped_ptr<cuda::DescriptorMatcher> _pMatcherGpuHamming;  
		Ptr<cuda::DescriptorMatcher> _pMatcherGpuHamming;
		boost::scoped_ptr<cv::FlannBasedMatcher> _pMatcherFlann;

		GLUquadricObj*   _quadratic;	// Storage For Our Quadratic Objects

		float _fMatchThreshold;
		int _nSearchRange;
		int _descriptor_bytes;

		//relocalization
		vector<Mat> _v_selected_world, _v_selected_curr, _v_selected_2d_curr, _v_selected_bv_curr, _v_selected_weights, _v_selected_inliers, _v_selected_inliers_reloc;
		vector<Mat> _v_nls_selected_world, _v_nls_selected_curr,_v_mds_selected_world, _v_mds_selected_curr;
		int _best_k; 

		//prev or global
		GpuMat _gpu_pts_global_reloc;
		GpuMat _gpu_nls_global_reloc;
		GpuMat _gpu_mds_global_reloc;
		
		//current
		GpuMat _gpu_pts_curr_reloc;    // 1. pts
		GpuMat _gpu_nls_curr_reloc;    // 2. nls
		GpuMat _gpu_mds_curr_reloc;    // 3. mds
		GpuMat _gpu_2d_curr_reloc;     // 4. float2
		GpuMat _gpu_bv_curr_reloc;
		GpuMat _gpu_weight_reloc;
		GpuMat _gpu_keypoint_curr_reloc;  // 5. keypoint
		GpuMat _gpu_descriptor_curr_reloc;// 6. descriptor
		GpuMat _gpu_hamming_distance_reloc;//7. hamming distances 
		GpuMat _gpu_idx_curr_2_prev_reloc; //8. index
		GpuMat _gpu_distinctive_reloc; //8. index

		GpuMat _gpu_idx_all_curr;//gpu 
		GpuMat _gpu_DOs_all_curr;
		GpuMat _gpu_other_all_curr;
		GpuMat _gpu_counter;

		int _nAppearanceMatched;
		int _total_refined_features;

	public:
		int _nStage; //tracking, relocalisation
		int _buffer_size;
		int _nHessianThreshold;
		int _nMinFeatures[5];
		double _aMinICPEnergy[5];
		int _aMaxChainLength[5];
		int _nPyrHeight;
		int _nResolution;

		int _K_tracking;
		int _K_relocalisation;
		int _K;
		int _nFeatureScale;
		float _fDistThreshold; // the distance threshold of pair-wise link
		float _fExpThreshold;
		float _fPercantage;
		float _fWeight; // the weighter to weight between similarity score and pair-wise features
		bool _bTrackingOnly; // this is for testing relocalisation module
		bool _bIsVolumeLoaded;
		bool _bLoadVolume;
		bool _bStoreAll;
		bool _bRelocWRTVolume;
		int _nRansacIterationasTracking;
		int _nRansacIterationasRelocalisation;

	};//CKinFuTracker

}//geometry
}//btl

#endif
