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

#ifndef BTL_CUDA_FEATURE_VOLUME_HEADER
#define BTL_CUDA_FEATURE_VOLUME_HEADER
#include "DllExportDef.h"

using namespace cv::cuda;
using namespace std;
using namespace pcl::device;
namespace btl { namespace device{

int DLL_EXPORT cuda_refine_inliers(float fDistThre_, float NormalAngleThre_, float VisualAngleThre_, int nAppearanceMatched_, const matrix3_cmf& Rw_, const float3& Tw_,
									const GpuMat& pts_global_reloc_, const GpuMat& nls_global_reloc_, const GpuMat& pts_curr_reloc_, const GpuMat& nls_curr_reloc_,
									GpuMat* p_relined_inliers_);

void DLL_EXPORT cuda_integrate_features(const pcl::device::Intr& intr, const pcl::device::Mat33& RwInv_, const float3& Cw_, int nFeatureScale_, const short3& resolution_,
												const cv::cuda::GpuMat& cvgmVolume_, const float3& volume_size, const float fTruncDistanceM_, const float& fVoxelSize_,
												const float fFeatureVoxelSize_[], const vector<short3>& vResolution_, GpuMat* pcvgmFeatureVolumeIdx_,
												const GpuMat& cvgmKeyPointCurr_, const GpuMat& cvgmDescriptorCurr_, const GpuMat& gpu_key_array_2d_, const int nKeypoints_,
												vector<int>* p_vOffset_, vector<GpuMat>* pcvgmGlobalKeyPoint_, vector<GpuMat>* pcvgmGlobalDescriptor_);

std::vector<int> DLL_EXPORT cuda_get_occupied_vg(const GpuMat*  feature_volume_idx_,
									 const float* fVoxelSize_, const int nFeatureScale_,
									 GpuMat* ptr_pts_world_, const vector<short3>& vResolution_,
									 GpuMat* ptr_feature_idx_);

void DLL_EXPORT cuda_nonmax_suppress_n_integrate(const pcl::device::Intr& intr_, const pcl::device::Mat33& RwInv_, const float3& Cw_, const short3& resolution_,
													const GpuMat& volume_, const float3& volume_size, const float& fVoxelSize_,
													const float fFeatureVoxelSize_[], int nFeatureScale_, const vector<short3>& vResolution_, GpuMat* feature_volume_,
													const GpuMat& pts_curr_, const GpuMat& nls_curr_,
													GpuMat& key_points_curr_, const GpuMat& descriptors_curr_, const GpuMat& distance_curr_, const int nEffectiveKeyPoints_,
													GpuMat& gpu_inliers_, const int nTotalInliers_, GpuMat* p_volume_coordinate_, GpuMat* p_counter_,
													vector<int>* p_vOffset_, vector<GpuMat>* pcvgmGlobalKeyPoint_, vector<GpuMat>* pcvgmGlobalDescriptor_);


}//device
}//btl
#endif