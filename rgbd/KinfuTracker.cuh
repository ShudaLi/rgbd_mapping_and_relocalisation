//Copyright(c) 2015 by Shuda Li[lishuda1980@gmail.com]
//
//Mapping and Relocalisation is licensed under the GPLv3 license.
//Details can be found in the following.
//
//For using the code or comparing to it in your research, you are
//expected to cite :
//Li, Shuda, &Calway, (2015) Andrew.RGBD Relocalisation Using
//Pairwise Geometry and Concise Key Point Sets.
//In Intl.Conf.on Robotics and Automatiro(ICRA) 2015.
//
//Permission is hereby granted, free of charge, to any person
//obtaining a copy of this software and associated documentation
//files(the "Software"), to deal in the Software without
//restriction, including without limitation the rights to use,
//copy, modify, merge, publish, distribute, sublicense, and / or sell
//copies of the Software, and to permit persons to whom the
//Software is furnished to do so, subject to the following
//conditions :
//
//The above copyright notice and this permission notice shall be
//included in all copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
//EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
//OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
//NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
//HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
//WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
//OTHER DEALINGS IN THE SOFTWARE.

#ifndef KINFUTRACKER_CUDA_HEADER
#define KINFUTRACKER_CUDA_HEADER
#include "DllExportDef.h"

using namespace cv::cuda;
using namespace std;
using namespace pcl::device;
namespace btl{ namespace device{

int DLL_EXPORT cuda_collect_distinctive(const float& fPercentage_, const GpuMat& idx_, short n_features_, const GpuMat& cvgmC2PPairs_, const GpuMat& cvgmDistance_, GpuMat* ptr_idx_curr_2_prev_selected_ );
//For KNN K=2
int DLL_EXPORT cuda_collect_point_pairs_2nn(const float fMatchThreshold_, const float& fPercentage_, int nMatchingMethod_, int nOffset_, int nKeyPoints_,
							  const GpuMat& cvgmC2PPairs_, const GpuMat& cvgmDistance_,
							  const GpuMat& cvgmPtsWorldPrev_, const GpuMat& cvgmPtsLocalCurr_, 
							  const GpuMat& gpu_2d_curr_, const GpuMat& gpu_bv_curr_, const GpuMat& gpu_weight_, const GpuMat& gpu_key_points_curr_, const GpuMat& gpu_descriptor_curr_,
							  const GpuMat& Nls_Prev_, const GpuMat& MDs_Prev_,  const GpuMat& Nls_LocalCurr_, const GpuMat& MDs_LocalCurr_, 
						      GpuMat* ptr_pts_training_selected_, GpuMat* ptr_pts_curr_selected_,
							  GpuMat* ptr_nls_training_selected_,GpuMat* ptr_nls_curr_selected_, 							  
							  GpuMat* ptr_mds_training_selected_,GpuMat* ptr_mds_curr_selected_,
							  GpuMat* ptr_2d_curr_selected_, GpuMat* ptr_bv_curr_selected_, GpuMat* ptr_weight_selected_, GpuMat* ptr_key_point_selected_, GpuMat* ptr_descriptor_curr_selected_,
							  GpuMat* ptr_idx_curr_2_prev_selected_, GpuMat* ptr_hamming_dist_selected_, GpuMat* ptr_ratio_selected_ );
void DLL_EXPORT cuda_calc_adjacency_mt(const GpuMat& global_pts_, const GpuMat& curr_pts_,
							  const GpuMat& global_nls_, const GpuMat& curr_nls_, const GpuMat& curr_2d_, 
							  GpuMat* ptr_credibility_ );

void DLL_EXPORT cuda_calc_adjacency_mt_binary(const GpuMat& global_pts_, const GpuMat& curr_pts_,
									const GpuMat& global_nls_, const GpuMat& curr_nls_, const GpuMat& curr_2d_,
									GpuMat* ptr_credibility_, GpuMat* ptr_geometry_);

void DLL_EXPORT cuda_collect_all(const GpuMat& global_pts_, const GpuMat& curr_pts_,
					const GpuMat& global_nls_, const GpuMat& curr_nls_, const GpuMat& curr_2d_,
					GpuMat* rotations_, GpuMat* translation_, GpuMat* votes_, GpuMat* ptr_vote_bit_flags_);

void DLL_EXPORT cuda_calc_common_ratio(const int r, const int c, const GpuMat& votes_, const GpuMat& ptr_vote_bit_flags_, GpuMat* common_);

vector<int> DLL_EXPORT cuda_extract_key_points(const pcl::device::Intr& intr, const float fVoxelSize_[], const int nFeatureScale_,
														 const GpuMat& pts_, const GpuMat& nls_, const GpuMat& depth_, const GpuMat& reliability_, 
														 const GpuMat& descriptor_, const GpuMat& key_points_, int n_detected_feature_,
														 vector<GpuMat>* ptr_v_descriptor_, vector<GpuMat>* ptr_v_key_points_,
														 vector<GpuMat>* ptr_v_pts_, vector<GpuMat>* ptr_v_nls_, vector<GpuMat>* ptr_v_mds_, vector<GpuMat>* ptr_v_2d_, vector<GpuMat>* ptr_v_bv_, vector<GpuMat>* ptr_v_weigts_ );

void DLL_EXPORT cuda_apply_1to1_constraint(const GpuMat& gpu_credibility_, const GpuMat& gpu_idx_, cv::Mat* p_1_to_1_);
void DLL_EXPORT cuda_apply_1to1_constraint_binary(const GpuMat& credibility_, const GpuMat& geometry_, const GpuMat& gpu_idx_, cv::Mat* p_1_to_1_);
void DLL_EXPORT cuda_select_inliers_from_am(const GpuMat& gpu_credibility_, const GpuMat& gpu_column_idx_, float fExpThreshold_, int TotalTrials_, int ChainLength_, GpuMat* p_inliers_, GpuMat* p_node_numbers_);
void DLL_EXPORT cuda_select_inliers_from_am_2(const GpuMat& gpu_credibility_, const GpuMat& gpu_column_idx_, float fExpThreshold_, int TotalTrials_, int ChainLength_, GpuMat* p_inliers_, GpuMat* p_node_numbers_);

}//device
}//btl

#endif