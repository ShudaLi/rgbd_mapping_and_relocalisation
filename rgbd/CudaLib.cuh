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

#ifndef BTL_CUDA_HEADER
#define BTL_CUDA_HEADER
#include "DllExportDef.h"

namespace btl { namespace device
{
	using namespace cv::cuda;
void DLL_EXPORT cudaTestFloat3( const GpuMat& cvgmIn_, GpuMat* pcvgmOut_ );
void DLL_EXPORT cuda_depth2disparity2( const GpuMat& cvgmDepth_, float fCutOffDistance_, GpuMat* pcvgmDisparity_, float factor_ = 1000.f );
void DLL_EXPORT cuda_disparity2depth( const GpuMat& cvgmDisparity_, GpuMat* pcvgmDepth_ );
void DLL_EXPORT cuda_bilateral_filtering(const GpuMat& cvgmSrc_, const float& fSigmaSpace_, const float& fSigmaColor_, GpuMat* pcvgmDst_ );
void DLL_EXPORT cuda_pyr_down (const GpuMat& cvgmSrc_, const float& fSigmaColor_, GpuMat* pcvgmDst_);
void DLL_EXPORT cuda_unproject_rgb ( const GpuMat& cvgmDepths_, 
										const float& fFxRGB_, const float& fFyRGB_, const float& uRGB_, const float& vRGB_, unsigned int uLevel_,
										GpuMat* pcvgmPts_);
void DLL_EXPORT cuda_fast_normal_estimation(const GpuMat& cvgmPts_, GpuMat* pcvgmNls_ );
void DLL_EXPORT cuda_estimate_normals(const GpuMat& vmap, GpuMat* nmap, GpuMat* reliability = NULL);
//get scale depth
void DLL_EXPORT cuda_scale_depth(unsigned short usPyrLevel_, const float fFx_, const float fFy_, const float u_, const float v_, GpuMat* pcvgmDepth_);
void DLL_EXPORT cuda_transform_local2world(const float* pRw_/*col major*/, const float* pTw_, GpuMat* pcvgmPts_, GpuMat* pcvgmNls_, GpuMat* pcvgmMDs_);
//resize the normal or vertex map to half of its size
void DLL_EXPORT cuda_resize_map (bool bNormalize_, const GpuMat& cvgmSrc_, GpuMat* pcvgmDst_);
void DLL_EXPORT cuda_init_idx(int nCols_, int type_, GpuMat* p_idx_);
void DLL_EXPORT cuda_sort_column_idx(GpuMat* const& p_key_, GpuMat*const & p_idx_);
}//device
}//btl
#endif
