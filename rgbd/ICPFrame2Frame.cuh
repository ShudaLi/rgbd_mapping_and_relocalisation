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

#ifndef BTL_CUDA_REGISTRATION_HEADER
#define BTL_CUDA_REGISTRATION_HEADER
#include "DllExportDef.h"

namespace btl { namespace device {
	using namespace cv::cuda;
	using namespace pcl::device;
GpuMat DLL_EXPORT cuda_icp_fr_2_fr(const Intr& sCamIntr_, float fDistThres_, float fCosAngleThres_,
									  const Mat33& RwCurTrans_, const float3& TwCur_, 
									  const Mat33& RwPrev_, const float3& TwPrev_, const GpuMat& depth_curr_,
									  const GpuMat& cvgmVMapWorldPrev_, const GpuMat& cvgmNMapWorldPrev_, 
									  const GpuMat& cvgmVMapLocalCur_,  const GpuMat& cvgmNMapLocalCur_ );

double DLL_EXPORT calc_energy_icp_fr_2_fr(const Intr& sCamIntr_, float fDistThres_, float fCosAngleThres_,
									const Mat33& RwCurTrans_, const float3& TwCur_,
									const Mat33& RwPrev_, const float3& TwPrev_, const GpuMat& depth_curr_,
									const GpuMat& cvgmVMapWorldPrev_, const GpuMat& cvgmNMapWorldPrev_,
									const GpuMat& cvgmVMapLocalCur_, const GpuMat& cvgmNMapLocalCur_, GpuMat& mask_);

}//device
}//btl

#endif