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
#define EXPORT

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#endif

#include <thrust/device_ptr.h> 
#include <thrust/sort.h> 


#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda/common.hpp>
#include "OtherUtil.hpp"
#include <math_constants.h>
#include "pcl/limits.hpp"
#include "pcl/device.hpp"
#include "pcl/vector_math.hpp"
#include <vector>
#include "CudaLib.cuh"
//#include "../KeyFrame.h"
//#include <boost/shared_ptr.hpp>

using namespace cv;
using namespace cv::cuda;

namespace btl{ namespace device		
{

//depth to disparity
__global__ void kernelInverse(const cv::cuda::PtrStepSz<float> cvgmIn_, cv::cuda::PtrStepSz<float> cvgmOut_){
    const int nX = blockDim.x * blockIdx.x + threadIdx.x;
    const int nY = blockDim.y * blockIdx.y + threadIdx.y;
	if (nX >= cvgmIn_.cols || nY >= cvgmIn_.rows) return;
	if(fabsf(cvgmIn_.ptr(nY)[nX]) > 0.01f )
		cvgmOut_.ptr(nY)[nX] = 1.f/cvgmIn_.ptr(nY)[nX];
	else
		cvgmOut_.ptr(nY)[nX] = 0;//pcl::device::numeric_limits<float>::quiet_NaN();
}//kernelInverse

void cudaDepth2Disparity( const cv::cuda::GpuMat& cvgmDepth_, cv::cuda::GpuMat* pcvgmDisparity_ ){
	//not necessary as pcvgmDisparity has been allocated in VideoSourceKinect()
	//pcvgmDisparity_->create(cvgmDepth_.size(),CV_32F);
	//define grid and block
	dim3 block(32, 8);
    dim3 grid(cv::cuda::device::divUp(cvgmDepth_.cols, block.x), cv::cuda::device::divUp(cvgmDepth_.rows, block.y));
	//run kernel
	kernelInverse<<<grid,block>>>( cvgmDepth_,*pcvgmDisparity_ );
	//cudaSafeCall ( cudaGetLastError () );
	//cudaSafeCall(cudaDeviceSynchronize());
	return;
}//cudaDepth2Disparity

__global__ void kernelInverse2(const cv::cuda::PtrStepSz<float> cvgmIn_, float fCutOffDistance_, float factor_, cv::cuda::PtrStepSz<float> cvgmOut_){
    const int nX = blockDim.x * blockIdx.x + threadIdx.x;
    const int nY = blockDim.y * blockIdx.y + threadIdx.y;
	//if (nX <2)
		//printf("nX %d, nY %d, out %f\n", nX, nY);
	if (nX >= cvgmIn_.cols || nY >= cvgmIn_.rows) return;

	if (fabsf(cvgmIn_.ptr(nY)[nX]) > 0.01f && cvgmIn_.ptr(nY)[nX] < fCutOffDistance_){
		float tmp = factor_ / cvgmIn_.ptr(nY)[nX];
		cvgmOut_.ptr(nY)[nX] = tmp;
	}
	else{
		cvgmOut_.ptr(nY)[nX] = 0; //pcl::device::numeric_limits<float>::quiet_NaN();
	}
}//kernelInverse

void cuda_depth2disparity2( const cv::cuda::GpuMat& cvgmDepth_, float fCutOffDistance_, cv::cuda::GpuMat* pcvgmDisparity_, float factor_ /*= 1000.f*/){
	//convert the depth from mm to m
	//not necessary as pcvgmDisparity has been allocated in VideoSourceKinect()
	//pcvgmDisparity_->create(cvgmDepth_.size(),CV_32F);
	//define grid and block
	fCutOffDistance_ *= factor_; 
	dim3 block(32, 8);
    dim3 grid(cv::cuda::device::divUp(cvgmDepth_.cols, block.x), cv::cuda::device::divUp(cvgmDepth_.rows, block.y));
	//run kernel
	kernelInverse2<<<grid,block>>>( cvgmDepth_,fCutOffDistance_,factor_, *pcvgmDisparity_ );
	//cudaSafeCall(cudaDeviceSynchronize());
	//cudaSafeCall(cudaGetLastError());
	return;
}//cudaDepth2Disparity


void cuda_disparity2depth( const cv::cuda::GpuMat& cvgmDisparity_, cv::cuda::GpuMat* pcvgmDepth_ ){
	pcvgmDepth_->create(cvgmDisparity_.size(),CV_32F);
	//define grid and block
	dim3 block(32, 8);
    dim3 grid(cv::cuda::device::divUp(cvgmDisparity_.cols, block.x), cv::cuda::device::divUp(cvgmDisparity_.rows, block.y));
	//run kernel
	kernelInverse<<<grid,block>>>( cvgmDisparity_,*pcvgmDepth_ );
	//cudaSafeCall ( cudaGetLastError () );
	//cudaSafeCall(cudaDeviceSynchronize());
	return;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//global constant used by kernelUnprojectIR() and cudaUnProjectIR()
__constant__ float _aIRCameraParameter[4];// 1/f_x, 1/f_y, u, v for IR camera; constant memory declaration
__constant__ float _aR[9];
__constant__ float _aRT[3];
//global constant used by kernelProjectRGB() and cudaProjectRGB()
__constant__ float _aRGBCameraParameter[4]; //fFxRGB_,fFyRGB_,uRGB_,vRGB_
__constant__ float _aSigma2InvHalf[2]; //sigma_space2_inv_half,sigma_color2_inv_half

__global__ void kernelBilateral (const cv::cuda::PtrStepSz<float> src, cv::cuda::PtrStepSz<float> dst )
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= src.cols || y >= src.rows)  return;

    const int R = 2;//static_cast<int>(sigma_space * 1.5);
    const int D = R * 2 + 1;

    float fValueCentre = src.ptr (y)[x];
	//if fValueCentre is NaN
	if(fabs( fValueCentre ) < 0.00001f) return; 

    int tx = min (x - D/2 + D, src.cols - 1);
    int ty = min (y - D/2 + D, src.rows - 1);

    float sum1 = 0;
    float sum2 = 0;

    for (int cy = max (y - D/2, 0); cy < ty; ++cy)
    for (int cx = max (x - D/2, 0); cx < tx; ++cx){
        float  fValueNeighbour = src.ptr (cy)[cx];
		//if fValueNeighbour is NaN
		//if(fabs( fValueNeighbour - fValueCentre ) > 0.00005f) continue; 
        float space2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
        float color2 = (fValueCentre - fValueNeighbour) * (fValueCentre - fValueNeighbour);
        float weight = __expf (-(space2 * _aSigma2InvHalf[0] + color2 * _aSigma2InvHalf[1]) );

        sum1 += fValueNeighbour * weight;
        sum2 += weight;
    }//for for each pixel in neigbbourhood

    dst.ptr (y)[x] = sum1/sum2;
	return;
}//kernelBilateral

void cuda_bilateral_filtering(const cv::cuda::GpuMat& cvgmSrc_, const float& fSigmaSpace_, const float& fSigmaColor_, cv::cuda::GpuMat* pcvgmDst_ )
{
	pcvgmDst_->setTo(0);// (std::numeric_limits<float>::quiet_NaN());
	//constant definition
	size_t sN = sizeof(float) * 2;
	float* const pSigma = (float*) malloc( sN );
	pSigma[0] = 0.5f / (fSigmaSpace_ * fSigmaSpace_);
	pSigma[1] = 0.5f / (fSigmaColor_ * fSigmaColor_);
	cudaSafeCall( cudaMemcpyToSymbol(_aSigma2InvHalf, pSigma, sN) );
	//define grid and block
	dim3 block(32, 8);
    dim3 grid(cv::cuda::device::divUp(cvgmSrc_.cols, block.x), cv::cuda::device::divUp(cvgmSrc_.rows, block.y));
	//run kernel
    kernelBilateral<<<grid,block>>>( cvgmSrc_,*pcvgmDst_ );
	//cudaSafeCall( cudaGetLastError () );
	//cudaSafeCall( cudaDeviceSynchronize() );

	//release temporary pointers
	free(pSigma);
	return;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernelPyrDown (const cv::cuda::PtrStepSz<float> cvgmSrc_, cv::cuda::PtrStepSz<float> cvgmDst_, float fSigmaColor_ )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cvgmDst_.cols || y >= cvgmDst_.rows) return;

    const int D = 5;

    float center = cvgmSrc_.ptr (2 * y)[2 * x];
	if( isnan<float>(center) ){//center!=center ){
		cvgmDst_.ptr (y)[x] = pcl::device::numeric_limits<float>::quiet_NaN();
		return;
	}//if center is NaN
    int tx = min (2 * x - D / 2 + D, cvgmSrc_.cols - 1); //ensure tx <= cvgmSrc.cols-1
    int ty = min (2 * y - D / 2 + D, cvgmSrc_.rows - 1); //ensure ty <= cvgmSrc.rows-1
    int cy = max (0, 2 * y - D / 2);

    float sum = 0;
    int count = 0;

    for (; cy < ty; ++cy)
    for (int cx = max (0, 2 * x - D / 2); cx < tx; ++cx) {
        float val = cvgmSrc_.ptr (cy)[cx];
        if (fabsf (val - center) < 3 * fSigmaColor_){//
			sum += val;
			++count;
        } //if within 3*fSigmaColor_
    }//for each pixel in the neighbourhood 5x5
    cvgmDst_.ptr (y)[x] = sum / count;
}//kernelPyrDown()

void cuda_pyr_down (const cv::cuda::GpuMat& cvgmSrc_, const float& fSigmaColor_, cv::cuda::GpuMat* pcvgmDst_)
{
	dim3 block (32, 8);
	dim3 grid (cv::cuda::device::divUp (pcvgmDst_->cols, block.x), cv::cuda::device::divUp (pcvgmDst_->rows, block.y));
	kernelPyrDown<<<grid, block>>>(cvgmSrc_, *pcvgmDst_, fSigmaColor_);
	//cudaSafeCall ( cudaGetLastError () );
	//cudaSafeCall(cudaDeviceSynchronize());
	return;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernelUnprojectRGBCVmCVm (const cv::cuda::PtrStepSz<float> cvgmDepths_, const unsigned short uScale_, cv::cuda::PtrStepSz<float3> cvgmPts_ )
{
    const int nX = blockDim.x * blockIdx.x + threadIdx.x;
    const int nY = blockDim.y * blockIdx.y + threadIdx.y;

    if (nX >= cvgmPts_.cols || nY >= cvgmPts_.rows)  return;

	float3& pt = cvgmPts_.ptr(nY)[nX];
	const float fDepth = cvgmDepths_.ptr(nY)[nX];

	if( 0.4f < fDepth && fDepth < 10.f ){
		pt.z = fDepth;
		pt.x = ( nX*uScale_  - _aRGBCameraParameter[2] ) * _aRGBCameraParameter[0] * pt.z; //_aRGBCameraParameter[0] is 1.f/fFxRGB_
		pt.y = ( nY*uScale_  - _aRGBCameraParameter[3] ) * _aRGBCameraParameter[1] * pt.z; 
	}
	else {
		pt.x = pt.y = pt.z = pcl::device::numeric_limits<float>::quiet_NaN();
	}
	return;
}
void cuda_unproject_rgb ( const cv::cuda::GpuMat& cvgmDepths_, 
						const float& fFxRGB_,const float& fFyRGB_,const float& uRGB_, const float& vRGB_, unsigned int uLevel_, 
						cv::cuda::GpuMat* pcvgmPts_ )
{
	unsigned short uScale = 1<< uLevel_;
	pcvgmPts_->setTo(0);
	//constant definition
	size_t sN = sizeof(float) * 4;
	float* const pRGBCameraParameters = (float*) malloc( sN );
	pRGBCameraParameters[0] = 1.f/fFxRGB_;
	pRGBCameraParameters[1] = 1.f/fFyRGB_;
	pRGBCameraParameters[2] = uRGB_;
	pRGBCameraParameters[3] = vRGB_;
	cudaSafeCall( cudaMemcpyToSymbol(_aRGBCameraParameter, pRGBCameraParameters, sN) );
	
	dim3 block (32, 8);
	dim3 grid (cv::cuda::device::divUp (pcvgmPts_->cols, block.x), cv::cuda::device::divUp (pcvgmPts_->rows, block.y));
	kernelUnprojectRGBCVmCVm<<<grid, block>>>(cvgmDepths_, uScale, *pcvgmPts_ );
	//cudaSafeCall ( cudaGetLastError () );
	//cudaSafeCall(cudaDeviceSynchronize());
	return;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernelFastNormalEstimation (const cv::cuda::PtrStepSz<float3> cvgmPts_, cv::cuda::PtrStepSz<float3> cvgmNls_ )
{
    const int nX = blockDim.x * blockIdx.x + threadIdx.x;
    const int nY = blockDim.y * blockIdx.y + threadIdx.y;

    if (nX >= cvgmPts_.cols || nY >= cvgmPts_.rows ) return;
	float3& fN = cvgmNls_.ptr(nY)[nX];
	if (nX == cvgmPts_.cols - 1 || nY >= cvgmPts_.rows - 1 ){
		fN.x = fN.y = fN.z = pcl::device::numeric_limits<float>::quiet_NaN();
		return;
	}
	const float3& pt = cvgmPts_.ptr(nY)[nX];
	const float3& pt1= cvgmPts_.ptr(nY)[nX+1]; //right 
	const float3& pt2= cvgmPts_.ptr(nY+1)[nX]; //down

	if(isnan<float>(pt.z) ||isnan<float>(pt1.z) ||isnan<float>(pt2.z) ){
		fN.x = fN.y = fN.z = pcl::device::numeric_limits<float>::quiet_NaN();
		return;
	}//if input or its neighour is NaN,
	float3 v1;
	v1.x = pt1.x-pt.x;
	v1.y = pt1.y-pt.y;
	v1.z = pt1.z-pt.z;
	float3 v2;
	v2.x = pt2.x-pt.x;
	v2.y = pt2.y-pt.y;
	v2.z = pt2.z-pt.z;
	//n = v1 x v2 cross product
	float3 n;
	n.x = v1.y*v2.z - v1.z*v2.y;
	n.y = v1.z*v2.x - v1.x*v2.z;
	n.z = v1.x*v2.y - v1.y*v2.x;
	//normalization
	float norm = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);

	if( norm < 1.0e-10 ) {
		fN.x = fN.y = fN.z = pcl::device::numeric_limits<float>::quiet_NaN();
		return;
	}//set as NaN,
	n.x /= norm;
	n.y /= norm;
	n.z /= norm;

	if( -n.x*pt.x - n.y*pt.y - n.z*pt.z <0 ){ //this gives (0-pt).dot3( n ); 
		fN.x = -n.x;
		fN.y = -n.y;
		fN.z = -n.z;
	}//if facing away from the camera
	else{
		fN.x = n.x;
		fN.y = n.y;
		fN.z = n.z;
	}//else
	return;
}

void cuda_fast_normal_estimation(const cv::cuda::GpuMat& cvgmPts_, cv::cuda::GpuMat* pcvgmNls_ )
{
	pcvgmNls_->setTo(0);
	dim3 block (32, 8);
	dim3 grid (cv::cuda::device::divUp (cvgmPts_.cols, block.x), cv::cuda::device::divUp (cvgmPts_.rows, block.y));
	kernelFastNormalEstimation<<<grid, block>>>(cvgmPts_, *pcvgmNls_ );
	//cudaSafeCall ( cudaGetLastError () );
	//cudaSafeCall(cudaDeviceSynchronize());

}

__global__ void kernelScaleDepthCVmCVm (cv::cuda::PtrStepSz<float> cvgmDepth_, const pcl::device::Intr sCameraIntrinsics_)
{
    int nX = threadIdx.x + blockIdx.x * blockDim.x;
    int nY = threadIdx.y + blockIdx.y * blockDim.y;

    if (nX >= cvgmDepth_.cols || nY >= cvgmDepth_.rows)  return;

    float& fDepth = cvgmDepth_.ptr(nY)[nX];
    float fTanX = (nX - sCameraIntrinsics_.cx) / sCameraIntrinsics_.fx;
    float fTanY = (nY - sCameraIntrinsics_.cy) / sCameraIntrinsics_.fy;
    float fSec = sqrtf (fTanX*fTanX + fTanY*fTanY + 1);
    fDepth *= fSec; //meters
}//kernelScaleDepthCVmCVm()
//scaleDepth is to transform raw depth into scaled depth which is the distance from the 3D point to the camera centre
//     *---* 3D point
//     |  / 
//raw  | /scaled depth
//depth|/
//     * camera center
//
void cuda_scale_depth(unsigned short usPyrLevel_, const float fFx_, const float fFy_, const float u_, const float v_, cv::cuda::GpuMat* pcvgmDepth_){
	pcl::device::Intr sCameraIntrinsics(fFx_,fFy_,u_,v_);
	dim3 block(32, 8);
    dim3 grid(cv::cuda::device::divUp(pcvgmDepth_->cols, block.x), cv::cuda::device::divUp(pcvgmDepth_->rows, block.y));
	kernelScaleDepthCVmCVm<<< grid,block >>>(*pcvgmDepth_, sCameraIntrinsics(usPyrLevel_) );
	//cudaSafeCall ( cudaGetLastError () );
	//cudaSafeCall(cudaDeviceSynchronize());
	return;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__constant__ float _aRwTrans[9];//row major 
__constant__ float _aTw[3]; 
__global__ void kernelTransformLocalToWorldCVCV(cv::cuda::PtrStepSz<float3> cvgmPts_, cv::cuda::PtrStepSz<float3> cvgmNls_, cv::cuda::PtrStepSz<float3> cvgmMDs_){ 
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
    int nY = threadIdx.y + blockIdx.y * blockDim.y;
    if (nX >= cvgmPts_.cols || nY >= cvgmPts_.rows)  return;
	//convert Pts
	float3& Pt = cvgmPts_.ptr(nY)[nX];
	float3 PtTmp; 
	//PtTmp = X_c - Tw
	PtTmp.x = Pt.x - _aTw[0];
	PtTmp.y = Pt.y - _aTw[1];
	PtTmp.z = Pt.z - _aTw[2];
	//Pt = RwTrans * PtTmp
	Pt.x = _aRwTrans[0]*PtTmp.x + _aRwTrans[1]*PtTmp.y + _aRwTrans[2]*PtTmp.z;
	Pt.y = _aRwTrans[3]*PtTmp.x + _aRwTrans[4]*PtTmp.y + _aRwTrans[5]*PtTmp.z;
	Pt.z = _aRwTrans[6]*PtTmp.x + _aRwTrans[7]*PtTmp.y + _aRwTrans[8]*PtTmp.z;
	{
		//convert Nls
		float3& Nl = cvgmNls_.ptr(nY)[nX];
		float3 NlTmp;
		//Nlw = RwTrans*Nlc
		NlTmp.x = _aRwTrans[0]*Nl.x + _aRwTrans[1]*Nl.y + _aRwTrans[2]*Nl.z;
		NlTmp.y = _aRwTrans[3]*Nl.x + _aRwTrans[4]*Nl.y + _aRwTrans[5]*Nl.z;
		NlTmp.z = _aRwTrans[6]*Nl.x + _aRwTrans[7]*Nl.y + _aRwTrans[8]*Nl.z;
		Nl = NlTmp;
	}

	if( cvgmMDs_.cols != 0 && cvgmMDs_.rows != 0 ){
		float3& MD = cvgmMDs_.ptr(nY)[nX];
		float3 MDTmp;
		//MDw = RwTrans*MDc
		MDTmp.x = _aRwTrans[0]*MD.x + _aRwTrans[1]*MD.y + _aRwTrans[2]*MD.z;
		MDTmp.y = _aRwTrans[3]*MD.x + _aRwTrans[4]*MD.y + _aRwTrans[5]*MD.z;
		MDTmp.z = _aRwTrans[6]*MD.x + _aRwTrans[7]*MD.y + _aRwTrans[8]*MD.z;
		MD = MDTmp;
	}
	return;
}//kernelTransformLocalToWorld()
void cuda_transform_local2world(const float* pRw_/*col major*/, const float* pTw_, cv::cuda::GpuMat* pcvgmPts_, cv::cuda::GpuMat* pcvgmNls_, cv::cuda::GpuMat* pcvgmMDs_){
	if ( pcvgmPts_->cols == 0 || pcvgmNls_->cols == 0 ) return;
	size_t sN1 = sizeof(float) * 9;
	cudaSafeCall( cudaMemcpyToSymbol(_aRwTrans, pRw_, sN1) );
	size_t sN2 = sizeof(float) * 3;
	cudaSafeCall( cudaMemcpyToSymbol(_aTw, pTw_, sN2) );
	dim3 block(32, 8);
    dim3 grid(cv::cuda::device::divUp(pcvgmPts_->cols, block.x), cv::cuda::device::divUp(pcvgmPts_->rows, block.y));
	kernelTransformLocalToWorldCVCV<<<grid,block>>>(*pcvgmPts_,*pcvgmNls_, *pcvgmMDs_);
	//cudaSafeCall ( cudaGetLastError () );
	//cudaSafeCall(cudaDeviceSynchronize());
	return;
}//transformLocalToWorld()
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<bool normalize>
__global__ void kernelResizeMap (const cv::cuda::PtrStepSz<float3> cvgmSrc_, cv::cuda::PtrStepSz<float3> cvgmDst_)
{
	using namespace pcl::device;
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= cvgmDst_.cols || y >= cvgmDst_.rows) return;

    float3 qnan; qnan.x = qnan.y = qnan.z = pcl::device::numeric_limits<float>::quiet_NaN ();

    int xs = x * 2;
    int ys = y * 2;

    float3 x00 = cvgmSrc_.ptr (ys + 0)[xs + 0];
    float3 x01 = cvgmSrc_.ptr (ys + 0)[xs + 1];
    float3 x10 = cvgmSrc_.ptr (ys + 1)[xs + 0];
    float3 x11 = cvgmSrc_.ptr (ys + 1)[xs + 1];

    if (isnan (x00.x) || isnan (x01.x) || isnan (x10.x) || isnan (x11.x))
    {
		cvgmDst_.ptr (y)[x] = qnan;
		return;
    }
    else
    {
		float3 n;

		n = (x00 + x01 + x10 + x11) / 4;

		if (normalize)
			n = normalized<float, float3>(n);

		cvgmDst_.ptr (y)[x] = n;
    }
}//kernelResizeMap()

void cuda_resize_map (bool bNormalize_, const cv::cuda::GpuMat& cvgmSrc_, cv::cuda::GpuMat* pcvgmDst_ )
{
    int in_cols = cvgmSrc_.cols;
    int in_rows = cvgmSrc_.rows;

    int out_cols = in_cols / 2;
    int out_rows = in_rows / 2;

    pcvgmDst_->create (out_rows, out_cols,cvgmSrc_.type());

    dim3 block (32, 8);
    dim3 grid (cv::cuda::device::divUp (out_cols, block.x), cv::cuda::device::divUp (out_rows, block.y));
	if(bNormalize_)
		kernelResizeMap<true><<<grid, block>>>(cvgmSrc_, *pcvgmDst_);
	else
		kernelResizeMap<false><<<grid, block>>>(cvgmSrc_, *pcvgmDst_);
	//cudaSafeCall ( cudaGetLastError () );
    //cudaSafeCall (cudaDeviceSynchronize ());
	return;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernelShapeClassifier(const float fThreshold, const cv::cuda::PtrStepSz<float3> cvgmPt_, const cv::cuda::PtrStepSz<float3> cvgmNl_, cv::cuda::PtrStepSz<uchar3> cvgmRGB_){
	using namespace pcl::device;
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
    int nY = threadIdx.y + blockIdx.y * blockDim.y;
    if (nX ==0 || nY == 0 || nX >= cvgmRGB_.cols-1 || nY >= cvgmRGB_.rows-1)  return;

	const float3& Pt = cvgmPt_.ptr(nY)[nX];
	
	if(isnan<float>(Pt.x)) return;

	const float3& Nl = cvgmNl_.ptr(nY)[nX];
	uchar3& RGB= cvgmRGB_.ptr(nY)[nX];
	
	const float3& Left= cvgmPt_.ptr(nY)[nX-1];
	const float3& LeftN=cvgmNl_.ptr(nY)[nX-1];

	const float3& Right= cvgmPt_.ptr(nY)[nX+1];
	const float3& Up= cvgmPt_.ptr(nY-1)[nX];
	const float3& Down= cvgmPt_.ptr(nY+1)[nX-1];
	//line-line intersection
	//http://en.wikipedia.org/wiki/Line-line_intersection
	float3 OutProduct[3];
	outProductSelf<float3>( LeftN, OutProduct);
	
	{//if it is a border pixel
		RGB = RGB*0.5 + make_uchar3(0,255,0)*0.5;
	}
	return;
}//kernelShapeClassifier()
void shapeClassifier(const float fThreshold_, const cv::cuda::GpuMat& cvgmPt_, const cv::cuda::GpuMat& cvgmNl_, cv::cuda::GpuMat* pcvgmRGB_){
	dim3 block(32, 8);
    dim3 grid(cv::cuda::device::divUp(pcvgmRGB_->cols, block.x), cv::cuda::device::divUp(pcvgmRGB_->rows, block.y));
	kernelShapeClassifier<<<grid,block>>>(fThreshold_,cvgmPt_,cvgmNl_,*pcvgmRGB_);
	//cudaSafeCall ( cudaGetLastError () );
	//cudaSafeCall(cudaDeviceSynchronize());
	return;
}//boundaryDetector()
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
void cudaConvert(const cv::cuda::PtrStepSz<float3> cvgmSrc_, cv::cuda::PtrStepSz<float3> cvgmDst_){
}



__global__ void kernel_init_idx(PtrStepSz<float> gpu_idx_){
	//traverse each element of the matrix
	const int nC = blockDim.x * blockIdx.x + threadIdx.x;
	if (nC >= gpu_idx_.cols) return;
	gpu_idx_.ptr()[nC] = nC;
	return;
}


void cuda_init_idx(int nCols_, int type_, GpuMat* p_idx_){
	if (p_idx_->empty()) p_idx_->create(1, nCols_, type_);

	dim3 block(64, 1);
	dim3 grid(1, 1, 1);
	grid.x = cv::cuda::device::divUp(nCols_, block.x);
	kernel_init_idx <<< grid, block >>> (*p_idx_);
	//cudaSafeCall(cudaGetLastError());
	//cudaSafeCall(cudaDeviceSynchronize());

	return;
}

void cuda_sort_column_idx(cv::cuda::GpuMat* const & p_key_, GpuMat* const & p_idx_){

	cuda_init_idx(p_key_->cols, p_key_->type(), p_idx_);

	thrust::device_ptr<float> X((float*)p_key_->data);
	thrust::device_ptr<float> V((float*)p_idx_->data);

	thrust::sort_by_key( X, X + p_key_->cols, V, thrust::greater<float>() );

	return;
}


enum
{
	kx = 5,
	ky = 5,
	STEP = 1
};

__global__ void kernel_estimate_normal_eigen(int rows, int cols, const PtrStep<float3> vmap, PtrStep<float3> nmap)
{
	const int u = threadIdx.x + blockIdx.x * blockDim.x;
	const int v = threadIdx.y + blockIdx.y * blockDim.y;

	if (u >= cols || v >= rows)	return;

	nmap.ptr(v)[u].x = pcl::device::numeric_limits<float>::quiet_NaN();

	float3 vt = vmap.ptr(v)[u];
	if (isnan(vt.x))
		return;

	int ty = min(v - ky / 2 + ky, rows - 1);
	int tx = min(u - kx / 2 + kx, cols - 1);

	float3 centroid = make_float3(0.f, 0.f, 0.f);
	int counter = 0;
	for (int cy = max(v - ky / 2, 0); cy < ty; cy += STEP)
		for (int cx = max(u - kx / 2, 0); cx < tx; cx += STEP)
		{
			float3 v_x = vmap.ptr(cy)[cx];
			if (!isnan(v_x.x))
			{
				centroid = centroid + v_x;
				++counter;
			}
		}

	if (counter < kx * ky / 2)
		return;

	centroid *= 1.f / counter;

	float cov[] = { 0, 0, 0, 0, 0, 0 };

	for (int cy = max(v - ky / 2, 0); cy < ty; cy += STEP)
		for (int cx = max(u - kx / 2, 0); cx < tx; cx += STEP)
		{
			float3 v;
			v = vmap.ptr(cy)[cx];
			if (isnan(v.x))
				continue;

			float3 d = v - centroid;

			cov[0] += d.x * d.x;               //cov (0, 0)
			cov[1] += d.x * d.y;               //cov (0, 1)
			cov[2] += d.x * d.z;               //cov (0, 2)
			cov[3] += d.y * d.y;               //cov (1, 1)
			cov[4] += d.y * d.z;               //cov (1, 2)
			cov[5] += d.z * d.z;               //cov (2, 2)
		}

	typedef Eigen33::Mat33 Mat33;
	Eigen33 eigen33(cov);

	Mat33 tmp;
	Mat33 vec_tmp;
	Mat33 evecs;
	float3 evals;
	eigen33.compute(tmp, vec_tmp, evecs, evals);

	float3 n = normalized<float, float3>(evecs[0]);

	if (dot3<float, float3>(vt, n) <= 0)
		nmap.ptr(v)[u] = n;
	else
		nmap.ptr(v)[u] = make_float3(-n.x, -n.y, -n.z);
	return;
}


__global__ void kernel_estimate_normal_eigen(int rows, int cols, const PtrStep<float3> vmap, PtrStep<float3> nmap, PtrStep<float> reliability)
{
	const int u = threadIdx.x + blockIdx.x * blockDim.x;
	const int v = threadIdx.y + blockIdx.y * blockDim.y;

	if (u >= cols || v >= rows)	return;

	nmap.ptr(v)[u].x = pcl::device::numeric_limits<float>::quiet_NaN();
	reliability.ptr(v)[u] = 0;// pcl::device::numeric_limits<float>::quiet_NaN();

	float3 vt = vmap.ptr(v)[u];
	if (isnan(vt.x))
		return;

	int ty = min(v - ky / 2 + ky, rows - 1);
	int tx = min(u - kx / 2 + kx, cols - 1);

	float3 centroid = make_float3(0.f, 0.f, 0.f);
	int counter = 0;
	for (int cy = max(v - ky / 2, 0); cy < ty; cy += STEP)
		for (int cx = max(u - kx / 2, 0); cx < tx; cx += STEP)
		{
			float3 v_x = vmap.ptr(cy)[cx];
			if (!isnan(v_x.x))
			{
				centroid = centroid + v_x;
				++counter;
			}
		}

	if (counter < kx * ky / 2)
		return;

	centroid *= 1.f / counter;

	float cov[] = { 0, 0, 0, 0, 0, 0 };

	for (int cy = max(v - ky / 2, 0); cy < ty; cy += STEP)
		for (int cx = max(u - kx / 2, 0); cx < tx; cx += STEP)
		{
			float3 v;
			v = vmap.ptr(cy)[cx];
			if (isnan(v.x))
				continue;

			float3 d = v - centroid;

			cov[0] += d.x * d.x;               //cov (0, 0)
			cov[1] += d.x * d.y;               //cov (0, 1)
			cov[2] += d.x * d.z;               //cov (0, 2)
			cov[3] += d.y * d.y;               //cov (1, 1)
			cov[4] += d.y * d.z;               //cov (1, 2)
			cov[5] += d.z * d.z;               //cov (2, 2)
		}

	typedef Eigen33::Mat33 Mat33;
	Eigen33 eigen33(cov);

	Mat33 tmp;
	Mat33 vec_tmp;
	Mat33 evecs;
	float3 evals;
	eigen33.compute(tmp, vec_tmp, evecs, evals);

	float3 n = normalized<float, float3>(evecs[0]);

	if (dot3<float, float3>(vt, n) <= 0)
		nmap.ptr(v)[u] = n;
	else
		nmap.ptr(v)[u] = make_float3(-n.x, -n.y, -n.z);

	evals.z = abs(evals.z);
	evals.y = abs(evals.y);
	evals.x = abs(evals.x);

	float re = (evals.y - evals.x) / (evals.x + evals.y);
	re = re >= 1.f ? 1.f : re;
	reliability.ptr(v)[u] = re;

	return;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void cuda_estimate_normals(const GpuMat& vmap, GpuMat* nmap, GpuMat* reliability /* = NULL*/)
{
	int cols = vmap.cols;
	int rows = vmap.rows;

	dim3 block(32, 8);
	dim3 grid(1, 1, 1);
	grid.x = cv::cuda::device::divUp(cols, block.x);
	grid.y = cv::cuda::device::divUp(rows, block.y);

	if (!reliability)
		kernel_estimate_normal_eigen << <grid, block >> >(rows, cols, vmap, *nmap);
	else
		kernel_estimate_normal_eigen << <grid, block >> >(rows, cols, vmap, *nmap, *reliability);

	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	return;
}

}//device
}//btl
