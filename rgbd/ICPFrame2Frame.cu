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

#define EXPORT

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#endif

#include <opencv2/cudaarithm.hpp>
//#include <opencv2/core/gpu_types.hpp>
#include <opencv2/cudev/common.hpp>
#include <opencv2/core/cuda/utility.hpp>
//#include "cv/common.hpp" //copied from opencv
#include "OtherUtil.hpp"
#include <math_constants.h>
#include "pcl/limits.hpp"
#include "pcl/device.hpp"
#include "pcl/vector_math.hpp"
#include "pcl/block.hpp"
#include <vector>
#include "ICPFrame2Frame.cuh"



namespace btl{ namespace device {
typedef double float_type;
using namespace pcl::device;
using namespace cv::cuda;
using namespace std;
//using namespace cv::cudev;

//device__ unsigned int __devuTotalFeaturePair;
//__device__ unsigned int __devuTotalICPPair;


__constant__ Intr __sCamIntr;

__constant__ Mat33  __mRwCurTrans;
__constant__ float3 __vTwCur;
__constant__ Mat33  __mRwPrev;
__constant__ float3 __vTwPrev;
__constant__ float __fDistThres;
__constant__ float _fSinAngleThres;
__constant__ float __fCosAngleThres;

__constant__ int __nCols;
__constant__ int __nRows;


struct SDeviceICPEnergyRegistration
{
    enum {
		CTA_SIZE_X = 16,
		CTA_SIZE_Y = 16,
		CTA_SIZE = CTA_SIZE_X * CTA_SIZE_Y
    };

    struct SDevPlus {
		__forceinline__ __device__ float operator () (const float_type &lhs, const volatile float_type& rhs) const {
			return (lhs + rhs);
		}
    };
	PtrStepSz<float> _depth_curr;

    PtrStepSz<float3> _cvgmVMapLocalCur;
    PtrStepSz<float3> _cvgmNMapLocalCur;
	PtrStepSz<uchar> _mask;
	PtrStepSz<uchar> _mask2;

    PtrStepSz<float3> _cvgmVMapWorldPrev;
	PtrStepSz<float3> _cvgmNMapWorldPrev;



    mutable PtrStepSz<float_type> _cvgmBuf;
	mutable PtrStepSz<float_type> _cvgmE;

	__device__ __forceinline__ bool is_normal_legal(const float3& nl_) const{
		float n = nl_.x*nl_.x + nl_.y*nl_.y + nl_.z*nl_.z;
		if (fabs(n - 1.f) > 0.5f) return false;
		else return true;
	}

	// nX_, nY, are the current frame pixel index
	// *pf3NlPrev_, *pf3PtPrev_, are the corresponding normal and position in previous frame, which is defined in world
	// *pf3PtCur_ is the current 3D point defined in world. Note that the input depth frame is defined in local camera reference frame
	__device__ __forceinline__ bool searchForCorrespondence(int nX_, int nY_, float3& f3NlWorldRef, float3& f3PtWorldRef, float3& f3PtWorldCur, float& weight) const{
		if (nX_ >= __nCols || nY_ >= __nRows) return false;
		//retrieve normal
		const float3 f3NlLocalCur = _cvgmNMapLocalCur.ptr(nY_)[nX_]; if (isnan (f3NlLocalCur.x) || isnan (f3NlLocalCur.y) || isnan (f3NlLocalCur.z) ) return false;
		//transform the current vetex to reference camera coodinate system
		float3 f3PtLocalCur = _cvgmVMapLocalCur.ptr(nY_)[nX_]; if ( isnan (f3PtLocalCur.x) || isnan (f3PtLocalCur.y) || isnan (f3PtLocalCur.z) ) return false; //retrieve vertex from current frame
		//weight = 160.f / fmaxf(fabsf( nY_ - __sCamIntr.cy ), fabsf( nX_ - __sCamIntr.cx ));
		if (_depth_curr.rows != 0){
			weight /= _depth_curr.ptr(nY_)[nX_];// (_depth_curr.ptr(nY_)[nX_] * _depth_curr.ptr(nY_)[nX_]);
		}
		f3PtWorldCur = __mRwCurTrans * (f3PtLocalCur - __vTwCur); //transform LocalCur into World
		//printf("1.1 (%f %f %f; %f %f %f; %f %f %f)\n", nX_, nY_, __mRwCurTrans.data[0].x, __mRwCurTrans.data[0].y, __mRwCurTrans.data[0].z, __mRwCurTrans.data[1].x, __mRwCurTrans.data[1].y, __mRwCurTrans.data[1].z, __mRwCurTrans.data[2].x, __mRwCurTrans.data[2].y, __mRwCurTrans.data[2].z);

		float3 f3PtCur_LocalPrev = __mRwPrev * f3PtWorldCur + __vTwPrev; //transform WorldCur into Prev Local
		//projection onto reference image
		int2 n2Ref;        
		n2Ref.x = __float2int_rd (f3PtCur_LocalPrev.x * __sCamIntr.fx / f3PtCur_LocalPrev.z + __sCamIntr.cx +.5f);  
		n2Ref.y = __float2int_rd (f3PtCur_LocalPrev.y * __sCamIntr.fy / f3PtCur_LocalPrev.z + __sCamIntr.cy +.5f);  
		//if projected out of the frame, return false
		if (n2Ref.x < 0 || n2Ref.y < 0 || n2Ref.x >= __nCols || n2Ref.y >= __nRows || f3PtCur_LocalPrev.z < 0) return false;
		//retrieve corresponding reference normal
		f3NlWorldRef = _cvgmNMapWorldPrev.ptr(n2Ref.y)[n2Ref.x];  if (isnan (f3NlWorldRef.x) || isnan (f3NlWorldRef.y) || isnan (f3NlWorldRef.z))  return false;
		//retrieve corresponding reference vertex
		f3PtWorldRef = _cvgmVMapWorldPrev.ptr(n2Ref.y)[n2Ref.x];  if (isnan (f3PtWorldRef.x) || isnan (f3PtWorldRef.y) || isnan (f3PtWorldRef.x))  return false;
		//printf("%d\t%d f3PtWorldRef (%f %f %f) (%f %f %f)\n", nX_, nY_, f3PtWorldRef.x, f3PtWorldRef.y, f3PtWorldRef.z, f3PtWorldCur.x, f3PtWorldCur.y, f3PtWorldCur.z);
		//check distance
		float fDist = norm<float, float3>(f3PtWorldRef - f3PtWorldCur); if (fDist != fDist) return false;
		//printf("%d\t%d (%f %f)\n", nX_, nY_, fDist, __fDistThres);
		if (fDist > __fDistThres+0.027*_depth_curr.ptr(nY_)[nX_])  return (false);
		//transform current normal to world
	    float3 f3NlWorldCur = __mRwCurTrans * f3NlLocalCur; 
		//check normal angle
		float fCos = dot3<float, float3>(f3NlWorldCur, f3NlWorldRef);
		//float fSin = norm ( cross(f3NlWorldCur, f3NlWorldRef) ); 
		//if (fSin >= _fSinAngleThres) return (false);
		if (fCos < __fCosAngleThres) return (false);
		//printf("1.1 nX %d nY %d  __fCosAngleThres %f)\n", nX_, nY_, __fCosAngleThres);
		return (true);
    }//searchForCorrespondence()


	// nX_, nY, are the current frame pixel index
	// *pf3NlPrev_, *pf3PtPrev_, are the corresponding normal and position in previous frame, which is defined in world
	// *pf3PtCur_ is the current 3D point defined in world. Note that the input depth frame is defined in local camera reference frame
	__device__ __forceinline__ bool searchForCorrespondence2(int nX_, int nY_, float3& f3NlWorldRef, float3& f3PtWorldRef, float3& f3PtWorldCur, float& weight) {
		if (nX_ >= __nCols || nY_ >= __nRows) return false;
		//retrieve normal
		const float3 f3NlLocalCur = _cvgmNMapLocalCur.ptr(nY_)[nX_]; if (isnan(f3NlLocalCur.x) || isnan(f3NlLocalCur.y) || isnan(f3NlLocalCur.z)) return false;
		//transform the current vetex to reference camera coodinate system
		float3 f3PtLocalCur = _cvgmVMapLocalCur.ptr(nY_)[nX_]; if (isnan(f3PtLocalCur.x) || isnan(f3PtLocalCur.y) || isnan(f3PtLocalCur.z)) return false; //retrieve vertex from current frame
		f3PtWorldCur = __mRwCurTrans * (f3PtLocalCur - __vTwCur); //transform LocalCur into World
		//printf("1.1 (%f %f %f; %f %f %f; %f %f %f)\n", nX_, nY_, __mRwCurTrans.data[0].x, __mRwCurTrans.data[0].y, __mRwCurTrans.data[0].z, __mRwCurTrans.data[1].x, __mRwCurTrans.data[1].y, __mRwCurTrans.data[1].z, __mRwCurTrans.data[2].x, __mRwCurTrans.data[2].y, __mRwCurTrans.data[2].z);
		//weight = 160.f / fmaxf(fabsf(nY_ - __sCamIntr.cy), fabsf(nX_ - __sCamIntr.cx));
		if (_depth_curr.rows != 0){
			weight /= _depth_curr.ptr(nY_)[nX_];// (_depth_curr.ptr(nY_)[nX_] * _depth_curr.ptr(nY_)[nX_]);
			//weight /= (_depth_curr.ptr(nY_)[nX_] * _depth_curr.ptr(nY_)[nX_]);
		}
		float3 f3PtCur_LocalPrev = __mRwPrev * f3PtWorldCur + __vTwPrev; //transform WorldCur into Prev Local
		//projection onto reference image
		int2 n2Ref;
		n2Ref.x = __float2int_rd(f3PtCur_LocalPrev.x * __sCamIntr.fx / f3PtCur_LocalPrev.z + __sCamIntr.cx + .5f);
		n2Ref.y = __float2int_rd(f3PtCur_LocalPrev.y * __sCamIntr.fy / f3PtCur_LocalPrev.z + __sCamIntr.cy + .5f);
		//if projected out of the frame, return false
		if (n2Ref.x < 0 || n2Ref.y < 0 || n2Ref.x >= __nCols || n2Ref.y >= __nRows || f3PtCur_LocalPrev.z < 0) return false;
		//retrieve corresponding reference normal
		f3NlWorldRef = _cvgmNMapWorldPrev.ptr(n2Ref.y)[n2Ref.x];  if (isnan(f3NlWorldRef.x) || isnan(f3NlWorldRef.y) || isnan(f3NlWorldRef.z))  return false;
		//retrieve corresponding reference vertex
		f3PtWorldRef = _cvgmVMapWorldPrev.ptr(n2Ref.y)[n2Ref.x];  if (isnan(f3PtWorldRef.x) || isnan(f3PtWorldRef.y) || isnan(f3PtWorldRef.x))  return false;
		//printf("%d\t%d f3PtWorldRef (%f %f %f) (%f %f %f)\n", nX_, nY_, f3PtWorldRef.x, f3PtWorldRef.y, f3PtWorldRef.z, f3PtWorldCur.x, f3PtWorldCur.y, f3PtWorldCur.z);
		//check distance
		float fDist = norm<float, float3>(f3PtWorldRef - f3PtWorldCur); if (fDist != fDist) return false;

		_mask2.ptr(nY_)[nX_] = 1;
		//printf("%d\t%d (%f %f)\n", nX_, nY_, fDist, __fDistThres);
		if (fDist > __fDistThres + 0.027*_depth_curr.ptr(nY_)[nX_])  return (false);
		//transform current normal to world
		float3 f3NlWorldCur = __mRwCurTrans * f3NlLocalCur;
		//check normal angle
		float fCos = dot3<float, float3>(f3NlWorldCur, f3NlWorldRef);
		//float fSin = norm ( cross(f3NlWorldCur, f3NlWorldRef) ); 
		//if (fSin >= _fSinAngleThres) return (false);
		if (fCos < __fCosAngleThres) return (false);
		//printf("1.1 nX %d nY %d  __fCosAngleThres %f)\n", nX_, nY_, __fCosAngleThres);
		return (true);
	}//searchForCorrespondence2()

	__device__ __forceinline__ void calc_energy() {
		int nX = threadIdx.x + blockIdx.x * CTA_SIZE_X;
		int nY = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

		float3 f3NlPrev, f3PtPrev, f3PtCurr;
		// read out current point in world and find its corresponding point in previous frame, which are also defined in world
		float fE = 0.f; float weight = 1.f;
		if (searchForCorrespondence2(nX, nY, f3NlPrev, f3PtPrev, f3PtCurr, weight)){
			//printf("2 nX %d nY %d PtCurr (%f %f %f)\n", nX, nY, f3PtCurr.x, f3PtCurr.y, f3PtCurr.z );
			fE = fabs( dot3<float, float3>(f3NlPrev, f3PtPrev - f3PtCurr) );
			fE *= weight;
			//printf("\t%d\t%d\t%f\n", nX, nY, fE);
			_mask.ptr(nY)[nX] = uchar(1);
			//atomicInc(&__devuTotalICPPair, -1);
		}//if correspondence found

		__shared__ float_type smem[CTA_SIZE]; // CTA_SIZE is 32*8 == the number of threads in the block
		int nThrID = Block::flattenedThreadId();

		smem[nThrID] = fE; //fill all the shared memory
		__syncthreads();

		Block::reduce<CTA_SIZE>(smem, SDevPlus());
		if (nThrID == 0) _cvgmE.ptr()[blockIdx.x + gridDim.x * blockIdx.y] = smem[0];
		return;
	}

	//32*8 threads and cols/32 * rows/8 blocks
	__device__ __forceinline__ void operator () () const {
		int nX = threadIdx.x + blockIdx.x * CTA_SIZE_X;
		int nY = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

		float row[7]; // it has to be there for all threads, otherwise, some thread will add un-initialized fE into total energy. 
		float3& f3PtPrev = *(float3*)row;
		float3& f3NlPrev = *(float3*)(row+3);
		float3 f3PtCurr;
		float weight = 1.f;
		// read out current point in world and find its corresponding point in previous frame, which are also defined in world
		//printf("2 nX %d nY %d PtCurr (%f %f %f)\n", nX, nY, f3PtCurr.x, f3PtCurr.y, f3PtCurr.z );
		if (searchForCorrespondence(nX, nY, f3NlPrev, f3PtPrev, f3PtCurr, weight)){
			row[6] = weight* dot3<float, float3>(f3NlPrev, f3PtPrev - f3PtCurr);//order matters; energy
			row[0] = weight* (f3PtCurr.y * f3NlPrev.z - f3PtCurr.z * f3NlPrev.y);//cross(f3PtCurr, f3NlPrev); 
			row[1] = weight* (f3PtCurr.z * f3NlPrev.x - f3PtCurr.x * f3NlPrev.z);
			row[2] = weight* (f3PtCurr.x * f3NlPrev.y - f3PtCurr.y * f3NlPrev.x);
			row[3] *= weight;
			row[4] *= weight;
			row[5] *= weight;
		}//if correspondence found
		else{ 
			memset(row, 0, 28);//28= 7*bits(float)
		}
		
		__shared__ float_type smem[CTA_SIZE]; // CTA_SIZE is 32*8 == the number of threads in the block
	    int nThrID = Block::flattenedThreadId ();

		int nShift = 0;
		for (int i = 0; i < 6; ++i){ //__nRows
			#pragma unroll
			for (int j = i; j < 7; ++j){ // __nCols + b
				__syncthreads ();
				smem[nThrID] = row[i] * row[j]; //fill all the shared memory
				__syncthreads ();

				Block::reduce<CTA_SIZE>(smem, SDevPlus ()); //reduce to thread 0;
				if (nThrID == 0) _cvgmBuf.ptr(nShift++)[blockIdx.x + gridDim.x * blockIdx.y] = smem[0]; //nShift < 27 = 21 + 6, upper triangle of 6x6
			}//for
		}//for
		return;
    }//operator()
};//SDeviceICPEnergyRegistration

struct STranformReduction
{
    enum{
		CTA_SIZE = 512,
		STRIDE = CTA_SIZE, // 512 threads per block

		B = 6, COLS = 6, ROWS = 6, DIAG = 6, 
		UPPER_DIAG_MAT = (COLS * ROWS - DIAG) / 2 + DIAG, 
		TOTAL = UPPER_DIAG_MAT + B,
		GRID_X = TOTAL
    };

    PtrStepSz<float_type> _cvgmBuf;
	int length; //# of blocks
    mutable float_type* pOutput;

	// 512 threeads and 27 blocks
    __device__ __forceinline__ void  operator () () const
    {
		const float_type *beg = _cvgmBuf.ptr (blockIdx.x); // 27 * # of blocks in previous kernel launch
		const float_type *end = beg + length;

		int tid = threadIdx.x;

		float_type sum = 0.f;
		for (const float_type *t = beg + tid; t < end; t += STRIDE) 
			sum += *t;

		__shared__ float_type smem[CTA_SIZE];

		smem[tid] = sum;
		__syncthreads ();

		Block::reduce<CTA_SIZE>(smem, SDeviceICPEnergyRegistration::SDevPlus ());

		if (tid == 0) pOutput[blockIdx.x] = smem[0];
    }//operator ()
};//STranformReduction

__global__ void kernel_icp_frame_2_frm ( SDeviceICPEnergyRegistration sICP ) {
    sICP ();
}

__global__ void kernel_calc_icp_energy( SDeviceICPEnergyRegistration sICP) {
	sICP.calc_energy();
}

__global__ void kernelTransformEstimator ( STranformReduction sTR ) {
	sTR ();
}

GpuMat cuda_icp_fr_2_fr(const Intr& sCamIntr_, float fDistThres_, float fCosAngleThres_,
					  const Mat33& RwCurTrans_, const float3& TwCur_, 
					  const Mat33& RwPrev_, const float3& TwPrev_, const GpuMat& depth_curr_,
					  const GpuMat& cvgmVMapWorldPrev_, const GpuMat& cvgmNMapWorldPrev_, 
					  const GpuMat& cvgmVMapLocalCur_,  const GpuMat& cvgmNMapLocalCur_){
	cudaSafeCall(cudaMemcpyToSymbol(__sCamIntr, &sCamIntr_, sizeof(Intr))); //copy host memory to constant memory on the device.
	cudaSafeCall(cudaMemcpyToSymbol(__mRwCurTrans, &RwCurTrans_, sizeof(Mat33)));
	cudaSafeCall(cudaMemcpyToSymbol(__vTwCur, &TwCur_, sizeof(float3))); //copy host memory to constant memory on the device.
	cudaSafeCall(cudaMemcpyToSymbol(__mRwPrev, &RwPrev_, sizeof(Mat33)));
	cudaSafeCall(cudaMemcpyToSymbol(__vTwPrev, &TwPrev_, sizeof(float3))); //copy host memory to constant memory on the device.
	cudaSafeCall(cudaMemcpyToSymbol(__fDistThres, &fDistThres_, sizeof(float))); //copy host memory to constant memory on the device.

	cudaSafeCall(cudaMemcpyToSymbol(__fCosAngleThres, &fCosAngleThres_, sizeof(float))); //copy host memory to constant memory on the device.
	cudaSafeCall(cudaMemcpyToSymbol(__nCols, &(cvgmVMapLocalCur_.cols), sizeof(int))); //copy host memory to constant memory on the device.
	cudaSafeCall(cudaMemcpyToSymbol(__nRows, &(cvgmVMapLocalCur_.rows), sizeof(int))); //copy host memory to constant memory on the device.

	SDeviceICPEnergyRegistration sICP;
	sICP._depth_curr = depth_curr_;
	sICP._cvgmVMapLocalCur = cvgmVMapLocalCur_;
	sICP._cvgmNMapLocalCur = cvgmNMapLocalCur_;

	sICP._cvgmVMapWorldPrev = cvgmVMapWorldPrev_;
	sICP._cvgmNMapWorldPrev = cvgmNMapWorldPrev_;

	dim3 block (SDeviceICPEnergyRegistration::CTA_SIZE_X, SDeviceICPEnergyRegistration::CTA_SIZE_Y);
    dim3 grid (1, 1, 1);
	grid.x = cv::cudev::divUp ( cvgmVMapWorldPrev_.cols, block.x );
	grid.y = cv::cudev::divUp ( cvgmVMapWorldPrev_.rows, block.y );
		
	GpuMat cvgmBuf(STranformReduction::TOTAL, grid.x * grid.y, CV_64FC1); cvgmBuf.setTo( 0. );
	//the # of rows is STranformReduction::TOTAL, 27, which is calculated in this way:
	// | 1  2  3  4  5  6  7 |
	// |    8  9 10 11 12 13 |
	// |      14 15 16 17 18 |
	// |         19 20 21 22 |
	// |            23 24 25 |
	// |               26 27 |
	//the # of cols is equal to the # of blocks.
	sICP._cvgmBuf = cvgmBuf;
	
	kernel_icp_frame_2_frm<<<grid, block>>>(sICP);
	cudaSafeCall ( cudaGetLastError () );
	//cudaSafeCall ( cudaDeviceSynchronize() );

	//for debug
	//cv::Mat Buf; cvgmBuf.download(Buf);
	//PRINT(Buf);
	//cv::Mat E; cvgmE.download(E);
	//PRINT(E);

	STranformReduction sTR;
	sTR._cvgmBuf = cvgmBuf;
	sTR.length = grid.x * grid.y; // # of the blocks
    
	GpuMat SumBuf; SumBuf.create(1, STranformReduction::TOTAL, CV_64FC1); // # of elements in 6x6 upper triangle matrix plus 6 = 27
	sTR.pOutput = (float_type*) SumBuf.data;

	kernelTransformEstimator<<<STranformReduction::TOTAL, STranformReduction::CTA_SIZE>>>(sTR);
	cudaSafeCall ( cudaGetLastError () );
	//cudaSafeCall ( cudaDeviceSynchronize () );

	return SumBuf;
}//registration()

double calc_energy_icp_fr_2_fr( const Intr& sCamIntr_, float fDistThres_, float fCosAngleThres_,
					  const Mat33& RwCurTrans_, const float3& TwCur_,
					  const Mat33& RwPrev_, const float3& TwPrev_, const GpuMat& depth_curr_,
					  const GpuMat& cvgmVMapWorldPrev_, const GpuMat& cvgmNMapWorldPrev_,
					  const GpuMat& cvgmVMapLocalCur_, const GpuMat& cvgmNMapLocalCur_, GpuMat& mask_){
	cudaSafeCall(cudaMemcpyToSymbol(__sCamIntr, &sCamIntr_, sizeof(Intr))); //copy host memory to constant memory on the device.
	cudaSafeCall(cudaMemcpyToSymbol(__mRwCurTrans, &RwCurTrans_, sizeof(Mat33)));
	cudaSafeCall(cudaMemcpyToSymbol(__vTwCur, &TwCur_, sizeof(float3))); //copy host memory to constant memory on the device.
	cudaSafeCall(cudaMemcpyToSymbol(__mRwPrev, &RwPrev_, sizeof(Mat33)));
	cudaSafeCall(cudaMemcpyToSymbol(__vTwPrev, &TwPrev_, sizeof(float3))); //copy host memory to constant memory on the device.
	cudaSafeCall(cudaMemcpyToSymbol(__fDistThres, &fDistThres_, sizeof(float))); //copy host memory to constant memory on the device.

	cudaSafeCall(cudaMemcpyToSymbol(__fCosAngleThres, &fCosAngleThres_, sizeof(float))); //copy host memory to constant memory on the device.
	cudaSafeCall(cudaMemcpyToSymbol(__nCols, &(cvgmVMapLocalCur_.cols), sizeof(int))); //copy host memory to constant memory on the device.
	cudaSafeCall(cudaMemcpyToSymbol(__nRows, &(cvgmVMapLocalCur_.rows), sizeof(int))); //copy host memory to constant memory on the device.

	assert(mask_.cols == cvgmVMapLocalCur_.cols && mask_.rows == cvgmVMapLocalCur_.rows);
	mask_.setTo(uchar(0));

	GpuMat mask2 = mask_.clone();
	mask2.setTo(0);

	SDeviceICPEnergyRegistration sICP;

	sICP._depth_curr = depth_curr_;
	sICP._cvgmVMapLocalCur = cvgmVMapLocalCur_;
	sICP._cvgmNMapLocalCur = cvgmNMapLocalCur_;

	sICP._cvgmVMapWorldPrev = cvgmVMapWorldPrev_;
	sICP._cvgmNMapWorldPrev = cvgmNMapWorldPrev_;

	sICP._mask = mask_;
	sICP._mask2= mask2;

	dim3 block(SDeviceICPEnergyRegistration::CTA_SIZE_X, SDeviceICPEnergyRegistration::CTA_SIZE_Y);
	dim3 grid(1, 1, 1);
	grid.x = cv::cudev::divUp(cvgmVMapWorldPrev_.cols, block.x);
	grid.y = cv::cudev::divUp(cvgmVMapWorldPrev_.rows, block.y);

	GpuMat cvgmE(1, grid.x * grid.y, CV_64FC1); cvgmE.setTo(0.);
	sICP._cvgmE = cvgmE;
	//void* pTotalICP;
	//cudaSafeCall(cudaGetSymbolAddress(&pTotalICP, __devuTotalICPPair));
	//cudaSafeCall(cudaMemset(pTotalICP, 0, sizeof(unsigned int)));

	kernel_calc_icp_energy << <grid, block >> >(sICP);
	cudaSafeCall(cudaGetLastError());
	//cudaSafeCall(cudaDeviceSynchronize());


	double dEnergy = sum(cvgmE)[0];
	//unsigned int uTotalICPPairs;
	//cudaSafeCall(cudaMemcpy(&uTotalICPPairs, pTotalICP, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	int nPairs = sum(mask_)[0];
	int total = sum(mask2)[0];

	//cout << "total = " << total << endl;
	//cout << nPairs << "\t";
	//The following equations are come from equation (10) in
	//Chetverikov, D., Stepanov, D., & Krsek, P. (2005). 
	//Robust Euclidean alignment of 3D point sets: the trimmed iterative closest point algorithm. 
	//IVC, 23(3), 299¨C309. doi:10.1016/j.imavis.2004.05.007
	dEnergy /= nPairs;
	float xee = float(nPairs) / float(total);// (cvgmVMapLocalCur_.cols * cvgmVMapLocalCur_.rows);
	dEnergy /= (xee*xee*xee);
	//cout << "energy = " << dEnergy << endl;
	return dEnergy;
}//registration()

}//device
}//btl