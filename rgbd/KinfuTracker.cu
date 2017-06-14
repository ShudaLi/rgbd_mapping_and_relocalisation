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


#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <cuda.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/cudev/functional/functional.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaarithm.hpp>

#include "pcl/device.hpp"
#include "KinfuTracker.cuh"
#define INFO
#include "OtherUtil.hpp"
#include "assert.h"
#include "CudaLib.cuh"

namespace btl{ namespace device{

using namespace std;
using namespace cv;
using namespace cv::cuda;

__device__ int __devnTotal;
__device__ int __devnTotal_lv1;
__device__ int __devnTotal_lv2;
__device__ int __devnTotal_lv3;
__device__ int __devnTotal_lv4;



__device__ __host__ __forceinline__ float axial_noise_kinect(float theta_, float z_){
	float sigma_a;
	if (fabs(theta_) <= 1.0472f) //float(M_PI / 3.)
		sigma_a = .0012f + .0019f*(z_ - 0.4f)*(z_ - 0.4f);
	else
		sigma_a = .0012f + .0019f*(z_ - 0.4f)*(z_ - 0.4f) + .0001f * theta_* theta_ / sqrt(z_) / ( 1.5708f - theta_) / ( 1.5708f - theta_);
	return sigma_a;
}

//- Collect most similar pairs of 3D points according to its hamming distances
//- Only those matching differences is smaller than 60 (matchThreshold) and 
//  the 1st match is smaller than 60% (fPercentage) of 2nd match.
struct SDevCollect{
	enum{ IDF = 0, Jose, GM, GM_GPU, GM_GPU_IM, BUILD_GT };

	float _fMatchThreshold;
	float _fPercentage;
	bool _bGlobalModel;
	int _nMatchingMethod;
	int _nOffset;
	int _nKeyPoints;

	PtrStepSz<int2> _cvgmC2PPairs;// idx is current; int2.x is 1st match, int2.y is the 2nd
	PtrStepSz<float2> _cvgmDistance;
	PtrStepSz<int> _cvgmC2PPairs3NN;// idx is current; cols are 1st 2nd 3rd best matches
	PtrStepSz<float> _cvgmDistance3NN;

	//frame
	PtrStepSz<float3> _points_world_prev;
	PtrStepSz<float> _cvgmGlobalKeypointWorld;
	PtrStepSz<float3> _normal_prev;
	PtrStepSz<float3> _main_dir_prev;

	PtrStepSz<float3> _points_local_curr; 
	PtrStepSz<float3> _normal_local_curr; 
	PtrStepSz<float3> _main_dir_local_curr; 
	PtrStepSz<float> _2d_keypoint_curr; 
	PtrStepSz<float2> _2d_curr;
	PtrStepSz<float3> _bv_curr;
	PtrStepSz<short> _wei_curr;
	PtrStepSz<uchar> _descriptors_curr; 

	//output
	PtrStepSz<float3> _points_world_prev_selected;
	PtrStepSz<float3> _normal_world_prev_selected;
	PtrStepSz<float3> _main_dir_world_prev_selected;
	
	PtrStepSz<int2>   _gpu_pairs_selected; //idx of current to previous pair 
	PtrStepSz<float>  _gpu_hamming_dist_selected; //idx of current to previous pair 
	PtrStepSz<float>  _gpu_ratio_selected;
	

	PtrStepSz<float3> _points_local_curr_selected;
	PtrStepSz<float3> _normal_local_curr_selected;
	PtrStepSz<float3> _main_dir_local_curr_selected;
	PtrStepSz<float>  _2d_keypoint_curr_selected; 
	PtrStepSz<float2> _2d_curr_selected;
	PtrStepSz<float3> _bv_curr_selected;
	PtrStepSz<short>  _wg_curr_selected;

	PtrStepSz<uchar> _descriptors_curr_selected; 

	__device__ __forceinline__ void collect2NN(){

		const int nIdx = threadIdx.x + blockIdx.x * blockDim.x; // the idx of pairs from current frame to prev (global)
		if( nIdx >= _cvgmDistance.cols ) return; //# of pairs == # of distances

		float2 f2Distance = _cvgmDistance.ptr()[nIdx];  if ( f2Distance.x > _fMatchThreshold ) return;
		int2 n2Matches = _cvgmC2PPairs.ptr()[nIdx];

		//the distance of first match is larger than 0.6*distance of second, 
		//which means 1st match and 2nd match are similar, return;
		bool bCollect2nd = false;

		if ( f2Distance.x > _fPercentage * f2Distance.y ){//ambiguous matching
			//printf("id. %d %f %f\n", nIdx, f2Distance.x, f2Distance.y );

			if ( _nMatchingMethod == IDF || _nMatchingMethod == Jose ){ return; } //for IDF and Jose
			else if (_nMatchingMethod == GM || _nMatchingMethod == GM_GPU || _nMatchingMethod == GM_GPU_IM || _nMatchingMethod == BUILD_GT) { bCollect2nd = true; }
		}
		//it will reach here if any one of the following conditions is ture:
		//1. confident matching and collect only 1st (IDF, Jose, GM)
		//2. ambiguous  and collect both 1st and 2nd (GM)
		//pt in 3d
		float3 pt_prv_1, nl_prv_1, md_prv_1; //first match
		float3 pt_prv_2, nl_prv_2, md_prv_2; //second match

		if( n2Matches.x>= _nKeyPoints ||  n2Matches.y >= _nKeyPoints ) {
			//printf("here. 1. %d x%d y%d k%d t%d\n", nIdx, n2Matches.x, n2Matches.y, _nKeyPoints, _cvgmGlobalKeypointWorld.cols );
			return;
		}
	    if( nIdx >= _points_local_curr.cols || nIdx >= _normal_local_curr.cols || nIdx >= _main_dir_local_curr.cols || nIdx >= _2d_curr.cols || nIdx >= _2d_keypoint_curr.cols || nIdx >= _descriptors_curr.rows ){
			//printf("here. 1. %d pc%d ncy%d mc%d 2c%d kc%d dc%d \n", nIdx, _points_local_curr.cols, _normal_local_curr.cols, _main_dir_local_curr.cols,_2d_curr.cols,  _2d_keypoint_curr.cols, _descriptors_curr.rows );
			return;
		}

		if( !_bGlobalModel ){ //pick from previous frame
			//1st
			pt_prv_1 = _points_world_prev.ptr()[n2Matches.x]; if (pt_prv_1.x != pt_prv_1.x || pt_prv_1.y != pt_prv_1.y || pt_prv_1.z != pt_prv_1.z) return;
			nl_prv_1 = _normal_prev.ptr()[ n2Matches.x ];
			md_prv_1 = _main_dir_prev.ptr()[ n2Matches.x ];
			if( bCollect2nd ){
				//2nd
				pt_prv_2 = _points_world_prev.ptr()[ n2Matches.y ];
				nl_prv_2 = _normal_prev.ptr()[ n2Matches.y ];
				md_prv_2 = _main_dir_prev.ptr()[ n2Matches.y ];
			}
		}
		else{//pick from the global feature model
			//1st
			pt_prv_1 = make_float3( _cvgmGlobalKeypointWorld.ptr(0)[ n2Matches.x ],  _cvgmGlobalKeypointWorld.ptr(1)[ n2Matches.x ], _cvgmGlobalKeypointWorld.ptr(2)[ n2Matches.x ] );
			nl_prv_1 = make_float3( _cvgmGlobalKeypointWorld.ptr(3)[ n2Matches.x ],  _cvgmGlobalKeypointWorld.ptr(4)[ n2Matches.x ], _cvgmGlobalKeypointWorld.ptr(5)[ n2Matches.x ] );
			md_prv_1 = make_float3( _cvgmGlobalKeypointWorld.ptr(9)[ n2Matches.x ],  _cvgmGlobalKeypointWorld.ptr(10)[ n2Matches.x ], _cvgmGlobalKeypointWorld.ptr(11)[ n2Matches.x ] );
			if( bCollect2nd ){
				//2nd
				pt_prv_2 = make_float3( _cvgmGlobalKeypointWorld.ptr(0)[ n2Matches.y ],  _cvgmGlobalKeypointWorld.ptr(1)[ n2Matches.y ], _cvgmGlobalKeypointWorld.ptr(2)[ n2Matches.y ] );
				nl_prv_2 = make_float3( _cvgmGlobalKeypointWorld.ptr(3)[ n2Matches.y ],  _cvgmGlobalKeypointWorld.ptr(4)[ n2Matches.y ], _cvgmGlobalKeypointWorld.ptr(5)[ n2Matches.y ] );
				md_prv_2 = make_float3( _cvgmGlobalKeypointWorld.ptr(9)[ n2Matches.y ],  _cvgmGlobalKeypointWorld.ptr(10)[ n2Matches.y ], _cvgmGlobalKeypointWorld.ptr(11)[ n2Matches.y ] );
			}
			//printf("here. 1. %d 1x%f y%f z%f 2x%f y%f z%f nx%f ny%f nz%f 2nx%f ny%f nz%f\n", nIdx, pt_prv_1.x, pt_prv_1.y, pt_prv_1.z,  pt_prv_2.x, pt_prv_2.y, pt_prv_2.z,              
			//	nl_prv_1.x,nl_prv_1.y, nl_prv_1.z, nl_prv_2.x, nl_prv_2.y, nl_prv_2.z  );
		}
		//pt and normal in current frame
		float3 pt_cur = _points_local_curr.ptr()[ nIdx ];
		float3 nl_cur = _normal_local_curr.ptr()[ nIdx ];
		float3 md_cur = _main_dir_local_curr.ptr()[ nIdx ];
		float2 kp_2d_curr = _2d_curr.ptr()[nIdx];
		float3 bv_curr = _bv_curr.ptr()[nIdx];
		short3 weigh = make_short3(_wei_curr.ptr(0)[nIdx], _wei_curr.ptr(1)[nIdx], _wei_curr.ptr(2)[nIdx]);
		
		const int nNew = atomicAdd( &__devnTotal, 1 ); //have to be this way
		const int nCounter = nNew + _nOffset;
		if( nCounter >= _points_world_prev_selected.cols ) return;

		//collect 1st match (IDF, Jose, GM)
		_points_world_prev_selected.ptr()[ nCounter ] = pt_prv_1;
		_normal_world_prev_selected.ptr()[ nCounter ] = nl_prv_1;
		_main_dir_world_prev_selected.ptr()[ nCounter ] = md_prv_1;

		_points_local_curr_selected.ptr()[ nCounter ] = pt_cur;//1.
		_normal_local_curr_selected.ptr()[ nCounter ] = nl_cur;//2.
		_main_dir_local_curr_selected.ptr()[ nCounter ] = md_cur;//2.5
		_2d_curr_selected.ptr()[ nCounter ] = kp_2d_curr;//3.
		_bv_curr_selected.ptr()[nCounter] = bv_curr;//3.5
		_wg_curr_selected.ptr(0)[nCounter] = weigh.x;
		_wg_curr_selected.ptr(1)[nCounter] = weigh.y;
		_wg_curr_selected.ptr(2)[nCounter] = weigh.z;
		//copy key point 4.
		for (int r=0; r< _2d_keypoint_curr.rows; r++) {
			if (r == 6) {
				_2d_keypoint_curr_selected.ptr(r)[nCounter] = (1 - f2Distance.x / f2Distance.y) > 0 ? (1 - f2Distance.x / f2Distance.y) : 0;
				//printf("here. %d matchability %f\n", nIdx, _2d_keypoint_curr_selected.ptr(r)[nCounter]);
			}
			else
				_2d_keypoint_curr_selected.ptr(r)[ nCounter ] = _2d_keypoint_curr.ptr(r)[ nIdx ];
		}
		//copy descriptor 5.
		memcpy( _descriptors_curr_selected.ptr( nCounter ), _descriptors_curr.ptr( nIdx ),  _descriptors_curr_selected.cols*sizeof(uchar) );
		//collect c_2_p match 
		_gpu_pairs_selected .ptr()[ nCounter ] = make_int2( nIdx, n2Matches.x );//n2Matches is the 1st best and 2nd best of previous frame.
		_gpu_hamming_dist_selected.ptr()[ nCounter ] =  f2Distance.x; //HMD of 1st 
		_gpu_ratio_selected.ptr()[nCounter] = (f2Distance.y - f2Distance.x) / f2Distance.y;
														   //Note that only best matches is selected.
		
		if(bCollect2nd){//collect 2nd best (GM)
			const int nNew = atomicAdd( &__devnTotal, 1 ); //have to be this way
			const int nCounter = nNew + _nOffset;
			if( nCounter >= _points_world_prev_selected.cols ) return;
			
			_points_world_prev_selected.ptr()[ nCounter ] = pt_prv_2;
			_normal_world_prev_selected.ptr()[ nCounter ] = nl_prv_2;
			_main_dir_world_prev_selected.ptr()[ nCounter ] = md_prv_2;

			_points_local_curr_selected.ptr()[ nCounter ] = pt_cur;//1.
			_normal_local_curr_selected.ptr()[ nCounter ] = nl_cur;//2.
			_main_dir_local_curr_selected.ptr()[ nCounter ] = md_cur;//2.5
			_2d_curr_selected.ptr()[nCounter] = kp_2d_curr;//3.
			_bv_curr_selected.ptr()[nCounter] = bv_curr;//3.5
			_wg_curr_selected.ptr(0)[nCounter] = weigh.x;
			_wg_curr_selected.ptr(1)[nCounter] = weigh.y;
			_wg_curr_selected.ptr(2)[nCounter] = weigh.z;
			//copy key point 4.
			for (int r=0; r< _2d_keypoint_curr.rows; r++) {
				if (r == 6)
					_2d_keypoint_curr_selected.ptr(r)[nCounter] =  0;
				else
					_2d_keypoint_curr_selected.ptr(r)[nCounter] = _2d_keypoint_curr.ptr(r)[nIdx];
			}
			//copy descriptor 5.
			memcpy( _descriptors_curr_selected.ptr( nCounter ), _descriptors_curr.ptr( nIdx ),  _descriptors_curr_selected.cols*sizeof(uchar) );
			//collect c_2_p match
			_gpu_pairs_selected .ptr()[ nCounter ] = make_int2( nIdx, n2Matches.y ); //2nd best
			_gpu_hamming_dist_selected.ptr()[ nCounter ] =  f2Distance.y; //HMD of 2nd 
			_gpu_ratio_selected.ptr()[nCounter] = 0.f;
			//printf("here. 1. %d c%d np%d  pc%d nc%d 2c%d kpc%d dc%d c2p%d h%d\n", nIdx, nCounter, _normal_world_prev_selected.cols, _points_local_curr_selected.cols, _normal_local_curr_selected.cols, _2d_curr_selected.cols, _2d_keypoint_curr_selected.cols, _descriptors_curr_selected.rows, _gpu_pairs_selected.cols, _gpu_hamming_dist_selected.cols);
		}
		return;
	}

	__device__ __forceinline__ void collect3NN(){

		const int nIdx = threadIdx.x + blockIdx.x * blockDim.x; // the idx of pairs from current frame to prev (global)
		if (nIdx >= _cvgmDistance3NN.rows) return; //# of pairs == # of distances

		float fD1st = _cvgmDistance3NN.ptr(nIdx)[0];  if (fD1st > _fMatchThreshold) return;
		float fD2nd = _cvgmDistance3NN.ptr(nIdx)[1];
		float fD3rd = _cvgmDistance3NN.ptr(nIdx)[2];
		
		int  idx1st = _cvgmC2PPairs3NN.ptr(nIdx)[0];
		int  idx2nd = _cvgmC2PPairs3NN.ptr(nIdx)[1];
		int  idx3rd = _cvgmC2PPairs3NN.ptr(nIdx)[2];
		
		//printf("error. 1. %d 1st %d, 2nd %d, 3rd %d, total %d, t%d\n", nIdx, idx1st, idx2nd, idx3rd, _nKeyPoints, _cvgmGlobalKeypointWorld.cols);

		//the distance of first match is larger than 0.6*distance of second, 
		//which means 1st match and 2nd match are similar, return;
		int bCollect2nd = 1;

		if (fD1st > _fPercentage * fD2nd){//ambiguous matching
			//printf("id. %d %f %f\n", nIdx, f2Distance.x, f2Distance.y );
			if (_nMatchingMethod == IDF || _nMatchingMethod == Jose) { 
				return; 
			} //for IDF and Jose
			else if (_nMatchingMethod == GM || _nMatchingMethod == GM_GPU || _nMatchingMethod == GM_GPU_IM || _nMatchingMethod == BUILD_GT) { 
				bCollect2nd = 2; 
				if (fD3rd - fD2nd < 100.f){
					bCollect2nd = 3;
				}
			}
		}

		//it will reach here if any one of the following conditions is ture:
		//1. confident matching and collect only 1st (IDF, Jose, GM)
		//2. ambiguous  and collect both 1st and 2nd or 3rd (GM)
		//pt in 3d
		float3 pt_prv_1, nl_prv_1, md_prv_1; //first match
		float3 pt_prv_2, nl_prv_2, md_prv_2; //second match
		float3 pt_prv_3, nl_prv_3, md_prv_3; //third match

		if (idx1st >= _nKeyPoints || idx2nd >= _nKeyPoints || idx3rd >= _nKeyPoints) {
			//printf("error. 1. %d 1st %d, 2nd %d, 3rd %d, total %d, t%d\n", nIdx, idx1st, idx2nd, idx3rd, _nKeyPoints, _cvgmGlobalKeypointWorld.cols);
			return;
		}
		if (nIdx >= _points_local_curr.cols || nIdx >= _normal_local_curr.cols || nIdx >= _main_dir_local_curr.cols || nIdx >= _2d_curr.cols || nIdx >= _2d_keypoint_curr.cols || nIdx >= _descriptors_curr.rows){
			//printf("error. 1. %d pc%d ncy%d mc%d 2c%d kc%d dc%d \n", nIdx, _points_local_curr.cols, _normal_local_curr.cols, _main_dir_local_curr.cols, _2d_curr.cols, _2d_keypoint_curr.cols, _descriptors_curr.rows);
			return;
		}

		if (!_bGlobalModel){ //pick from previous frame
			//1st
			pt_prv_1 = _points_world_prev.ptr()[idx1st];
			nl_prv_1 = _normal_prev.ptr()[idx1st];
			md_prv_1 = _main_dir_prev.ptr()[idx1st];
			if (bCollect2nd>=2){
				//2nd
				pt_prv_2 = _points_world_prev.ptr()[idx2nd];
				nl_prv_2 = _normal_prev.ptr()[idx2nd];
				md_prv_2 = _main_dir_prev.ptr()[idx2nd];
				if (bCollect2nd == 3){
					//3rd
					pt_prv_3 = _points_world_prev.ptr()[idx3rd];
					nl_prv_3 = _normal_prev.ptr()[idx3rd];
					md_prv_3 = _main_dir_prev.ptr()[idx3rd];
				}
			}
		}
		else{//pick from the global feature model
			//1st
			pt_prv_1 = make_float3(_cvgmGlobalKeypointWorld.ptr(0)[idx1st], _cvgmGlobalKeypointWorld.ptr(1)[idx1st], _cvgmGlobalKeypointWorld.ptr(2)[idx1st]);
			nl_prv_1 = make_float3(_cvgmGlobalKeypointWorld.ptr(3)[idx1st], _cvgmGlobalKeypointWorld.ptr(4)[idx1st], _cvgmGlobalKeypointWorld.ptr(5)[idx1st]);
			md_prv_1 = make_float3(_cvgmGlobalKeypointWorld.ptr(9)[idx1st], _cvgmGlobalKeypointWorld.ptr(10)[idx1st], _cvgmGlobalKeypointWorld.ptr(11)[idx1st]);
			if (bCollect2nd>=2){
				//2nd
				pt_prv_2 = make_float3(_cvgmGlobalKeypointWorld.ptr(0)[idx2nd], _cvgmGlobalKeypointWorld.ptr(1)[idx2nd], _cvgmGlobalKeypointWorld.ptr(2)[idx2nd]);
				nl_prv_2 = make_float3(_cvgmGlobalKeypointWorld.ptr(3)[idx2nd], _cvgmGlobalKeypointWorld.ptr(4)[idx2nd], _cvgmGlobalKeypointWorld.ptr(5)[idx2nd]);
				md_prv_2 = make_float3(_cvgmGlobalKeypointWorld.ptr(9)[idx2nd], _cvgmGlobalKeypointWorld.ptr(10)[idx2nd], _cvgmGlobalKeypointWorld.ptr(11)[idx2nd]);
				if (bCollect2nd == 3){
					//3rd
					pt_prv_3 = make_float3(_cvgmGlobalKeypointWorld.ptr(0)[idx3rd], _cvgmGlobalKeypointWorld.ptr(1)[idx3rd], _cvgmGlobalKeypointWorld.ptr(2)[idx3rd]);
					nl_prv_3 = make_float3(_cvgmGlobalKeypointWorld.ptr(3)[idx3rd], _cvgmGlobalKeypointWorld.ptr(4)[idx3rd], _cvgmGlobalKeypointWorld.ptr(5)[idx3rd]);
					md_prv_3 = make_float3(_cvgmGlobalKeypointWorld.ptr(9)[idx3rd], _cvgmGlobalKeypointWorld.ptr(10)[idx3rd], _cvgmGlobalKeypointWorld.ptr(11)[idx3rd]);
				}
			}
			//printf("here. 1. %d 1x%f y%f z%f 2x%f y%f z%f nx%f ny%f nz%f 2nx%f ny%f nz%f\n", nIdx, pt_prv_1.x, pt_prv_1.y, pt_prv_1.z,  pt_prv_2.x, pt_prv_2.y, pt_prv_2.z,              
			//	nl_prv_1.x,nl_prv_1.y, nl_prv_1.z, nl_prv_2.x, nl_prv_2.y, nl_prv_2.z  );
		}
		////pt and normal in current frame
		float3 pt_cur = _points_local_curr.ptr()[nIdx];
		float3 nl_cur = _normal_local_curr.ptr()[nIdx];
		float3 md_cur = _main_dir_local_curr.ptr()[nIdx];
		float2 kp_2d_curr = _2d_curr.ptr()[nIdx];

		const int nNew = atomicAdd(&__devnTotal, bCollect2nd); //have to be this way
		int nCounter = nNew + _nOffset;// -bCollect2nd + 1;
		if (nCounter >= _points_world_prev_selected.cols) return;
		//printf("error. 1. %d nNew %d, nCounter %d, _nOffset %d, bCollect2nd %d, __devnTotal %d\n", nIdx, nNew, nCounter, _nOffset, bCollect2nd, __devnTotal);

		//collect 1st match (IDF, Jose, GM)
		_points_world_prev_selected.ptr()[nCounter] = pt_prv_1;
		_normal_world_prev_selected.ptr()[nCounter] = nl_prv_1;
		_main_dir_world_prev_selected.ptr()[nCounter] = md_prv_1;

		_points_local_curr_selected.ptr()[nCounter] = pt_cur;//1.
		_normal_local_curr_selected.ptr()[nCounter] = nl_cur;//2.
		_main_dir_local_curr_selected.ptr()[nCounter] = md_cur;//2.5
		_2d_curr_selected.ptr()[nCounter] = kp_2d_curr;//3.
		//copy key point 4.
		for (int r = 0; r< _2d_keypoint_curr.rows; r++) {
			_2d_keypoint_curr_selected.ptr(r)[nCounter] = _2d_keypoint_curr.ptr(r)[nIdx];
		}
		//copy descriptor 5.
		memcpy(_descriptors_curr_selected.ptr(nCounter), _descriptors_curr.ptr(nIdx), _descriptors_curr_selected.cols*sizeof(uchar));
		//collect c_2_p match 
		_gpu_pairs_selected.ptr()[nCounter] = make_int2(nIdx, idx1st);//n2Matches is the 1st best and 2nd best of previous frame.
		_gpu_hamming_dist_selected.ptr()[nCounter] = fD1st; //HMD of 1st 
		//Note that only best matches is selected.

		if (bCollect2nd>=2){//collect 2nd best (GM)
			const int nCounter = nNew + _nOffset + 1;
			if (nCounter >= _points_world_prev_selected.cols) return;

			_points_world_prev_selected.ptr()[nCounter] = pt_prv_2;
			_normal_world_prev_selected.ptr()[nCounter] = nl_prv_2;
			_main_dir_world_prev_selected.ptr()[nCounter] = md_prv_2;

			_points_local_curr_selected.ptr()[nCounter] = pt_cur;//1.
			_normal_local_curr_selected.ptr()[nCounter] = nl_cur;//2.
			_main_dir_local_curr_selected.ptr()[nCounter] = md_cur;//2.5
			_2d_curr_selected.ptr()[nCounter] = kp_2d_curr;//3.
			//copy key point 4.
			for (int r = 0; r< _2d_keypoint_curr.rows; r++) {
				_2d_keypoint_curr_selected.ptr(r)[nCounter] = _2d_keypoint_curr.ptr(r)[nIdx];
			}
			//copy descriptor 5.
			memcpy(_descriptors_curr_selected.ptr(nCounter), _descriptors_curr.ptr(nIdx), _descriptors_curr_selected.cols*sizeof(uchar));
			//collect c_2_p match
			_gpu_pairs_selected.ptr()[nCounter] = make_int2(nIdx, idx2nd); //2nd best
			_gpu_hamming_dist_selected.ptr()[nCounter] = fD2nd; //HMD of 2nd 
			//printf("here. 1. %d c%d np%d  pc%d nc%d 2c%d kpc%d dc%d c2p%d h%d\n", nIdx, nCounter, _normal_world_prev_selected.cols, _points_local_curr_selected.cols, _normal_local_curr_selected.cols, _2d_curr_selected.cols, _2d_keypoint_curr_selected.cols, _descriptors_curr_selected.rows, _gpu_pairs_selected.cols, _gpu_hamming_dist_selected.cols);

			if (bCollect2nd == 2){//collect 2nd best (GM)
				const int nCounter = nNew + _nOffset + 2;
				if (nCounter >= _points_world_prev_selected.cols) return;

				_points_world_prev_selected.ptr()[nCounter] = pt_prv_3;
				_normal_world_prev_selected.ptr()[nCounter] = nl_prv_3;
				_main_dir_world_prev_selected.ptr()[nCounter] = md_prv_3;

				_points_local_curr_selected.ptr()[nCounter] = pt_cur;//1.
				_normal_local_curr_selected.ptr()[nCounter] = nl_cur;//2.
				_main_dir_local_curr_selected.ptr()[nCounter] = md_cur;//2.5
				_2d_curr_selected.ptr()[nCounter] = kp_2d_curr;//3.
				//copy key point 4.
				for (int r = 0; r < _2d_keypoint_curr.rows; r++) {
					_2d_keypoint_curr_selected.ptr(r)[nCounter] = _2d_keypoint_curr.ptr(r)[nIdx];
					
				}
				//copy descriptor 5.
				memcpy(_descriptors_curr_selected.ptr(nCounter), _descriptors_curr.ptr(nIdx), _descriptors_curr_selected.cols*sizeof(uchar));
				//collect c_2_p match
				_gpu_pairs_selected.ptr()[nCounter] = make_int2(nIdx, idx3rd); //2nd best
				_gpu_hamming_dist_selected.ptr()[nCounter] = fD3rd; //HMD of 2nd 
				//printf("here. 1. %d c%d np%d  pc%d nc%d 2c%d kpc%d dc%d c2p%d h%d\n", nIdx, nCounter, _normal_world_prev_selected.cols, _points_local_curr_selected.cols, _normal_local_curr_selected.cols, _2d_curr_selected.cols, _2d_keypoint_curr_selected.cols, _descriptors_curr_selected.rows, _gpu_pairs_selected.cols, _gpu_hamming_dist_selected.cols);
			}
		}
		return;
	}

};

__global__ void kernelCollect2NN( SDevCollect sDC_ ){
	sDC_.collect2NN();
}

__global__ void kernelCollect3NN(SDevCollect sDC_){
	sDC_.collect3NN();
}

//for 2 NN
//nMatchingMethod_, the function collect different points given different matching methods:
//                  1. for JOSE and IDF, only 1st matche are collected
//					2. for GM, if 1st match is far better than 2nd only 1st collected
//							   if 1st match is similar to 2nd both are collected.
//Input 3d points, normals, keypoint and descriptors are pre-selected to remove those key points without 3D points 
//All ***_selected_ are the same length equal to # of selected pairs
//return the # of the pairs

int cuda_collect_point_pairs_2nn( const float fMatchThreshold_, const float& fPercentage_, int nMatchingMethod_, int nOffset_,int nKeyPoints_,
							  const GpuMat& cvgmC2PPairs_, const GpuMat& cvgmDistance_,
							  const GpuMat& cvgmPtsWorldPrev_, const GpuMat& cvgmPtsLocalCurr_, 
							  const GpuMat& gpu_2d_curr_, const GpuMat& gpu_bv_curr_, const GpuMat& gpu_weight_, const GpuMat& gpu_key_points_curr_, const GpuMat& gpu_descriptor_curr_,
							  const GpuMat& Nls_Prev_, const GpuMat& MDs_Prev_,  const GpuMat& Nls_LocalCurr_, const GpuMat& MDs_LocalCurr_, 
						      GpuMat* ptr_pts_training_selected_, GpuMat* ptr_pts_curr_selected_,
							  GpuMat* ptr_nls_training_selected_,GpuMat* ptr_nls_curr_selected_,
							  GpuMat* ptr_mds_training_selected_,GpuMat* ptr_mds_curr_selected_,
							  GpuMat* ptr_2d_curr_selected_, GpuMat* ptr_bv_curr_selected_, GpuMat* ptr_weight_selected_, GpuMat* ptr_key_point_selected_, GpuMat* ptr_descriptor_curr_selected_,
							  GpuMat* ptr_idx_curr_2_prev_selected_, GpuMat* ptr_hamming_dist_selected_, GpuMat* ptr_ratio_selected_){
	SDevCollect sDC;

	sDC._fMatchThreshold = fMatchThreshold_;
	sDC._fPercentage = fPercentage_;
	sDC._nOffset = nOffset_;
	sDC._nKeyPoints = nKeyPoints_;
	sDC._cvgmC2PPairs = cvgmC2PPairs_;
	sDC._cvgmDistance = cvgmDistance_;
//input
	//current normal + point*
	sDC._normal_local_curr = Nls_LocalCurr_;
	sDC._main_dir_local_curr = MDs_LocalCurr_;
	sDC._points_local_curr = cvgmPtsLocalCurr_; 
	sDC._2d_keypoint_curr = gpu_key_points_curr_;
	sDC._2d_curr = gpu_2d_curr_;
	sDC._bv_curr = gpu_bv_curr_;
	sDC._wei_curr = gpu_weight_;
	sDC._descriptors_curr = gpu_descriptor_curr_;

	sDC._nMatchingMethod = nMatchingMethod_;
	//previous normal + point
	if( cvgmPtsWorldPrev_.rows == 1 ){
		sDC._points_world_prev = cvgmPtsWorldPrev_;//point
		if( Nls_Prev_.empty() || MDs_Prev_.empty() )
		{
			std::cout << " Failure - Nls_Prev_ & MDs_Prev_ cannot be empty. " << std::endl;
			return -1;
		}
		sDC._normal_prev = Nls_Prev_;//nl
		sDC._main_dir_prev = MDs_Prev_;//md
		sDC._bGlobalModel = false;
	}
	else if( cvgmPtsWorldPrev_.rows == 12 ){
		//for global model, it combines both point and normal. 
		sDC._cvgmGlobalKeypointWorld = cvgmPtsWorldPrev_;//point + nl 
		sDC._bGlobalModel = true;
	}
	else{
		std::cout << " Failure - Incorrect format of _cvgmGlobalKeypointWorld.rows = " << cvgmPtsWorldPrev_.rows << std::endl;
		return -1;
	}

//output
	//normal
	sDC._normal_world_prev_selected = *ptr_nls_training_selected_;
	sDC._main_dir_world_prev_selected = *ptr_mds_training_selected_;//2.5
	sDC._normal_local_curr_selected = *ptr_nls_curr_selected_;//2.
	sDC._main_dir_local_curr_selected = *ptr_mds_curr_selected_;//2.5
	//point
	sDC._points_world_prev_selected = *ptr_pts_training_selected_;
	sDC._points_local_curr_selected = *ptr_pts_curr_selected_;//1.
	//2d
	sDC._2d_keypoint_curr_selected = *ptr_key_point_selected_; //3.
	sDC._2d_curr_selected = *ptr_2d_curr_selected_;//4.
	sDC._bv_curr_selected = *ptr_bv_curr_selected_;//4.5
	sDC._wg_curr_selected = *ptr_weight_selected_;//4.6
	sDC._descriptors_curr_selected = *ptr_descriptor_curr_selected_;//5.

	//idx pairs
	sDC._gpu_pairs_selected  = *ptr_idx_curr_2_prev_selected_;
	sDC._gpu_hamming_dist_selected = *ptr_hamming_dist_selected_;
	sDC._gpu_ratio_selected = *ptr_ratio_selected_;

	void* pTotal;
	cudaSafeCall( cudaGetSymbolAddress(&pTotal, __devnTotal) );
	cudaSafeCall( cudaMemset(pTotal, 0, sizeof(int)) );

	dim3 block(128,1,1);
    dim3 grid(1,1,1);
    grid.x = cv::cudev::divUp( cvgmC2PPairs_.cols, block.x );
	
	kernelCollect2NN<<<grid, block>>>( sDC );
	cudaSafeCall ( cudaGetLastError () );
	//cudaSafeCall ( cudaDeviceSynchronize () );

	int nTotal;
	cudaSafeCall( cudaMemcpy(&nTotal, pTotal, sizeof(int), cudaMemcpyDeviceToHost ) );

	nTotal += nOffset_; 
	nTotal = min( nTotal, ptr_pts_training_selected_->cols );
	return nTotal;
}

__constant__ short __n_features;
__constant__ float __f_percentage;
__global__ void kernel_collect_distinct(PtrStepSz<short> idx_, PtrStepSz<int2> cvgmC2PPairs_, PtrStepSz<float2> cvgmDistance_, PtrStepSz<short2> idx_curr_2_prev_selected_)
{
	const short nC = blockDim.x * blockIdx.x + threadIdx.x;
	if (nC >= __n_features) return;

	short idx = idx_.ptr()[nC];
	if (cvgmDistance_.ptr()[idx].x / cvgmDistance_.ptr()[idx].y >= __f_percentage) return;

	const int nNew = atomicAdd(&__devnTotal, 1); //have to be this way
	idx_curr_2_prev_selected_.ptr()[nNew].x = idx;
	idx_curr_2_prev_selected_.ptr()[nNew].y = cvgmC2PPairs_.ptr()[idx].x;

	return;
}

int cuda_collect_distinctive(const float& fPercentage_, const GpuMat& idx_, short n_fetaures_, const GpuMat& cvgmC2PPairs_, const GpuMat& cvgmDistance_,
							 GpuMat* ptr_idx_curr_2_prev_selected_)
{
	if (n_fetaures_ <= 0) return 0;
	cudaSafeCall(cudaMemcpyToSymbol(__n_features,   &n_fetaures_, sizeof(short))); //copy host memory to constant memory on the device.
	cudaSafeCall(cudaMemcpyToSymbol(__f_percentage, &fPercentage_, sizeof(float))); //copy host memory to constant memory on the device.

	void* pTotal;
	cudaSafeCall(cudaGetSymbolAddress(&pTotal, __devnTotal));
	cudaSafeCall(cudaMemset(pTotal, 0, sizeof(int)));

	dim3 block(256, 1, 1);
	dim3 grid(1, 1, 1);

	grid.x = cv::cuda::device::divUp(n_fetaures_, block.x);
	kernel_collect_distinct <<<grid, block >>>( idx_, cvgmC2PPairs_, cvgmDistance_, *ptr_idx_curr_2_prev_selected_);
	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaGetLastError());

	int nTotal;
	cudaSafeCall(cudaMemcpy(&nTotal, pTotal, sizeof(int), cudaMemcpyDeviceToHost));

	return nTotal;
}


__global__ void vote_pose(int r, int c, int nTotal, const matrix3f& Rw, const float3& tw, PtrStepSz<matrix3f> _rotation, PtrStepSz<float3> _translation, PtrStepSz<int> _votes,
	PtrStepSz<float3> _points_world_prev, PtrStepSz<float3> _normal_world_prev, PtrStepSz<float3> _points_local_curr, PtrStepSz<float3> _normal_local_curr	){
	//traverse each element of the matrix
	const int nC = blockDim.x * blockIdx.x + threadIdx.x;

	if (nC >= nTotal) return;

	float3 anchor_global = _points_world_prev.ptr()[nC];
	float3 anchor_normal_global = _normal_world_prev.ptr()[nC];

	float3 anchor_c = _points_local_curr.ptr()[nC];
	float3 anchor_normal_c = _normal_local_curr.ptr()[nC];

	float d = norm<float, float3>(Rw*anchor_global + tw - anchor_c);
	float d2 = dot3<float, float3>(Rw*anchor_normal_global, anchor_normal_c);
	if (d < 0.1f && d2 > 0.7){
		atomicAdd(_votes.ptr(r) + c, 1);
	}
	return;
}

__constant__ uchar _popCountTable[] =
{
	0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
	1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
	1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
	3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
}; //will be referenced by SDevCalcAM::accumulate_concensus() & dev_calc_hamming::calc_common()


struct SDevCalcAM{
	//prev
	PtrStepSz<float3> _points_world_prev;
	PtrStepSz<float3> _normal_world_prev;
	//curr
	PtrStepSz<float3> _points_local_curr; 
	PtrStepSz<float3> _normal_local_curr; 
	PtrStepSz<float2> _2d_curr; 

	int _nTotal;
	int _nBytes;

	//output
	PtrStepSz<float> _credability;
	PtrStepSz<float4> _geometry;

	PtrStepSz<float3> _rotation;
	//PtrStepSz<matrix3f> _rotation;
	PtrStepSz<float3> _translation;

	PtrStepSz<ushort> _votes;
	PtrStepSz<uchar> _vote_bit_flags;

	__device__ __forceinline__ void pairwise_am(){
		//traverse each element of the matrix
		const int nC = blockDim.x * blockIdx.x + threadIdx.x;
		const int nR = blockDim.y * blockIdx.y + threadIdx.y;

		if (nC >= _nTotal || nR >= _nTotal ) return;
		//if (nC <= nR ) return; // because only the upper triangle matrix need to be calculated
		
		//get anchor in current frame
		float3 anchor_c = _points_local_curr.ptr()[ nC ];
		float3 anchor_normal_c = _normal_local_curr.ptr()[ nC ];
		float2 anchor_2d_c = _2d_curr.ptr()[ nC ];
		
		//get second in current frame
		float3 second_c = _points_local_curr.ptr()[ nR ];
		float3 second_normal_c = _normal_local_curr.ptr()[ nR ];

		float3 d_c = second_c - anchor_c;
		float distance_c = norm<float,float3>( d_c );

		//get anchor in global model
		float3 anchor_global = _points_world_prev.ptr()[ nC ];
		float3 anchor_normal_global = _normal_world_prev.ptr()[ nC ];
		//get 2nd in global 
		float3 second_global = _points_world_prev.ptr()[ nR ];
		float3 second_normal_global = _normal_world_prev.ptr()[ nR ];

		float3 d_g = second_global - anchor_global;
		float distance_g = norm<float,float3>( d_g );

		if( fabs( distance_g ) < 0.005f || fabs( distance_c ) < 0.005f ) {
			_credability.ptr(nR)[nC] = _credability.ptr(nC)[nR] = 0.f; //impossible to a correct pair
			return;
		}
		
		//calc angle feature in current frame
		d_c = d_c/distance_c;
		float dp0_c = pcl::device::dot3<float, float3>( second_normal_c, anchor_normal_c ); dp0_c = dp0_c>1.f? 1.f : dp0_c; dp0_c = dp0_c<-1.f? -1.f : dp0_c;
		float a0_c =  acos( dp0_c );
		float dp1_c = pcl::device::dot3<float, float3>(d_c, anchor_normal_c); dp1_c = dp1_c>1.f ? 1.f : dp1_c; dp1_c = dp1_c<-1.f ? -1.f : dp1_c;
		float a1_c =  acos( dp1_c );
		float dp2_c = -pcl::device::dot3<float, float3>(d_c, second_normal_c); dp2_c = dp2_c>1.f ? 1.f : dp2_c; dp2_c = dp2_c<-1.f ? -1.f : dp2_c;
		float a2_c =  acos( dp2_c );

		//calc angle feature in global
		d_g = d_g/distance_g;
		float dp0 = pcl::device::dot3<float, float3>(second_normal_global, anchor_normal_global);		dp0 = dp0 > 1.f ? 1.f : dp0; dp0 = dp0 < -1.f ? -1.f : dp0;
		float a0_g =  acos( dp0 );
		float dp1 = pcl::device::dot3<float, float3>(d_g, anchor_normal_global); 	dp1 = dp1 > 1.f ? 1.f : dp1; dp1 = dp1 < -1.f ? -1.f : dp1;
		float a1_g =  acos( dp1 );
		float dp2 = -pcl::device::dot3<float, float3>(d_g, second_normal_global);  dp2 = dp2 > 1.f ? 1.f : dp2; dp2 = dp2 < 1.f ? -1.f : dp2;
		float a2_g =  acos( dp2 );

		float fDist = fabs( distance_c - distance_g ) ;
		float fAD0 = fabs( a0_c - a0_g ); //normal angle difference
		float fAD1 = fabs( a1_c - a1_g ); 
		float fAD2 = fabs( a2_c - a2_g ); 

		//get 2d img coordinate of second point in current frame
		//float2 second_2d_c = curr_2d_.ptr<float2>()[ P_2 ];
		//float dx_2d = second_2d_c.x - anchor_2d_c.x;
		//float dy_2d = second_2d_c.y - anchor_2d_c.y;
		_credability.ptr(nR)[nC] = __expf( - fDist - fAD0 - fAD1 - fAD2   ); //);//

		return;
	}

	
	__device__ __forceinline__ void pairwise_am_binary(){
		//traverse each element of the matrix
		const int nC = blockDim.x * blockIdx.x + threadIdx.x;
		const int nR = blockDim.y * blockIdx.y + threadIdx.y;

		if (nC >= _nTotal || nR >= _nTotal) return;
		//if (nC <= nR ) return; // because only the upper triangle matrix need to be calculated

		//get anchor in current frame
		float3 anchor_c = _points_local_curr.ptr()[nC];
		float3 anchor_normal_c = _normal_local_curr.ptr()[nC];
		float2 anchor_2d_c = _2d_curr.ptr()[nC];

		//get second in current frame
		float3 second_c = _points_local_curr.ptr()[nR];
		float3 second_normal_c = _normal_local_curr.ptr()[nR];

		float3 d_c = second_c - anchor_c;
		float distance_c = norm<float, float3>(d_c);

		//get anchor in global model
		float3 anchor_global = _points_world_prev.ptr()[nC];
		float3 anchor_normal_global = _normal_world_prev.ptr()[nC];
		//get 2nd in global 
		float3 second_global = _points_world_prev.ptr()[nR];
		float3 second_normal_global = _normal_world_prev.ptr()[nR];

		float3 d_g = second_global - anchor_global;
		float distance_g = norm<float, float3>(d_g);

		if (fabs(distance_g) < 0.01f || fabs(distance_c) < 0.01f) {
			_credability.ptr(nR)[nC] = 0.f; //impossible to a correct pair
			_geometry.ptr(nR)[nC] = make_float4(1000.f, 1000.f, 1000.f, 1000.f);
			return;
		}

		//calc angle feature in current frame
		d_c = d_c / distance_c;
		float dp0_c = pcl::device::dot3<float, float3>(second_normal_c, anchor_normal_c); dp0_c = dp0_c>1.f ? 1.f : dp0_c; dp0_c = dp0_c<-1.f ? -1.f : dp0_c;
		float a0_c = acos(dp0_c);
		float dp1_c = pcl::device::dot3<float, float3>(d_c, anchor_normal_c); dp1_c = dp1_c>1.f ? 1.f : dp1_c; dp1_c = dp1_c<-1.f ? -1.f : dp1_c;
		float a1_c = acos(dp1_c);
		float dp2_c = -pcl::device::dot3<float, float3>(d_c, second_normal_c); dp2_c = dp2_c>1.f ? 1.f : dp2_c; dp2_c = dp2_c<-1.f ? -1.f : dp2_c;
		float a2_c = acos(dp2_c);

		//calc angle feature in global
		d_g = d_g / distance_g;
		float dp0 = pcl::device::dot3<float, float3>(second_normal_global, anchor_normal_global);		dp0 = dp0 > 1.f ? 1.f : dp0; dp0 = dp0 < -1.f ? -1.f : dp0;
		float a0_g = acos(dp0);
		float dp1 = pcl::device::dot3<float, float3>(d_g, anchor_normal_global); 	dp1 = dp1 > 1.f ? 1.f : dp1; dp1 = dp1 < -1.f ? -1.f : dp1;
		float a1_g = acos(dp1);
		float dp2 = -pcl::device::dot3<float, float3>(d_g, second_normal_global);  dp2 = dp2 > 1.f ? 1.f : dp2; dp2 = dp2 < 1.f ? -1.f : dp2;
		float a2_g = acos(dp2);

		float fDist  = fabs(distance_c - distance_g);
		float fAD0 = fabs(a0_c - a0_g); //normal difference
		float fAD1 = fabs(a1_c - a1_g); 
		float fAD2 = fabs(a2_c - a2_g);

		_geometry.ptr(nR)[nC] = make_float4(fDist,fAD0,fAD1,fAD2);

		//get 2d img coordinate of second point in current frame
		//float2 second_2d_c = curr_2d_.ptr<float2>()[ P_2 ];
		//float dx_2d = second_2d_c.x - anchor_2d_c.x;
		//float dy_2d = second_2d_c.y - anchor_2d_c.y;

		//if (fDist < 0.05 && fAD0 < 0.088f && fAD1 < 0.088f && fAD2 < 0.088f )
		if (fDist < 0.05 &&  fAD0 < 1.5f && fAD1 + fAD2 < 3.f)
			_credability.ptr(nR)[nC] = 1.f; 
		else
			_credability.ptr(nR)[nC] = 0.f; 

		return;
	}

	__device__ __forceinline__ void calc_poses(){
		//traverse each element of the matrix
		const int nC = blockDim.x * blockIdx.x + threadIdx.x;
		const int nR = blockDim.y * blockIdx.y + threadIdx.y;

		if (nC >= _nTotal || nR >= _nTotal || nC == nR) return;
		//if (nC <= nR ) return; // because only the upper triangle matrix need to be calculated

		//get anchor in current frame
		float3 anchor_c = _points_local_curr.ptr()[nC];
		float3 anchor_normal_c = _normal_local_curr.ptr()[nC];
		float3 second_c = _points_local_curr.ptr()[nR];
		
		float3 d_c = second_c - anchor_c;
		float distance_c = norm<float, float3>(d_c);

		//get anchor in global model
		float3 anchor_global = _points_world_prev.ptr()[nC];
		float3 anchor_normal_global = _normal_world_prev.ptr()[nC];
		float3 second_global = _points_world_prev.ptr()[nR];

		float3 d_g = second_global - anchor_global;
		float distance_g = norm<float, float3>(d_g);


		matrix3f R_w; //impossible to be a correct pair
		//matrix3f R_w = _rotation.ptr(nR)[nC];
		float3& t_w = _translation.ptr(nR)[nC];

		if (fabs(distance_g) < 0.01f || fabs(distance_c) < 0.01f) {
			R_w = make_zero_33< float, float3, matrix3f>();
			float3* ptr = _rotation.ptr(nR) + nC*3;
			*ptr++ = make_float3(0.f, 0.f, 0.f);
			*ptr++ = make_float3(0.f, 0.f, 0.f);
			*ptr   = make_float3(0.f, 0.f, 0.f);
			return;
		}

		pose_estimation<float, float3, float4, matrix3f>(anchor_c, anchor_normal_c, second_c, anchor_global, anchor_normal_global, second_global, &R_w, &t_w);

		float3* ptr = _rotation.ptr(nR) + nC*3;
		*ptr++ = R_w.r[0];
		*ptr++ = R_w.r[1];
		*ptr   = R_w.r[2];

		//if (nC < 100 && nR < 100){
			//printf("nC %d nR %d\n", nC, nR);
			//print_vector<float3>(t_w);
			//print_matrix<matrix3f>(R_w);
		//}


		//vote_pose << <gridDim.x, blockDim.x >> >(nR, nC, _nTotal, R_w, t_w, _rotation, _translation, _votes, 
		//	_points_world_prev, _normal_world_prev, _points_local_curr, _normal_local_curr)；


		return;
	}

	__device__ __forceinline__ void set_voters(){
		//traverse each element of the matrix
		const int nC = blockDim.x * blockIdx.x + threadIdx.x;
		const int nR = blockDim.y * blockIdx.y + threadIdx.y;

		if (nC >= _nTotal || nR >= _nTotal || nC==nR ) return;

		matrix3f Rw;
		{
			float3* ptr = _rotation.ptr(nR) + nC * 3;
			Rw.r[0] = *ptr++;
			Rw.r[1] = *ptr++;
			Rw.r[2] = *ptr;
		}
		const float3& tw = _translation.ptr(nR)[nC];

		if (sum<float, matrix3f>(Rw) < 0.1f) return;

		//if (nC < 100 && nR < 100){
		//	printf("nC %d nR %d\n", nC, nR);
		//	//print_vector<float3>(tw);
		//	print_matrix<matrix3f>(Rw);
		//}

		//traverse each element of the matrix
		uchar flag = uchar(0);
		uchar* ptr = _vote_bit_flags.ptr(nR) + nC*_nBytes;

		int nTotalBits = _nBytes * 8;
		for (int i = 0; i < nTotalBits; i++){
			if (i < _nTotal){
				float3 anchor_global = _points_world_prev.ptr()[i];
				float3 anchor_normal_global = _normal_world_prev.ptr()[i];

				float3 anchor_c = _points_local_curr.ptr()[i];
				float3 anchor_normal_c = _normal_local_curr.ptr()[i];

				float d = norm<float, float3>(Rw*anchor_global + tw - anchor_c);
				float d2 = dot3<float, float3>(Rw*anchor_normal_global, anchor_normal_c);
				if (d < 0.1f && d2 > 0.7){
					flag |= uchar(1);
					//atomicAdd(_votes.ptr(nR) + nC, 1);
				}
			}
			
			if (i % 8 == 7){
				*ptr++ = flag; //dereference then increase
			}
			flag <<= 1;
		}
		
		//printf("nC %d nR %d votes %d\n", nC, nR, _votes.ptr(nR)[nC]);
			
		return;
	}

	__device__ __forceinline__ void accumulate_concensus(){
		//traverse each element of the matrix
		const int nC = blockDim.x * blockIdx.x + threadIdx.x;
		const int nR = blockDim.y * blockIdx.y + threadIdx.y;

		if (nC >= _nTotal || nR >= _nTotal ) return;
		
		ushort& vs = _votes.ptr(nR)[nC];
		vs = 0;

		if (nC == nR) {  return; }

		uchar* ptr_vote = _vote_bit_flags.ptr(nR) + nC*_nBytes;
		for (int i = 0; i < _nBytes; i++){
			vs += _popCountTable[ *ptr_vote++ ];
		}
	}
};


__global__ void kernelCalcAM( SDevCalcAM sDCAM_ ){
	sDCAM_.pairwise_am();
}

__global__ void kernelCalcAMBinary(SDevCalcAM sDCAM_){
	sDCAM_.pairwise_am_binary();
}

__global__ void kernelCalcPoses(SDevCalcAM sDCAM_){
	sDCAM_.calc_poses();
}

__global__ void kernel_set_voter_flags(SDevCalcAM sDCAM_){
	sDCAM_.set_voters();
}

__global__ void kernel_accumulate_concensus(SDevCalcAM sDCAM_){
	sDCAM_.accumulate_concensus();
}

void cuda_calc_adjacency_mt( const GpuMat& global_pts_, const GpuMat& curr_pts_, 
							  const GpuMat& global_nls_, const GpuMat& curr_nls_, const GpuMat& curr_2d_, 
							  GpuMat* ptr_credibility_ )
{
	int cols = global_pts_.cols;
	assert( cols == curr_pts_.cols && cols == curr_nls_.cols && 
		cols == global_pts_.cols && cols == global_nls_.cols &&
		cols == curr_2d_.cols && cols == ptr_credibility_->cols && cols == ptr_credibility_->rows );

	SDevCalcAM sDCAM;

	sDCAM._nTotal = cols;
//input
	sDCAM._points_world_prev = global_pts_;
	sDCAM._normal_world_prev = global_nls_;
	sDCAM._points_local_curr = curr_pts_;
	sDCAM._normal_local_curr = curr_nls_;
	sDCAM._2d_curr = curr_2d_;

//output
	sDCAM._credability = *ptr_credibility_;
	
	dim3 block(32,32,1);
    dim3 grid(1,1,1);
    grid.x = grid.y = cv::cudev::divUp( cols, block.x );
	
	kernelCalcAM<<<grid, block>>>( sDCAM );
	//cudaSafeCall ( cudaGetLastError () );
	//cudaSafeCall ( cudaDeviceSynchronize () );
	return;
}

void cuda_calc_adjacency_mt_binary ( const GpuMat& global_pts_, const GpuMat& curr_pts_, 
									 const GpuMat& global_nls_, const GpuMat& curr_nls_, const GpuMat& curr_2d_, 
									 GpuMat* ptr_credibility_, GpuMat* ptr_geometry_)
{
	int cols = global_pts_.cols;
	assert( cols == curr_pts_.cols && cols == curr_nls_.cols && 
			cols == global_pts_.cols && cols == global_nls_.cols &&
			cols == curr_2d_.cols && cols == ptr_credibility_->cols && cols == ptr_credibility_->rows && 
			cols == ptr_geometry_->cols && cols == ptr_geometry_->rows);

	SDevCalcAM sDCAM;

	sDCAM._nTotal = cols;
//input
	sDCAM._points_world_prev = global_pts_;
	sDCAM._normal_world_prev = global_nls_;
	sDCAM._points_local_curr = curr_pts_;
	sDCAM._normal_local_curr = curr_nls_;
	sDCAM._2d_curr = curr_2d_;

//output
	sDCAM._credability = *ptr_credibility_;
	sDCAM._geometry = *ptr_geometry_;
	
	dim3 block(32,32,1);
    dim3 grid(1,1,1);
    grid.x = grid.y = cv::cudev::divUp( cols, block.x );
	
	kernelCalcAMBinary<<<grid, block>>>( sDCAM );
	//cudaSafeCall ( cudaGetLastError () );
	//cudaSafeCall ( cudaDeviceSynchronize () );
	return;
}

void cuda_collect_all(const GpuMat& global_pts_, const GpuMat& curr_pts_,
	const GpuMat& global_nls_, const GpuMat& curr_nls_, const GpuMat& curr_2d_,
	GpuMat* rotations_, GpuMat* translation_, GpuMat* votes_, GpuMat* ptr_vote_bit_flags_)
{

	int cols = global_pts_.cols;
	assert( cols == curr_pts_.cols && cols == curr_nls_.cols &&
			cols == global_pts_.cols && cols == global_nls_.cols &&
			cols == curr_2d_.cols && cols == rotations_->rows && 
			cols == translation_->rows && cols == ptr_vote_bit_flags_->rows );

	SDevCalcAM sDCAM;

	sDCAM._nTotal = cols;
	sDCAM._nBytes = ptr_vote_bit_flags_->cols / cols;
	//input
	sDCAM._points_world_prev = global_pts_;
	sDCAM._normal_world_prev = global_nls_;
	sDCAM._points_local_curr = curr_pts_;
	sDCAM._normal_local_curr = curr_nls_;
	sDCAM._2d_curr = curr_2d_;

	//output
	votes_->setTo(0);
	ptr_vote_bit_flags_->setTo(0);

	sDCAM._rotation = *rotations_;
	sDCAM._translation = *translation_;
	sDCAM._votes = *votes_;
	sDCAM._vote_bit_flags = *ptr_vote_bit_flags_;
	
	dim3 block(16, 16, 1);
	dim3 grid(1, 1, 1);
	grid.x = grid.y = cv::cudev::divUp(cols, block.x);

	kernelCalcPoses <<<grid, block >>>(sDCAM);
	//cudaSafeCall(cudaDeviceSynchronize());
	//cudaSafeCall(cudaGetLastError());

	/*size_t sz;
	cudaDeviceGetLimit(&sz, cudaLimitPrintfFifoSize);
	std::cout << sz << std::endl;
	sz = 1048576 * 1000;
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, sz);*/

	kernel_set_voter_flags <<<grid, block >>>(sDCAM);
	//cudaSafeCall(cudaDeviceSynchronize());
	//cudaSafeCall(cudaGetLastError());

	/*sz = 1048576 ;
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, sz);*/

	kernel_accumulate_concensus<< <grid, block >> >(sDCAM);
	//cudaSafeCall(cudaDeviceSynchronize());
	//cudaSafeCall(cudaGetLastError());

	return;
}

struct dev_calc_hamming{
	PtrStepSz<ushort> _common;
	PtrStepSz<ushort> _votes;
	PtrStepSz<uchar> _vote_bit_flags;
	PtrStepSz<uchar> _best;
	int _nBytes;
	int _nTotal;

	__device__ __forceinline__ void calc_common(){
		//traverse each element of the matrix
		const int nC = blockDim.x * blockIdx.x + threadIdx.x;
		const int nR = blockDim.y * blockIdx.y + threadIdx.y;

		if (nC >= _nTotal || nR >= _nTotal ) return;

		//ushort& nDist   = _common.ptr(nR)[nC].x;
		ushort& nCommon = _common.ptr(nR)[nC];
		if (nC == nR) {
			nCommon =  0; //nDist = nBytes * 8;
			return;
		}

		uchar* ptr_vote = _vote_bit_flags.ptr(nR) + nC*_nBytes;
		uchar* ptr_best = _best.ptr();
		nCommon = 0;
		for (int i = 0; i < _nBytes; i++){
			nCommon += _popCountTable[(*ptr_best++) & (*ptr_vote++)];
			//nDist += _popCountTable[(*ptr_best++) ^ (*ptr_vote++)];
		}

		//nCommon = ushort(float(nCommon) / float(_votes.ptr(nR)[nC]) * 100.f + .5f);

		return;
	}
};
__global__ void kernel_calc_common(dev_calc_hamming obj_calc_hamming_){
	obj_calc_hamming_.calc_common();
}

void cuda_calc_common_ratio(const int r, const int c, const GpuMat& votes_, const GpuMat& ptr_vote_bit_flags_, GpuMat* common_){
	common_->setTo(0);

	dev_calc_hamming obj_calc_hamming;
	const int total = ptr_vote_bit_flags_.rows;
	const int nBytes = ptr_vote_bit_flags_.cols / total;

	obj_calc_hamming._nBytes = nBytes;
	obj_calc_hamming._nTotal = total;
	obj_calc_hamming._votes = votes_;
	obj_calc_hamming._vote_bit_flags = ptr_vote_bit_flags_;
	obj_calc_hamming._common = *common_;
	GpuMat best(ptr_vote_bit_flags_, cv::Rect( c*nBytes, r, nBytes, 1 ));
	assert(best.cols == nBytes && best.rows == 1);
	obj_calc_hamming._best = best;

	dim3 block(32, 32, 1);
	dim3 grid(1, 1, 1);
	grid.x = grid.y = cv::cudev::divUp(ptr_vote_bit_flags_.rows, block.x);

	kernel_calc_common<< <grid, block >> >( obj_calc_hamming );
	//cudaSafeCall(cudaDeviceSynchronize());
	//cudaSafeCall(cudaGetLastError());

	//flatten 

	return;
}

struct SDevExtractMultiscaleFeatures{
	pcl::device::Intr _intr;
	float _fVoxelSize;

	int _nFeatureScale;

	PtrStepSz<float3> _points;
	PtrStepSz<float3> _normals;
	PtrStepSz<float> _depth;
	PtrStepSz<float> _reliability;
	PtrStepSz<float> _key_points; //sequential keypoints
	PtrStepSz<uchar> _descriptors;
	int _n_detected_features;

	float _sigma_min;

	//multiscale features
	PtrStepSz<float3> _points_m[5];
	PtrStepSz<float3> _normals_m[5];
	PtrStepSz<float3> _main_dir_m[5];
	PtrStepSz<float2> _2d_m[5];
	PtrStepSz<float3> _bv_m[5];
	PtrStepSz<short> _wg_m[5];
	PtrStepSz<float> _key_points_m[5]; 
	PtrStepSz<uchar> _descriptors_m[5];

	__device__ __forceinline__ int scaleFromS(float s){
		int lvl = 0;
		if (_nFeatureScale == 1){
			if (s > 0.1f ) lvl = 0;
		}
				
		return lvl;
	}

	__device__ __forceinline__ int2 scaleFromS2(float s){
		int2 lvl;  lvl.x = lvl.y = -1;
		if (_nFeatureScale == 2){
			if (s > 0.1f && s <= 6.f) lvl.x = 0;
			if (s > 1.f) lvl.y = 1;
		}
		return lvl;
	}

	__device__ __forceinline__ int3 scaleFromS3(float s){
		int3 lvl;  lvl.x = lvl.y = lvl.z = -1; 
		if (_nFeatureScale == 3){
			if (s > 0.1f && s <= 1.f) lvl.x = 0;
			if (s > 1.f  && s <= 2.f) lvl.y = 1;
			if (s > 2.f) lvl.z = 2;
		}
		return lvl;
	}

	__device__ __forceinline__ void avgNormal( int x, int y, float3& avg_nl ){
		avg_nl.x = avg_nl.y = avg_nl.z = 0.f;
		for( int i= -3; i <= 3; i++ )
		for( int j= -3; j <= 3; j++ ){
			int c = x+i;
			int r = y+j;
			if ( c <0 || c >= _normals.cols || r <0 || r >= _normals.rows ) continue;
			float3 nl = _normals.ptr(r)[c];
			if (isnan<float>(nl.x) || isnan<float>(nl.y) || isnan<float>(nl.z) ) continue;
			avg_nl = avg_nl + nl;
		}
		return;
	}

	__device__ __forceinline__ void add_a_feature(int lvl, float fSize, float relia, float fRadius, int nIdx, const float3& nl, const float3& pt, const float2& kp_2d, const float3& bv, int* total_){
		int nCounter = accumulate(lvl, total_);
		if (nCounter >= _points_m[lvl].cols) return;

		float fAngle = _key_points.ptr(5)[nIdx] / 180.f*3.141592654f; //angle, left x positive is the 0, clockwise, 0-pi radian
		float2 pt2 = make_float2(_intr.fx*cos(fAngle) + kp_2d.x, _intr.fx*sin(fAngle) + kp_2d.y);

		if (fSize > 0.f){ //with 3d data
			float3 nn = nl;
			float3 v1; v1.x = kp_2d.x - _intr.cx; v1.y = kp_2d.y - _intr.cy; v1.z = _intr.fx; v1 = normalized<float, float3>(v1);
			float3 v2; v2.x = pt2.x - _intr.cx; v2.y = pt2.y - _intr.cy; v2.z = _intr.fx; v2 = normalized<float, float3>(v2);
			float3 md = v1 * dot3<float, float3>(nn, v2) - v2 * dot3<float, float3>(nn, v1); // nn x ( v1 x v2 );
			md = normalized<float, float3>(md);

			//descriptor
			memcpy((void*)_descriptors_m[lvl].ptr(nCounter), (void*)_descriptors.ptr(nIdx), sizeof(uchar)*_descriptors.cols);
			//key points format
			_key_points_m[lvl].ptr(0)[nCounter] = _key_points.ptr(0)[nIdx];
			_key_points_m[lvl].ptr(1)[nCounter] = _key_points.ptr(1)[nIdx];
			_key_points_m[lvl].ptr(2)[nCounter] = fSize; //the octave for 3-D feature size.
			_key_points_m[lvl].ptr(3)[nCounter] = md.x; //
			_key_points_m[lvl].ptr(4)[nCounter] = md.y; //
			_key_points_m[lvl].ptr(5)[nCounter] = md.z; //
			_key_points_m[lvl].ptr(6)[nCounter] = _key_points.ptr(6)[nIdx]; //hessian
			_key_points_m[lvl].ptr(7)[nCounter] = _key_points.ptr(5)[nIdx]; //angle
			_key_points_m[lvl].ptr(8)[nCounter] = fRadius; //2d feature radius
			_key_points_m[lvl].ptr(9)[nCounter] = lvl;

			//points
			_points_m[lvl].ptr()[nCounter] = pt;
			_normals_m[lvl].ptr()[nCounter] = nl;
			_main_dir_m[lvl].ptr()[nCounter] = md;
			_2d_m[lvl].ptr()[nCounter] = kp_2d;
			_bv_m[lvl].ptr()[nCounter] = bv;
			short w1 = _sigma_min / axial_noise_kinect(acos(-nl.z), pt.z) * 32767;
			short w2 = relia * 32767;
			_wg_m[lvl].ptr(0)[nCounter] = 32767;
			_wg_m[lvl].ptr(1)[nCounter] = w1;
			_wg_m[lvl].ptr(2)[nCounter] = w2;
		}
		else { //only 2-D data
			//descriptor
			memcpy((void*)_descriptors_m[lvl].ptr(nCounter), (void*)_descriptors.ptr(nIdx), sizeof(uchar)*_descriptors.cols);
			//key points format
			_key_points_m[lvl].ptr(0)[nCounter] = _key_points.ptr(0)[nIdx];
			_key_points_m[lvl].ptr(1)[nCounter] = _key_points.ptr(1)[nIdx];
			_key_points_m[lvl].ptr(2)[nCounter] = fSize; //the octave for 3-D feature size.
			_key_points_m[lvl].ptr(3)[nCounter] = nl.x; //
			_key_points_m[lvl].ptr(4)[nCounter] = nl.y; //
			_key_points_m[lvl].ptr(5)[nCounter] = nl.z; //
			_key_points_m[lvl].ptr(6)[nCounter] = _key_points.ptr(6)[nIdx]; //hessian
			_key_points_m[lvl].ptr(7)[nCounter] = _key_points.ptr(5)[nIdx]; //angle
			_key_points_m[lvl].ptr(8)[nCounter] = fRadius; //2d feature radius
			_key_points_m[lvl].ptr(9)[nCounter] = lvl;

			//points
			_points_m[lvl].ptr()[nCounter] = pt;
			_normals_m[lvl].ptr()[nCounter] = nl;
			_main_dir_m[lvl].ptr()[nCounter] = nl;
			_2d_m[lvl].ptr()[nCounter] = kp_2d;
			_bv_m[lvl].ptr()[nCounter] = bv;
			_wg_m[lvl].ptr(0)[nCounter] = 32767;
			_wg_m[lvl].ptr(1)[nCounter] = 0;
			_wg_m[lvl].ptr(2)[nCounter] = 0;
		}
	}

	__device__ __forceinline__ void mainFunc(int* total_){
		const int nIdx = threadIdx.x + blockIdx.x * blockDim.x; // the idx of pairs from current frame to prev (global)
		if (nIdx >= _n_detected_features) return; //# of pairs == # of distances
		if( _key_points.ptr(6)[nIdx] < 0.f ) return; //response < 0
		float2 kp_2d = make_float2( _key_points.ptr(0) [ nIdx ], _key_points.ptr(1) [ nIdx ] );
		float3 bv = make_float3( (kp_2d.x - _intr.cx) / _intr.fx, (kp_2d.y - _intr.cy) / _intr.fy, 1.f );
		bv = normalized<float, float3>( bv );
		int nX = __float2int_rd ( kp_2d.x +.5f );
		int nY = __float2int_rd ( kp_2d.y +.5f );

		if ( nX <0 || nX >= _points.cols || nY <0 || nY >= _points.rows ) return;
		bool only_2d = false;
		float3 pt = _points.ptr(nY)[nX];   if (isnan<float>(pt.x) || isnan<float>(pt.y) || isnan<float>(pt.z)) { only_2d = true; }
		float3 nl = _normals.ptr(nY)[nX];  if (isnan<float>(nl.x) || isnan<float>(nl.y) || isnan<float>(nl.z)) { only_2d = true; }
		float dp = _depth.ptr(nY)[nX]; if (isnan<float>(dp)){ only_2d = true; }
		float fRadius = _key_points.ptr(4)[nIdx] / 2.f; //radius
		float fReliability = _reliability.ptr(nY >> 2)[nX >> 2];
		if (!only_2d) {
			float fSize = fRadius * dp / _intr.fx;
			float s = fSize / _fVoxelSize; //octree level
			//printf("kp radius \t %f \t size \t %f\t voxel size \t %f \n", fRadius, fSize, _fVoxelSize);
			if (_nFeatureScale == 1){
				int lvl = scaleFromS(s);
				if (lvl >= 0) add_a_feature(lvl, fSize, fReliability, fRadius, nIdx, nl, pt, kp_2d, bv, total_);
			}
			else if (_nFeatureScale == 2){
				int2 lvl = scaleFromS2(s);
				if (lvl.x >= 0) add_a_feature(lvl.x, fSize, fReliability, fRadius, nIdx, nl, pt, kp_2d, bv, total_);
				if (lvl.y >= 0) add_a_feature(lvl.y, fSize, fReliability, fRadius, nIdx, nl, pt, kp_2d, bv, total_);
			}
			else if (_nFeatureScale == 3){
				int3 lvl = scaleFromS3(s);
				if (lvl.x >= 0) add_a_feature(lvl.x, fSize, fReliability, fRadius, nIdx, nl, pt, kp_2d, bv, total_);
				if (lvl.y >= 0) add_a_feature(lvl.y, fSize, fReliability, fRadius, nIdx, nl, pt, kp_2d, bv, total_);
				if (lvl.z >= 0) add_a_feature(lvl.z, fSize, fReliability, fRadius, nIdx, nl, pt, kp_2d, bv, total_);
			}
		}
		else{
			add_a_feature(0, -1.f, 0.f, fRadius, nIdx, nl, pt, kp_2d, bv, total_);
		}
		
		return;
	}
};

__global__ void kernel_extract_multi( SDevExtractMultiscaleFeatures sdEMF_, int* total_ ){
	sdEMF_.mainFunc( total_ );
}

vector<int> cuda_extract_key_points(  const pcl::device::Intr& intr, const float fVoxelSize_[], const int nFeatureScale_, 
												 const GpuMat& pts_, const GpuMat& nls_, const GpuMat& depth_, const GpuMat& reliability_,
												 const GpuMat& descriptor_, const GpuMat& key_points_, int n_detected_features_,
												 vector<GpuMat>* ptr_v_descriptor_, vector<GpuMat>* ptr_v_key_points_,
												 vector<GpuMat>* ptr_v_pts_, vector<GpuMat>* ptr_v_nls_, vector<GpuMat>* ptr_v_mds_, 
												 vector<GpuMat>* ptr_v_2d_, vector<GpuMat>* ptr_v_bv_, vector<GpuMat>* ptr_v_weigts_)
{
	int _nTotal = key_points_.cols; if (n_detected_features_ == 0) return vector<int>();//>> i 8, 4, 2, is not an very accurate prediction of # of features 

	SDevExtractMultiscaleFeatures sdEMF;
	sdEMF._nFeatureScale = nFeatureScale_;
	sdEMF._n_detected_features = n_detected_features_;
	sdEMF._intr = intr;
	sdEMF._fVoxelSize = fVoxelSize_[0];
	sdEMF._points = pts_;  //2d matrix
	sdEMF._normals = nls_; //2d matrix
	sdEMF._depth = depth_; //2d matrix
	sdEMF._reliability = reliability_;
	sdEMF._key_points = key_points_; 
	sdEMF._descriptors = descriptor_; 
	sdEMF._sigma_min = axial_noise_kinect(.0f, .4f);

	vector<GpuMat> pts_m(nFeatureScale_);
	vector<GpuMat> nls_m(nFeatureScale_);
	vector<GpuMat> mds_m(nFeatureScale_);
	vector<GpuMat> p2d_m(nFeatureScale_);
	vector<GpuMat> pbv_m(nFeatureScale_);
	vector<GpuMat> wei_m(nFeatureScale_);
	vector<GpuMat> des_m(nFeatureScale_);
	vector<GpuMat> kyp_m(nFeatureScale_);
	for (int i = 0; i<nFeatureScale_; i++){
		pts_m[i].create(1,_nTotal,CV_32FC3);
		nls_m[i].create(1,_nTotal,CV_32FC3);
		mds_m[i].create(1,_nTotal,CV_32FC3);
		p2d_m[i].create(1,_nTotal,CV_32FC2);
		pbv_m[i].create(1,_nTotal,CV_32FC3);
		wei_m[i].create(3, _nTotal, CV_16SC1);
		des_m[i].create(descriptor_.size(),CV_8UC1);
		kyp_m[i].create(key_points_.size(),CV_32FC1);

		sdEMF._points_m[i] = pts_m[i]; 
		sdEMF._normals_m[i] = nls_m[i]; 
		sdEMF._main_dir_m[i] = mds_m[i];
		sdEMF._2d_m[i] = p2d_m[i];
		sdEMF._bv_m[i] = pbv_m[i];
		sdEMF._wg_m[i] = wei_m[i];
		sdEMF._key_points_m[i] = kyp_m[i]; 
		sdEMF._descriptors_m[i] = des_m[i]; 
	}

	GpuMat total; total.create(1, nFeatureScale_, CV_32SC1); total.setTo(0);
	//int* total;   cudaSafeCall( cudaMallocManaged(&total, nFeatureScale_ * sizeof(int)) );
	//for (int i = 0; i < nFeatureScale_; i++){
	//	total[i] = 0;
	//}

	//extract
	dim3 block(256,1,1);
    dim3 grid(1,1,1);
    grid.x = cv::cudev::divUp( key_points_.cols, block.x );
	kernel_extract_multi<<<grid, block>>>( sdEMF, total.ptr<int>() );
	cudaSafeCall ( cudaGetLastError () );
	//cudaSafeCall ( cudaDeviceSynchronize () );

	Mat cpu_total; total.download(cpu_total);
	std::vector<int> vFeatures;
	for (int i = 0; i < nFeatureScale_; i++)
		vFeatures.push_back(cpu_total.ptr<int>()[i]);

	for (int i = 0; i<nFeatureScale_; i++){
		int nF = vFeatures[i];
		if( nF > 0 ){
			pts_m[i].colRange(0,nF).copyTo((*ptr_v_pts_)[i]);
			nls_m[i].colRange(0,nF).copyTo((*ptr_v_nls_)[i]);
			mds_m[i].colRange(0,nF).copyTo((*ptr_v_mds_)[i]);
			p2d_m[i].colRange(0,nF).copyTo((*ptr_v_2d_)[i]);
			pbv_m[i].colRange(0, nF).copyTo((*ptr_v_bv_)[i]);
			wei_m[i].colRange(0, nF).copyTo((*ptr_v_weigts_)[i]);
			kyp_m[i].colRange(0,nF).copyTo((*ptr_v_key_points_)[i]);
			des_m[i].rowRange(0,nF).copyTo((*ptr_v_descriptor_)[i]);
		}
		else{
			(*ptr_v_pts_)[i].release();
			(*ptr_v_nls_)[i].release();
			(*ptr_v_mds_)[i].release();
			(*ptr_v_2d_)[i].release();
			(*ptr_v_bv_)[i].release();
			(*ptr_v_weigts_)[i].release();
			(*ptr_v_key_points_)[i].release();
			(*ptr_v_descriptor_)[i].release();
		}
	}
	return vFeatures;
}

__global__ void kernel_fill_combination(PtrStepSz<int> gpu_idx_, PtrStepSz<float> gpu_credibility_, PtrStepSz<uchar> gpu_flag_, int total_idx_){
	//traverse each element of the matrix
	const int nC = blockDim.x * blockIdx.x + threadIdx.x;
	const int nR = blockDim.y * blockIdx.y + threadIdx.y;

	if (nC >= total_idx_ || nR >= total_idx_) return;

	if (nC == nR ) return; //elemenate diagonal elements.
	//for example, we have a feature-pairset of {Aa, Ab, Bc, Bt} the FCM is 
	//     Aa Ab Bc Bt init flag
	// Aa  -  1  1  1   1 -> 0 if there is a better matches than Aa, 
	// Ab  1  -  1  1   1 
	// Bc  1  1  -  1   1 
	// Bt  1  1  1  -   1 -> 0

	int r_idx = *gpu_idx_.ptr(nR + 1);
	int c_idx = *gpu_idx_.ptr(nC + 1);

	//check corresponding credibility
	if (gpu_credibility_.ptr(r_idx)[c_idx] < 0.001f || gpu_credibility_.ptr(c_idx)[r_idx] < 0.001f)  // indicates 1-to-m 
	{
		int anchor_idx = *gpu_idx_.ptr(0);
		if (gpu_credibility_.ptr(r_idx)[anchor_idx] < gpu_credibility_.ptr(c_idx)[anchor_idx])//elemenate non-optimal matches
			gpu_flag_.ptr(nR+1)[0] = 0;
	}

	return;
}

void cuda_apply_1to1_constraint(const GpuMat& gpu_credibility_, const GpuMat& gpu_idx_, cv::Mat* p_1_to_1_)
{
	GpuMat gpu_flag; gpu_flag.create(gpu_idx_.size(), CV_8UC1); gpu_flag.setTo(uchar(1));

	//create combination matrix from second elment in gpu_idx_ to last
	int number = gpu_idx_.rows - 1;

	dim3 block(16, 16);
	dim3 grid(1, 1, 1);
	grid.x = grid.y = cv::cudev::divUp(number, block.x);
	kernel_fill_combination <<< grid, block >>>(gpu_idx_, gpu_credibility_, gpu_flag, number);
	//cudaSafeCall(cudaGetLastError());
	//cudaSafeCall(cudaDeviceSynchronize());

	Mat flag; gpu_flag.download(flag);  
	Mat& one_to_one = *p_1_to_1_; one_to_one.create(gpu_idx_.size(), CV_32SC1);
	Mat cpu_idx; gpu_idx_.download(cpu_idx);

	//get one to one feature pairs
	int nNode = 0;
	for (int i = 0; i < cpu_idx.rows; i++){
		if (*flag.ptr<uchar>(i) == 1)
			*one_to_one.ptr<int>(nNode++) = *cpu_idx.ptr<int>(i);
	}

	//Mat credibility; gpu_credibility_.download(credibility);
	//int nColumn = cpu_idx.ptr<int>()[0];
	//cout << nColumn << endl;
	//cout << credibility.col(nColumn) << endl;
	one_to_one.pop_back( one_to_one.rows - nNode );
	return;
}


__global__ void kernel_fill_combination_binary(PtrStepSz<int> gpu_idx_, PtrStepSz<float> gpu_credibility_, PtrStepSz<float4> geometry_, PtrStepSz<uchar> gpu_flag_, int total_idx_){
	//traverse each element of the matrix
	const int nC = blockDim.x * blockIdx.x + threadIdx.x;
	const int nR = blockDim.y * blockIdx.y + threadIdx.y;

	if (nC >= total_idx_ || nR >= total_idx_) return;

	if (nC == nR) return; //elemenate diagonal elements.
	//for example, we have a feature-pairset of {Aa, Ab, Bc, Bt} the FCM is 
	//     Aa Ab Bc Bt init flag
	// Aa  -  1  1  1   1 -> 0 if there is a better matches than Aa, 
	// Ab  1  -  1  1   1 
	// Bc  1  1  -  1   1 
	// Bt  1  1  1  -   1 -> 0

	int r_idx = gpu_idx_.ptr()[nR + 1];
	int c_idx = gpu_idx_.ptr()[nC + 1];

	//check corresponding credibility
	float4 gm = geometry_.ptr(r_idx)[c_idx];
	if (gm.x > 999.f )  // indicates 1-to-m 
	{
		int anchor_idx = *gpu_idx_.ptr(0);
		float4 gm1 = geometry_.ptr(r_idx)[anchor_idx];
		float4 gm2 = geometry_.ptr(c_idx)[anchor_idx];
		if ( gm1.x + gm1.y + gm1.z + gm1.w > gm2.x + gm2.y + gm2.z + gm2.w )//elemenate non-optimal matches
			gpu_flag_.ptr(nR + 1)[0] = 0;
	}

	return;
}

void cuda_apply_1to1_constraint_binary(const GpuMat& credibility_, const GpuMat& geometry_, const GpuMat& gpu_idx_, cv::Mat* p_1_to_1_)
{
	GpuMat gpu_flag; gpu_flag.create(gpu_idx_.size(), CV_8UC1); gpu_flag.setTo(uchar(1));

	//create combination matrix from second elment in gpu_idx_ to last
	int number = gpu_idx_.cols - 1;

	dim3 block(16, 16);
	dim3 grid(1, 1, 1);
	grid.x = grid.y = cv::cudev::divUp(number, block.x);
	kernel_fill_combination_binary << < grid, block >> >(gpu_idx_, credibility_, geometry_, gpu_flag, number);
	//cudaSafeCall(cudaGetLastError());
	//cudaSafeCall(cudaDeviceSynchronize());

	Mat flag; gpu_flag.download(flag);
	Mat& one_to_one = *p_1_to_1_; one_to_one.create(gpu_idx_.cols,1, CV_32SC1);
	Mat cpu_idx; gpu_idx_.download(cpu_idx);

	//get one to one feature pairs
	int nNode = 0;
	for (int i = 0; i < cpu_idx.cols; i++){
		if (flag.ptr<uchar>()[i] == 1)
			*one_to_one.ptr<int>(nNode++) = cpu_idx.ptr<int>()[i];
	}

	//Mat credibility; gpu_credibility_.download(credibility);
	//int nColumn = cpu_idx.ptr<int>()[0];
	//cout << nColumn << endl;
	//cout << credibility.col(nColumn) << endl;
	one_to_one.pop_back(one_to_one.rows - nNode);
	return;
}


__constant__ float _fExpThreshold;


__global__ void kernel_select_inliers(PtrStepSz<float> credibility_, PtrStepSz<float> column_idx_, PtrStepSz<int> inliers_, PtrStepSz<uchar> mask_counting){
	//traverse each element of the matrix
	const int nC = blockDim.x * blockIdx.x + threadIdx.x; //nC traverse each row of the colomn 
	const int nTrial = blockDim.y * blockIdx.y + threadIdx.y; //nTrial   

	if ( nTrial >= inliers_.rows || nC >= credibility_.cols ) return;

	int c_idx = column_idx_.ptr()[nTrial]; ///it gives the column with the high consistent scores first.

	if (nC == 0){
		*inliers_.ptr(nTrial) = c_idx; //store the column index
		*mask_counting.ptr(nTrial) = 1;
	}

	if (credibility_.ptr(nC)[c_idx] > _fExpThreshold){
		inliers_.ptr(nTrial)[nC] = nC;
		mask_counting.ptr(nTrial)[nC] = 1;
	}
	return;
}

void cuda_select_inliers_from_am(const GpuMat& gpu_credibility_, const GpuMat& gpu_column_idx_, float fExpThreshold_, int TotalTrials_, GpuMat* p_inliers_, GpuMat* p_node_numbers_){
	//p_node_numbers_->create(TotalTrials_, 1, CV_32SC1); p_node_numbers_->setTo(0);

	cudaSafeCall( cudaMemcpyToSymbol(_fExpThreshold, &fExpThreshold_, sizeof(float)) );
	int nCols = gpu_credibility_.cols;
	GpuMat mask_counting; mask_counting.create(TotalTrials_, nCols, CV_8UC1); mask_counting.setTo(0);
	p_inliers_->create(TotalTrials_, nCols, CV_32SC1); p_inliers_->setTo(std::numeric_limits<int>::max());

	dim3 block(4, 64);
	dim3 grid(1, 1, 1);
	grid.x = cv::cudev::divUp(nCols, block.y);
	grid.y = cv::cudev::divUp(TotalTrials_, block.x);
	//kernel_select_inliers <<< grid, block >>>(gpu_credibility_, gpu_column_idx_, *p_inliers_, *p_node_numbers_);
	kernel_select_inliers <<< grid, block >>>(gpu_credibility_, gpu_column_idx_, *p_inliers_, mask_counting );
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	Mat count; count.create(TotalTrials_, 1, CV_32SC1); 
	for (int i = 0; i < TotalTrials_; i++){
		*count.ptr<int>(i) = cuda::sum(mask_counting.row(i))[0];
		//sort
		if(*count.ptr<int>(i)>0){
			thrust::device_ptr<int> K(p_inliers_->ptr<int>(i));
			thrust::sort(K, K + p_inliers_->cols);
		}
	}
	p_node_numbers_->upload(count);
	return;
}

__global__ void kernel_select_inliers_atomic(PtrStepSz<float> credibility_, PtrStepSz<float> column_idx_, PtrStepSz<int> inliers_, PtrStepSz<int> node_numbers){
	//traverse each element of the matrix
	const int nTrial = blockDim.x * blockIdx.x + threadIdx.x; //nTrial 
	const int nR = blockDim.y * blockIdx.y + threadIdx.y; //nR 

	if (nTrial >= inliers_.cols || nR >= credibility_.rows) return;

	int c_idx = column_idx_.ptr()[nTrial]; ///it gives the column with the high consistent scores first.

	if (nR == 0){
		inliers_.ptr(nTrial)[0] = c_idx;
	}

	if (credibility_.ptr(nR)[c_idx] > _fExpThreshold){
		const int nNew = atomicAdd(node_numbers.ptr(nTrial), 1); //have to be this way
		if (nNew < inliers_.cols-1)
			inliers_.ptr(nTrial)[nNew + 1] = nR;
	}
	return;
}

void cuda_select_inliers_from_am(const GpuMat& gpu_credibility_, const GpuMat& gpu_column_idx_, float fExpThreshold_, int TotalTrials_, int ChainLength_, GpuMat* p_inliers_, GpuMat* p_node_numbers_){
	p_node_numbers_->create(TotalTrials_, 1, CV_32SC1); p_node_numbers_->setTo(0);

	cudaSafeCall(cudaMemcpyToSymbol(_fExpThreshold, &fExpThreshold_, sizeof(float)));
	int nCols = gpu_credibility_.cols;
	p_inliers_->create(TotalTrials_, nCols, CV_32SC1); p_inliers_->setTo(std::numeric_limits<int>::max());

	dim3 block(4, 64);
	dim3 grid(1, 1, 1);
	grid.x = cv::cudev::divUp(nCols, block.y);
	grid.y = cv::cudev::divUp(TotalTrials_, block.x);
	kernel_select_inliers_atomic <<< grid, block >>>(gpu_credibility_, gpu_column_idx_, *p_inliers_, *p_node_numbers_);
	cudaSafeCall(cudaGetLastError());
	//cudaSafeCall(cudaDeviceSynchronize());
	return;
}

__global__ void kernel_copy_selected_rows(PtrStepSz<float> credibility_, PtrStepSz<float> column_idx_, PtrStepSz<float> selected_columns_){
	const int nTrial = blockDim.x * blockIdx.x + threadIdx.x; //nC is the idx of sorted columns
	const int nC = blockDim.y * blockIdx.y + threadIdx.y; //NR is the idx of each row in a selected column

	if (nTrial >= selected_columns_.rows || nC >= credibility_.cols) return;

	int r_idx = (int)column_idx_.ptr()[nTrial];// extract row idx from column idx, this is because credibility_ is roughly symetric
	selected_columns_.ptr(nTrial)[nC] = credibility_.ptr(r_idx)[nC];

	return;
}

void select_rows(const GpuMat& gpu_credibility_, const GpuMat& gpu_column_idx_, int TotalTrials_, GpuMat* p_selected_credibility_){
	p_selected_credibility_->create(TotalTrials_, gpu_credibility_.cols, CV_32FC1);

	dim3 block(4, 64);
	dim3 grid(1, 1, 1);
	grid.x = cv::cudev::divUp(TotalTrials_, block.x);
	grid.y = cv::cudev::divUp(gpu_credibility_.cols, block.y);
	kernel_copy_selected_rows <<< grid, block >>>(gpu_credibility_, gpu_column_idx_, *p_selected_credibility_);
	//cudaSafeCall(cudaGetLastError());
	//cudaSafeCall(cudaDeviceSynchronize());
	return;
}

__global__ void kernel_select_inliers_from_sorted(PtrStepSz<float> credibility_, PtrStepSz<float> column_idx_, PtrStepSz<float> sorted_idx_, PtrStepSz<int> inliers_, PtrStepSz<int> node_numbers){
	//traverse each element of the matrix
	const int nTrial = blockDim.x * blockIdx.x + threadIdx.x; //nC is the idx of sorted columns
	const int nC = blockDim.y * blockIdx.y + threadIdx.y; //NR is the idx of each row in a selected column

	if ( nTrial >= inliers_.cols || nC >= credibility_.cols ) return;
	
	int c_idx = column_idx_.ptr()[nTrial]; ///it gives the column with the high consitent scores first.
	int r_idx = int( sorted_idx_.ptr(nTrial)[nC] );

	if (nC == 0){
		inliers_.ptr(0)[nTrial] = c_idx;
	}

	if (credibility_.ptr(c_idx)[r_idx] > _fExpThreshold){
		const int nNew = atomicAdd( node_numbers.ptr(nTrial), 1 ); //have to be this way
		if ( nNew < inliers_.rows-1 )
			inliers_.ptr(nNew + 1)[nTrial] = r_idx;
	}
	return;
}

void cuda_select_inliers_from_am_2(const GpuMat& gpu_credibility_, const GpuMat& gpu_column_idx_, float fExpThreshold_, int TotalTrials_, int ChainLength_, GpuMat* p_inliers_, GpuMat* p_node_numbers_){
	//1. copy top n columns 
	GpuMat selected_credibility;  select_rows(gpu_credibility_, gpu_column_idx_, TotalTrials_, &selected_credibility);
	//2. sort each columns
	GpuMat sorted_idx; sorted_idx.create(selected_credibility.size(), CV_32FC1); //stores the column of idx ranked from high consistency to low
	Mat test,scores;
	for (int r = 0; r < TotalTrials_; r++){
		GpuMat tmp_cred_row = selected_credibility.row(r); //have to be this way for gcc
		GpuMat tmp_idx_row = sorted_idx.row(r);
		btl::device::cuda_sort_column_idx(&tmp_cred_row, &tmp_idx_row );
		//sorted_idx.download(test);
		//selected_credibility.download(scores);
	}

	//3. select the most consistent scores
	p_inliers_->create(ChainLength_, TotalTrials_, CV_32SC1);
	p_node_numbers_->create(TotalTrials_, 1, CV_32SC1); p_node_numbers_->setTo(0);
	cudaSafeCall(cudaMemcpyToSymbol(_fExpThreshold, &fExpThreshold_, sizeof(float)));

	dim3 block(4, 64);
	dim3 grid(1, 1, 1);
	grid.x = cv::cudev::divUp(TotalTrials_, block.x);
	grid.y = cv::cudev::divUp(ChainLength_, block.y);
	kernel_select_inliers_from_sorted << < grid, block >> >(gpu_credibility_, gpu_column_idx_, sorted_idx, *p_inliers_, *p_node_numbers_);
	//cudaSafeCall(cudaGetLastError());
	//cudaSafeCall(cudaDeviceSynchronize());

	return;
}


}//device
}//btl
