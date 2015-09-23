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

#define INFO
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#endif
#include <thrust/device_ptr.h> 
#include <thrust/sort.h> 

#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda/common.hpp>
#include "pcl/device.hpp"
#include "pcl/vector_math.hpp"
#include "FeatureVolume.cuh"
#include "OtherUtil.hpp"

#include <iostream>
#include <vector>

using namespace pcl::device;
using namespace cv;
using namespace cv::cuda;
using namespace std;
namespace btl
{
namespace device
{
	using namespace cv::cuda;
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
};


__constant__	float __fVoxelSize[5]; //feature voxel size 128,64,32
__constant__	short3 __nGridResolution[5];
__constant__	int __nMaxFeatures[5];
__constant__	int __nFeatureScale;
__constant__	int __nEffectiveKeyPoints;
__constant__	int __nTotalInliers;
__constant__	int __nTotal;

__constant__ Mat33 __Rcurr;
__constant__ float3 __tcurr; //Cw camera center
__constant__ float __time_step;
__constant__ float __cell_size;
__constant__ float __inv_cell_size;
__constant__ float __half_inv_cell_size;
__constant__ int __rows;
__constant__ int __cols;
__constant__ short3 __VOLUME;
__constant__ Intr __intr;
__constant__ float3 __volume_size;
__constant__ int __block_size_x;
__constant__ int __block_size_y;

	__device__ __forceinline__ float
	getMinTime ( const float3& dir)
	{
		float txmin = ((dir.x > 0 ? 0.f : __volume_size.x) - __tcurr.x) / dir.x;
		float tymin = ((dir.y > 0 ? 0.f : __volume_size.y) - __tcurr.y) / dir.y;
		float tzmin = ((dir.z > 0 ? 0.f : __volume_size.z) - __tcurr.z) / dir.z;

		return fmax ( fmax (txmin, tymin), tzmin);
	}

	__device__ __forceinline__ float
	getMaxTime (const float3& dir)
	{
		float txmax = ( (dir.x > 0 ? __volume_size.x : 0.f) - __tcurr.x) / dir.x;
		float tymax = ( (dir.y > 0 ? __volume_size.y : 0.f) - __tcurr.y) / dir.y;
		float tzmax = ( (dir.z > 0 ? __volume_size.z : 0.f) - __tcurr.z) / dir.z;

		return fmin (fmin (txmax, tymax), tzmax);
	}


struct SDevIntegrateGlobalVolume{
        enum{
            PARALLEL_THREAD_X = 256,
        };

		PtrStepSz<short2> _cvgmVolume;

		
		int _block_c;
		int _block_r;


		int _nOffset[5];
		PtrStepSz<int> _total;


		//lvl 0+1
        //PtrStepSz<uchar> _cvgmFeatureVolumeFlag[3];
        PtrStepSz<int>   _cvgmFeatureVolumeIdx[5];
        PtrStepSz<float> _cvgmKeyPointGlobal[5];
        PtrStepSz<uchar> _cvgmDescriptorGlobal[5];
		
		//camera centre
		PtrStepSz<int> _key_array_2d;

		PtrStepSz<int> _inliers;
		PtrStepSz<int> _feature_volume_coordinate;

		//point cloud of curr frame in world
        PtrStepSz<float3> _cvgmPtsWorld; 
        PtrStepSz<float3> _cvgmNlsWorld;
		//output
        PtrStepSz<float3> _voxel_centre_world[5]; 
        PtrStepSz<int> _feature_idx[5]; 

		//features
		PtrStepSz<uchar> _cvgmDescriptorCurr;
        PtrStepSz<float> _cvgmKeyPointCurr;

		//
	__device__ __forceinline__ bool
	isInside (const int3& g, int lvl_ ) const
	{
		return (g.x >= 0 && g.y >= 0 && g.z >= 0 && g.x < __nGridResolution[lvl_].x && g.y < __nGridResolution[lvl_].y && g.z < __nGridResolution[lvl_].z);
	}
	__device__ __forceinline__ int3
    getVoxel (float3 point,int lvl_ ) const
    {
		int vx = __float2int_rd (point.x / __fVoxelSize[lvl_]);        // round to negative infinity
		int vy = __float2int_rd (point.y / __fVoxelSize[lvl_]);
		int vz = __float2int_rd (point.z / __fVoxelSize[lvl_]);

		return make_int3 (vx, vy, vz);
    }

	__device__ __forceinline__ int2
		getVoxel2(float3 point, int lvl_) const
	{
		return make_int2( __float2int_rd( point.z / __fVoxelSize[lvl_] ), __float2int_rd( point.y / __fVoxelSize[lvl_] ) * __nGridResolution[lvl_].x + __float2int_rd( point.x / __fVoxelSize[lvl_] ) );
	}

	__device__ __forceinline__ float3  getVoxelCentre (int3 point, int lvl_ ) const
    {
		//float x = ( point.x + 0.5f ) * __fVoxelSize[lvl_];
		//float y = ( point.y + 0.5f ) * __fVoxelSize[lvl_];
		//float z = ( point.z + 0.5f ) * __fVoxelSize[lvl_];
		return make_float3((point.x + 0.5f) * __fVoxelSize[lvl_], (point.y + 0.5f) * __fVoxelSize[lvl_], (point.z + 0.5f) * __fVoxelSize[lvl_]);
    }

	__device__ __forceinline__ bool
	isEmpty(const int3& n3Grid_,  PtrStepSz<int>& cvgmFeatureVolumeIdx_, const int lvl_ ) const
	{
		return cvgmFeatureVolumeIdx_.ptr(__nGridResolution[lvl_].x * n3Grid_.y + n3Grid_.x)[n3Grid_.z] < 0;
	}
	__device__ __forceinline__ void
	setOccupied( const int3& n3Grid_, PtrStepSz<uchar>* pcvgmFeatureVolumeFlag_, const int lvl_ ){
		pcvgmFeatureVolumeFlag_->ptr( __nGridResolution[lvl_].x * n3Grid_.y + n3Grid_.x)[n3Grid_.z] ++;
	}
	__device__ __forceinline__ int
	voxelIdx( const int3& n3Grid_, const int lvl_ ){
		 return __nGridResolution[lvl_].z * (__nGridResolution[lvl_].x * n3Grid_.y + n3Grid_.x) + n3Grid_.z;
	}
	__device__ __forceinline__ void
    setIdx( const int3& n3Grid_,  int nIdx_, PtrStepSz<int>* pcvgmFeatureVolumeIdx_, const int lvl_ ){
		pcvgmFeatureVolumeIdx_->ptr (__nGridResolution[lvl_].x * n3Grid_.y + n3Grid_.x)[n3Grid_.z] = nIdx_;
	}
	__device__ __forceinline__ int
	getIdx( const int3& n3Grid_,  const PtrStepSz<int>& cvgmFeatureVolumeIdx_, const int lvl_  ){
		return cvgmFeatureVolumeIdx_.ptr (__nGridResolution[lvl_].x * n3Grid_.y + n3Grid_.x)[n3Grid_.z];
	}
	__device__ __forceinline__ bool
	isClose( const float3& f3OldPt_,const float3& f3OldNl_, const float3& f3Pt_, const float3& f3Nl_, const int lvl_ ) {
		using namespace pcl::device;
		float3 f3D = f3OldPt_ - f3Pt_;
		float fD = sqrtf(dot3<float, float3>(f3D, f3D));
		//float fCos = dot3(f3OldNl_, f3OldNl_);
		if( fD < __fVoxelSize[lvl_] /*&& fCos > 0.8*/ ) //+- 15 degree
			return true;
		else
			return false;
	}
	/*__device__ __forceinline__ bool
	isSimilar( const uchar* pDes1_, const uchar* pDes2_ ){
		short ucRes = 0;
		for(short s = 0; s<64; s++)
			ucRes += _popCountTable[ pDes1_[s] ^ pDes2_[s] ];
		return ucRes < _fMatchDistance;
	}*/
	__device__ __forceinline__ bool
	isDifferent( const uchar* pDes1_, const uchar* pDes2_ ){
		short ucRes = 0;
		for(short s = 0; s<64; s++)
			ucRes += _popCountTable[ pDes1_[s] ^ pDes2_[s] ];
		return ucRes > 200;
	}
	__device__ __forceinline__ void
	averageDescriptor( const float fWeight_, const uchar* pDes1_, uchar* pDes2_ ){
		uchar New;
		for( int i=0; i < 64 ; i ++ ){
			if ( _popCountTable[ pDes1_[i] ^ pDes2_[i] ] != 0){
				New = 0;
				for (int b=0; b< 8; b++){
					uchar mask = 1 << b;
					float bg = (pDes1_[i] & mask) >> b;
					float bc = (pDes2_[i] & mask) >> b;
					float bt = ( bg*fWeight_ + bc )/(fWeight_ + 1);
					uchar uR = bt >= .5f ? 1:0;
					uR <<= 7; //put it to the left
					New |= uR;
					if( b < 7)
						New >>= 1;//move right
				}
			}
			else {
				New = pDes1_[i];
			}
			pDes2_[i] = New;
		}
		return;
	}
	void __device__ __forceinline__ normalize(float3* pVec_ ) const{
		float norm = rsqrt( pVec_->x * pVec_->x + pVec_->y * pVec_->y + pVec_->z * pVec_->z );
		pVec_->x *= norm;
		pVec_->y *= norm;
		pVec_->z *= norm;
		return;
	}
	
	//traverse all voxels and searh for occupide voxels and return the voxel center to _cvgmPtsWorld.
	__device__ __forceinline__ void getOccupiedVoxels(){
		const int nX = blockDim.x * blockIdx.x + threadIdx.x;
		const int nY = blockDim.y * blockIdx.y + threadIdx.y;
		const int nZ = blockDim.z * blockIdx.z + threadIdx.z;

		//int3 n3Grid = make_int3( nX, nY, nZ );
		for (int lvl = 0; lvl < __nFeatureScale; lvl++){
			if (!(nX >= 0 && nY >= 0 && nZ >= 0 && nX < __nGridResolution[lvl].x && nY < __nGridResolution[lvl].y && nZ < __nGridResolution[lvl].z)) continue;
			float nR = __nGridResolution[lvl].x * nY + nX;
			int idx = _cvgmFeatureVolumeIdx[lvl].ptr(nR)[nZ];
			if (idx < 0) continue;
			const int nCounter = atomicAdd(_total.ptr() + lvl, 1);// accumulate(lvl, total_); //have to be this way
			if( nCounter >= __nMaxFeatures[lvl] ) return; // if not enough memory to hold the features, return
			_voxel_centre_world[lvl].ptr()[nCounter] = make_float3((nX + 0.5f) * __fVoxelSize[lvl], (nY + 0.5f) * __fVoxelSize[lvl], (nZ + 0.5f) * __fVoxelSize[lvl]);//getVoxelCentre(n3Grid, lvl);
			_feature_idx[lvl].ptr()[nCounter] = idx;// getIdx(n3Grid, _cvgmFeatureVolumeIdx[lvl], lvl);
		}
		return;
	}

	__device__ __forceinline__ float3
	get_ray_next (int x, int y) const
	{
		float3 ray_next;
		ray_next.x = (x - __intr.cx) / __intr.fx;
		ray_next.y = (y - __intr.cy) / __intr.fy;
		ray_next.z = 1;
		return ray_next;
	}


	__device__ __forceinline__ float
    readTsdf (int x, int y, int z) const
    {
		return unpack_tsdf (_cvgmVolume.ptr (__VOLUME.x * y + x)[z]);
    }
	__device__ __forceinline__ int3
    getVoxel (const float3& point) const
    {
		// round to negative infinity
		return make_int3(__float2int_rd(point.x * __inv_cell_size), __float2int_rd(point.y * __inv_cell_size), __float2int_rd(point.z * __inv_cell_size));
    }
	__device__ __forceinline__ bool
	checkInds (const int3& g) const
	{
		return (g.x >= 0 && g.y >= 0 && g.z >= 0 && g.x < __VOLUME.x && g.y < __VOLUME.y && g.z < __VOLUME.z);
	}

	__device__ __forceinline__ float
    interpolateTrilineary (const float3& origin, const float3& dir, float time) const
    {
		return interpolateTrilineary (origin + dir * time);
    }

    __device__ __forceinline__ float
    interpolateTrilineary (const float3& point) const
    {
		int3 g = getVoxel (point);

		if (g.x <= 0 || g.x >= __VOLUME.x - 1 || g.y <= 0 || g.y >= __VOLUME.y - 1 || g.z <= 0 || g.z >= __VOLUME.z - 1)
			return pcl::device::numeric_limits<float>::quiet_NaN ();

		float vx = (g.x + 0.5f) * __cell_size;
		float vy = (g.y + 0.5f) * __cell_size;
		float vz = (g.z + 0.5f) * __cell_size;

		g.x = (point.x < vx) ? (g.x - 1) : g.x;
		g.y = (point.y < vy) ? (g.y - 1) : g.y;
		g.z = (point.z < vz) ? (g.z - 1) : g.z;

		float a = (point.x - (g.x + 0.5f) * __cell_size) * __inv_cell_size;
		float b = (point.y - (g.y + 0.5f) * __cell_size) * __inv_cell_size;
		float c = (point.z - (g.z + 0.5f) * __cell_size) * __inv_cell_size;

		return  readTsdf (g.x + 0, g.y + 0, g.z + 0) * (1 - a) * (1 - b) * (1 - c) +
				readTsdf (g.x + 0, g.y + 0, g.z + 1) * (1 - a) * (1 - b) * c +
				readTsdf (g.x + 0, g.y + 1, g.z + 0) * (1 - a) * b * (1 - c) +
				readTsdf (g.x + 0, g.y + 1, g.z + 1) * (1 - a) * b * c +
				readTsdf (g.x + 1, g.y + 0, g.z + 0) * a * (1 - b) * (1 - c) +
				readTsdf (g.x + 1, g.y + 0, g.z + 1) * a * (1 - b) * c +
				readTsdf (g.x + 1, g.y + 1, g.z + 0) * a * b * (1 - c) +
				readTsdf (g.x + 1, g.y + 1, g.z + 1) * a * b * c;
    }
	//give the image coordinate x and y, find the interpolated 3D points and normal 
	//from the volumetric TSDF 
	__device__ __forceinline__ void
    getRefinedPtnNl (int x, int y, float3& Pt_, float3& Nl_ ) const
    {
		//if (x >= __cols || y >= __rows) return; has been tested outside
		Pt_.z = pcl::device::numeric_limits<float>::quiet_NaN ();

		Nl_ /*ray_dir*/= normalized<float, float3>(__Rcurr * get_ray_next(x, y));
		//printf("2.1 - r: %d c %d dir %f %f %f\n", y, x, Nl_.x, Nl_.y, Nl_.z);

        // computer time when entry and exit volume
		float time_start_volume = getMinTime(Nl_ /*ray_dir*/); time_start_volume = fmax(time_start_volume, 0.f);
		float time_exit_volume  = getMaxTime(Nl_ /*ray_dir*/);
        
		if (time_start_volume >= time_exit_volume)	return;

        float time_curr = time_start_volume;
		int3 g = getVoxel(__tcurr + Nl_ /*ray_dir*/ * time_curr);
        g.x = max (1, min (g.x, __VOLUME.x - 2));
        g.y = max (1, min (g.y, __VOLUME.y - 2));
        g.z = max (1, min (g.z, __VOLUME.z - 2));

        Pt_.x/*tsdf*/ = readTsdf (g.x, g.y, g.z);
		//printf("2.2 - tsdf %f = x%d y%d z%d\n", Pt_.x, g.x, g.y, g.z);
		//printf("2.3 - time_curr %f time_exist_volume %f time_step %f\n", time_curr, time_exit_volume, __time_step);
		//infinite loop guard
		for (; time_curr < time_exit_volume; time_curr += __time_step)//const float max_time = time_exit_volume; //3 * (volume_size.x + volume_size.y + volume_size.z);
        {
			Pt_.y/*tsdf_prev*/ = Pt_.x/*tsdf*/;

			g = getVoxel(__tcurr + Nl_ /*ray_dir*/ * (time_curr + __time_step));
			if (!checkInds(g))  break;

			Pt_.x/*tsdf*/ = readTsdf(g.x, g.y, g.z);
			if (Pt_.y/*tsdf_prev*/ < 0.f && Pt_.x/*tsdf*/ >= 0.f)	break;

			if (Pt_.y/*tsdf_prev*/ >= 0.f && Pt_.x/*tsdf*/ < 0.f)  //zero crossing
			{
				//printf("2.4 - zero crossing prev %f curr %f\n", Pt_.y, Pt_.x);

				Pt_.x/*Ftdt*/ = interpolateTrilineary(__tcurr, Nl_/*ray_dir*/, time_curr + __time_step); if (isnan(Pt_.x/*Ftdt*/)) break;
				Pt_.y/*Ft*/   = interpolateTrilineary(__tcurr, Nl_/*ray_dir*/, time_curr);               if (isnan(Pt_.y/*Ft*/)) break;

				Pt_ = __tcurr + Nl_/*ray_dir*/ * (time_curr - __time_step * Pt_.y/*Ft*/ / (Pt_.x/*Ftdt*/ - Pt_.y/*Ft*/));

				g = getVoxel(Pt_);
				if (g.x > 1 && g.y > 1 && g.z > 1 && g.x < __VOLUME.x - 2 && g.y < __VOLUME.y - 2 && g.z < __VOLUME.z - 2)
				{
					Nl_.x = interpolateTrilineary(make_float3(Pt_.x + __cell_size, Pt_.y, Pt_.z)) - interpolateTrilineary(make_float3(Pt_.x - __cell_size, Pt_.y, Pt_.z));
					Nl_.y = interpolateTrilineary(make_float3(Pt_.x, Pt_.y + __cell_size, Pt_.z)) - interpolateTrilineary(make_float3(Pt_.x, Pt_.y - __cell_size, Pt_.z));
					Nl_.z = interpolateTrilineary(make_float3(Pt_.x, Pt_.y, Pt_.z + __cell_size)) - interpolateTrilineary(make_float3(Pt_.x, Pt_.y, Pt_.z - __cell_size));
					Nl_ = normalized<float, float3>(Nl_);
				}
				else{
					Pt_.z = pcl::device::numeric_limits<float>::quiet_NaN();
				}
				break;
			}//if (tsdf_prev > 0.f && tsdf < 0.f) 
      }//for (; time_curr < max_time; time_curr += __time_step)
   }//operator ()
	
	//__device__ __forceinline__ void
	//getRefinedPtnNl(float3& Pt_, float3& Nl_) 
	//{
	//	//if (x >= __cols || y >= __rows) return; has been tested outside
	//	Nl_.z = pcl::device::numeric_limits<float>::quiet_NaN();

	//	Nl_ /*ray_dir*/ = normalized<float, float3>(__Rcurr * Pt_);
	//	//printf("2.1 - r: %d c %d dir %f %f %f\n", y, x, Nl_.x, Nl_.y, Nl_.z);

	//	// computer time when entry and exit volume
	//	float time_start_volume = getMinTime(Nl_ /*ray_dir*/); time_start_volume = fmax(time_start_volume, 0.f);
	//	float time_exit_volume = getMaxTime(Nl_ /*ray_dir*/);

	//	if (time_start_volume >= time_exit_volume)	return;

	//	float time_curr = time_start_volume;
	//	int3 g = getVoxel(__tcurr + Nl_ /*ray_dir*/ * time_curr);
	//	g.x = max(1, min(g.x, __VOLUME_X - 2));
	//	g.y = max(1, min(g.y, __VOLUME_X - 2));
	//	g.z = max(1, min(g.z, __VOLUME_X - 2));

	//	Pt_.x/*tsdf*/ = readTsdf(g.x, g.y, g.z);
	//	//printf("2.2 - tsdf %f = x%d y%d z%d\n", Pt_.x, g.x, g.y, g.z);
	//	//printf("2.3 - time_curr %f time_exist_volume %f time_step %f\n", time_curr, time_exit_volume, __time_step);
	//	//infinite loop guard
	//	for (; time_curr < time_exit_volume; time_curr += __time_step)//const float max_time = time_exit_volume; //3 * (volume_size.x + volume_size.y + volume_size.z);
	//	{
	//		Pt_.y/*tsdf_prev*/ = Pt_.x/*tsdf*/;

	//		g = getVoxel(__tcurr + Nl_ /*ray_dir*/ * (time_curr + __time_step));
	//		if (!checkInds(g))  break;

	//		Pt_.x/*tsdf*/ = readTsdf(g.x, g.y, g.z);
	//		if (Pt_.y/*tsdf_prev*/ < 0.f && Pt_.x/*tsdf*/ >= 0.f)	break;

	//		if (Pt_.y/*tsdf_prev*/ >= 0.f && Pt_.x/*tsdf*/ < 0.f)  //zero crossing
	//		{
	//			//printf("2.4 - zero crossing prev %f curr %f\n", Pt_.y, Pt_.x);

	//			Pt_.x/*Ftdt*/ = interpolateTrilineary(__tcurr, Nl_/*ray_dir*/, time_curr + __time_step); if (isnan(Pt_.x/*Ftdt*/)) break;
	//			Pt_.y/*Ft*/ = interpolateTrilineary(__tcurr, Nl_/*ray_dir*/, time_curr);               if (isnan(Pt_.y/*Ft*/)) break;

	//			Pt_ = __tcurr + Nl_/*ray_dir*/ * (time_curr - __time_step * Pt_.y/*Ft*/ / (Pt_.x/*Ftdt*/ - Pt_.y/*Ft*/));

	//			g = getVoxel(Pt_);
	//			if (g.x > 1 && g.y > 1 && g.z > 1 && g.x < __VOLUME_X - 2 && g.y < __VOLUME_X - 2 && g.z < __VOLUME_X - 2)
	//			{
	//				Nl_.x = interpolateTrilineary(make_float3(Pt_.x + __cell_size, Pt_.y, Pt_.z)) - interpolateTrilineary(make_float3(Pt_.x - __cell_size, Pt_.y, Pt_.z));
	//				Nl_.y = interpolateTrilineary(make_float3(Pt_.x, Pt_.y + __cell_size, Pt_.z)) - interpolateTrilineary(make_float3(Pt_.x, Pt_.y - __cell_size, Pt_.z));
	//				Nl_.z = interpolateTrilineary(make_float3(Pt_.x, Pt_.y, Pt_.z + __cell_size)) - interpolateTrilineary(make_float3(Pt_.x, Pt_.y, Pt_.z - __cell_size));
	//				Nl_ = normalized<float, float3>(Nl_);
	//			}
	//			else{
	//				Pt_.z = pcl::device::numeric_limits<float>::quiet_NaN();
	//			}
	//			break;
	//		}//if (tsdf_prev > 0.f && tsdf < 0.f) 
	//	}//for (; time_curr < max_time; time_curr += __time_step)
	//}//operator ()
	//1. traverse the image of idx, _key_array_2d, 
   //2. if the idx is legal, get its corresponding 3d points and surface normal 
   //3. if the pt and nl are legal
	__device__ __forceinline__ void insert_features_into_volume(int* total_,int* replace_){
		int c = _block_c + threadIdx.x * __block_size_x; //idx of key point in curr frame
		int r = _block_r + threadIdx.y * __block_size_y; //idx of key point in curr frame
		//printf("thread x: %d y: %d\n", threadIdx.x, threadIdx.y);
		//printf("block x: %d y: %d\n", blockIdx.x, blockIdx.y);
		//printf("thread blockDim: %d gridDim: %d\n", blockDim.x, gridDim.x);
		//printf("c: %d r: %d\n", c, r);
		if (c >= __cols || r >= __rows)  return;
		int nSelectedIdx = _key_array_2d.ptr(r)[c];
		if (nSelectedIdx < 0) return;
		//printf("nSelectedIdx = %d\n", nSelectedIdx);

		if (nSelectedIdx >= __nEffectiveKeyPoints)
			printf("Failure - idx error. 1. ID: %d i %d %f %f\n", c, nSelectedIdx, _cvgmKeyPointCurr.ptr(0)[ nSelectedIdx ], _cvgmKeyPointCurr.ptr(1)[ nSelectedIdx ] );

		//printf("2 - r: %d c %d\n", r, c);
        //get normal and pt
        float3 f3NewPt, f3NewNl;//curr frame in world
		getRefinedPtnNl( c, r, f3NewPt, f3NewNl );
		//printf("3 pt %f %f %f\n", f3NewPt.x, f3NewPt.y, f3NewPt.z );

		//if no depth data return;
        if( isnan<float>( f3NewPt.z )  ) return; 
        const float fResponseCurr = _cvgmKeyPointCurr.ptr(6)[ nSelectedIdx ];
		//printf("here. 0. response %f \n", fResponseCurr);
        if( fResponseCurr < 0.f ) return;
		const float3 f3VisualRay = normalized<float, float3>(__tcurr - f3NewPt);
		const float3 f3MD = make_float3( _cvgmKeyPointCurr.ptr(3)[ nSelectedIdx ], _cvgmKeyPointCurr.ptr(4)[ nSelectedIdx ], _cvgmKeyPointCurr.ptr(5)[ nSelectedIdx ] );
		const float fSize = _cvgmKeyPointCurr.ptr(2)[ nSelectedIdx ]; //the radias 
		const int lvl = (int)_cvgmKeyPointCurr.ptr(9)[nSelectedIdx]; //the radias 9
		//printf("lvl \t %d \n", lvl);
		//calc grid idx
        const int3 n3Grid = getVoxel( f3NewPt, lvl );
		//printf("here. 1. %d %d %d grid %d lvl %d\n", n3Grid.x, n3Grid.y, n3Grid.z, __nGridNo[lvl], lvl);
		if( isInside( n3Grid, lvl ) ){
			int nIdxGlobal = _cvgmFeatureVolumeIdx[lvl].ptr(__nGridResolution[lvl].x * n3Grid.y + n3Grid.x)[n3Grid.z];
			//if the voxel is empty
            //if( isEmpty( n3Grid, _cvgmFeatureVolumeFlag[lvl],lvl ) ){
            if( nIdxGlobal < 0 ){ //the voxel is empty
				const float fCos = dot3<float, float3>(f3VisualRay, f3NewNl);	if( fCos < 0 ) return; //cos alpha
				int nCol = atomicAdd(&total_[lvl], 1); //accumulate(lvl, total_);
				nCol += _nOffset[lvl]; if (nCol >= __nMaxFeatures[lvl]) return; // if not enough memory to hold the features, return
				//printf("New. c %d existing %d levl %d\n ",  nCol , _nOffset[lvl] , lvl);
                //setIdx( n3Grid, nCol , &_cvgmFeatureVolumeIdx[lvl], lvl );
				_cvgmFeatureVolumeIdx[lvl].ptr(__nGridResolution[lvl].x * n3Grid.y + n3Grid.x)[n3Grid.z] = nCol;
                //insert descriptor
                memcpy( _cvgmDescriptorGlobal[lvl].ptr( nCol  ) , _cvgmDescriptorCurr.ptr( nSelectedIdx ), sizeof(uchar)*64 );
                //insert 3d point
                _cvgmKeyPointGlobal[lvl].ptr(0)[ nCol ] = f3NewPt.x;
                _cvgmKeyPointGlobal[lvl].ptr(1)[ nCol ] = f3NewPt.y;
                _cvgmKeyPointGlobal[lvl].ptr(2)[ nCol ] = f3NewPt.z;
                //insert normal
                _cvgmKeyPointGlobal[lvl].ptr(3)[ nCol ] = f3NewNl.x;
                _cvgmKeyPointGlobal[lvl].ptr(4)[ nCol ] = f3NewNl.y;
                _cvgmKeyPointGlobal[lvl].ptr(5)[ nCol ] = f3NewNl.z;
                //insert response
                _cvgmKeyPointGlobal[lvl].ptr(6)[ nCol ] = fResponseCurr; //HESSIAN_ROW
                //insert feature size
                _cvgmKeyPointGlobal[lvl].ptr(7)[ nCol ] = fSize; 
				//insert cosine angle between point normal and the visual ray
				_cvgmKeyPointGlobal[lvl].ptr(8)[ nCol ] = fCos;
				_cvgmKeyPointGlobal[lvl].ptr(9)[ nCol ]  = f3MD.x;
				_cvgmKeyPointGlobal[lvl].ptr(10)[ nCol ] = f3MD.y;
				_cvgmKeyPointGlobal[lvl].ptr(11)[ nCol ] = f3MD.z;
                //integrate into volume
            }
            else{ //if the voxel is occupied, keep the salient one
				//Note that here will have serious 
                if ( isnan<float> ( _cvgmKeyPointGlobal[lvl].ptr(0)[nIdxGlobal] ) || isnan<float> ( _cvgmKeyPointGlobal[lvl].ptr(1)[nIdxGlobal] ) || isnan<float> ( _cvgmKeyPointGlobal[lvl].ptr(2)[nIdxGlobal] )
					|| isnan<float> ( _cvgmKeyPointGlobal[lvl].ptr(3)[nIdxGlobal] ) || isnan<float> ( _cvgmKeyPointGlobal[lvl].ptr(4)[nIdxGlobal] ) || isnan<float> ( _cvgmKeyPointGlobal[lvl].ptr(5)[nIdxGlobal] )
					|| isnan<float> ( _cvgmKeyPointGlobal[lvl].ptr(6)[nIdxGlobal] ) || isnan<float> ( _cvgmKeyPointGlobal[lvl].ptr(7)[nIdxGlobal] ) || isnan<float> ( _cvgmKeyPointGlobal[lvl].ptr(8)[nIdxGlobal] ) ) {
					//printf("Failure - read idx error. ID: %d lvl%d\n ", nIdx , lvl);
					return;
				}

				const float fResponseGlobal = _cvgmKeyPointGlobal[lvl].ptr(6)[ nIdxGlobal ];
				const float fCosAngleOld    = _cvgmKeyPointGlobal[lvl].ptr(8)[ nIdxGlobal ];
				const float fCosAngleNew = dot3<float, float3>(f3VisualRay, f3NewNl); //cos alpha
				//printf("here. 1. %d %d %d grid %d lvl %d %f %f\n", n3Grid.x, n3Grid.y, n3Grid.z, __nGridNo[lvl], lvl, fResponseGlobal, fResponseCurr);

				if ( fResponseGlobal < fResponseCurr && fCosAngleOld < fCosAngleNew ){

					//const int nCounter = accumulate(lvl, replace_);
					//printf("Replace. c %d lvl %d \n ", nIdxGlobal, lvl);
					//replace existing feature with the new
					//insert descriptor
					memcpy( _cvgmDescriptorGlobal[lvl].ptr( nIdxGlobal ) , _cvgmDescriptorCurr.ptr( nSelectedIdx ), sizeof(uchar)*64 );
					//insert pts
					_cvgmKeyPointGlobal[lvl].ptr(0)[nIdxGlobal] = f3NewPt.x;
					_cvgmKeyPointGlobal[lvl].ptr(1)[nIdxGlobal] = f3NewPt.y;
					_cvgmKeyPointGlobal[lvl].ptr(2)[nIdxGlobal] = f3NewPt.z;
					//insert normal
					_cvgmKeyPointGlobal[lvl].ptr(3)[nIdxGlobal] = f3NewNl.x;
					_cvgmKeyPointGlobal[lvl].ptr(4)[nIdxGlobal] = f3NewNl.y;
					_cvgmKeyPointGlobal[lvl].ptr(5)[nIdxGlobal] = f3NewNl.z;
					//insert response
					_cvgmKeyPointGlobal[lvl].ptr(6)[nIdxGlobal] = fResponseCurr; //HESSIAN_ROW
					//insert feature size
					_cvgmKeyPointGlobal[lvl].ptr(7)[nIdxGlobal] = fSize; 
					//insert cosine angle between point normal and the visual ray
					_cvgmKeyPointGlobal[lvl].ptr(8)[nIdxGlobal] = fCosAngleNew; 

					_cvgmKeyPointGlobal[lvl].ptr(9)[ nIdxGlobal ]  = f3MD.x;
					_cvgmKeyPointGlobal[lvl].ptr(10)[ nIdxGlobal ] = f3MD.y;
					_cvgmKeyPointGlobal[lvl].ptr(11)[ nIdxGlobal ] = f3MD.z;
				}
            }//if occupied
		}//if //inside
        return;
    }//mainFunc()

	__device__ __forceinline__ void insert_features_into_volume2(){
		int nX = blockDim.x * blockIdx.x + threadIdx.x; if (nX >= __nTotal) return;
		//printf("thread x: %d y: %d\n", threadIdx.x, threadIdx.y);
		int kp_idx = _inliers.ptr()[nX];
		if (kp_idx >= __nEffectiveKeyPoints || kp_idx < 0) {
			printf("Failure - idx error. 1. ID: %d i %d %f %f\n", nX, kp_idx, _cvgmKeyPointCurr.ptr(0)[kp_idx], _cvgmKeyPointCurr.ptr(1)[kp_idx]);
			return;
		}

		//get normal and pt
		float3 f3NewPt, f3NewNl;//curr frame in world
		getRefinedPtnNl((int)_cvgmKeyPointCurr.ptr(0)[kp_idx], (int)_cvgmKeyPointCurr.ptr(1)[kp_idx], f3NewPt, f3NewNl); if (isnan<float>(f3NewPt.z)) return;
		//printf("3 pt %f %f %f\n", f3NewPt.x, f3NewPt.y, f3NewPt.z );
		const int lvl = (int)_cvgmKeyPointCurr.ptr(9)[kp_idx]; //the radias 9
		//calc grid idx
		int2 volume_coo = getVoxel2(f3NewPt, lvl);
		if (__nGridResolution[lvl].z * volume_coo.y + volume_coo.x != _feature_volume_coordinate.ptr()[nX]) { /*printf("_feature_volume_coordinate.ptr()[nX] %d \n ", _feature_volume_coordinate.ptr()[nX]);*/ return; }//
		const float fResponseCurr = _cvgmKeyPointCurr.ptr(6)[kp_idx];  if (fResponseCurr < 0.f) return;
		const float fCosAngleNew = dot3<float, float3>(normalized<float, float3>(__tcurr - f3NewPt), f3NewNl); if (fCosAngleNew < 0) return; //cos alpha
		nX /*nIdxGlobal*/= _cvgmFeatureVolumeIdx[lvl].ptr(volume_coo.y)[volume_coo.x];
		//if the voxel is empty
		if (nX /*nIdxGlobal*/ < 0){ //the voxel is empty
			nX /*nCol*/ = atomicAdd((_total.ptr() + lvl), 1); //accumulate(lvl, total_);
			nX /*nCol*/ += _nOffset[lvl]; if (nX /*nCol*/ >= __nMaxFeatures[lvl]) return; // if not enough memory to hold the features, return
			//printf("New. c %d existing %d levl %d\n ",  nX , _nOffset[lvl], lvl);
			_cvgmFeatureVolumeIdx[lvl].ptr(volume_coo.y)[volume_coo.x] = nX /*nCol*/;
			//insert descriptor
			memcpy(_cvgmDescriptorGlobal[lvl].ptr(nX /*nCol*/), _cvgmDescriptorCurr.ptr(kp_idx), sizeof(uchar) * 64);
			//insert 3d point
			_cvgmKeyPointGlobal[lvl].ptr(0)[nX /*nCol*/] = f3NewPt.x;
			_cvgmKeyPointGlobal[lvl].ptr(1)[nX /*nCol*/] = f3NewPt.y;
			_cvgmKeyPointGlobal[lvl].ptr(2)[nX /*nCol*/] = f3NewPt.z;
			//insert normal
			_cvgmKeyPointGlobal[lvl].ptr(3)[nX /*nCol*/] = f3NewNl.x;
			_cvgmKeyPointGlobal[lvl].ptr(4)[nX /*nCol*/] = f3NewNl.y;
			_cvgmKeyPointGlobal[lvl].ptr(5)[nX /*nCol*/] = f3NewNl.z;
			//insert response
			_cvgmKeyPointGlobal[lvl].ptr(6)[nX /*nCol*/] = fResponseCurr; //HESSIAN_ROW
			//insert feature size
			_cvgmKeyPointGlobal[lvl].ptr(7)[nX /*nCol*/] = _cvgmKeyPointCurr.ptr(2)[kp_idx];
			//insert cosine angle between point normal and the visual ray
			_cvgmKeyPointGlobal[lvl].ptr(8)[nX /*nCol*/] = fCosAngleNew;
			_cvgmKeyPointGlobal[lvl].ptr(9)[nX /*nCol*/] = _cvgmKeyPointCurr.ptr(3)[kp_idx];
			_cvgmKeyPointGlobal[lvl].ptr(10)[nX /*nCol*/] = _cvgmKeyPointCurr.ptr(4)[kp_idx];
			_cvgmKeyPointGlobal[lvl].ptr(11)[nX /*nCol*/] = _cvgmKeyPointCurr.ptr(5)[kp_idx];
			//integrate into volume
		}
		else{ //if the voxel is occupied, keep the salient one
			//printf("Old. c %d existing %f levl %f\n ", nX, fResponseCurr, _cvgmKeyPointGlobal[lvl].ptr(6)[nX /*nIdxGlobal*/]);
			//Note that here will have serious 
			if (_cvgmKeyPointGlobal[lvl].ptr(6)[nX /*nIdxGlobal*/] < fResponseCurr && _cvgmKeyPointGlobal[lvl].ptr(8)[nX /*nIdxGlobal*/] < fCosAngleNew){

				//const int nCounter = accumulate(lvl, replace_);
				//printf("Replace. c %d lvl %d \n ", nIdxGlobal, lvl);
				//replace existing feature with the new
				//insert descriptor
				memcpy(_cvgmDescriptorGlobal[lvl].ptr(nX /*nIdxGlobal*/), _cvgmDescriptorCurr.ptr(kp_idx), sizeof(uchar) * 64);
				//insert pts
				_cvgmKeyPointGlobal[lvl].ptr(0)[nX /*nIdxGlobal*/] = f3NewPt.x;
				_cvgmKeyPointGlobal[lvl].ptr(1)[nX /*nIdxGlobal*/] = f3NewPt.y;
				_cvgmKeyPointGlobal[lvl].ptr(2)[nX /*nIdxGlobal*/] = f3NewPt.z;
				//insert normal
				_cvgmKeyPointGlobal[lvl].ptr(3)[nX /*nIdxGlobal*/] = f3NewNl.x;
				_cvgmKeyPointGlobal[lvl].ptr(4)[nX /*nIdxGlobal*/] = f3NewNl.y;
				_cvgmKeyPointGlobal[lvl].ptr(5)[nX /*nIdxGlobal*/] = f3NewNl.z;
				//insert response
				_cvgmKeyPointGlobal[lvl].ptr(6)[nX /*nIdxGlobal*/] = fResponseCurr; //HESSIAN_ROW
				//insert feature size
				_cvgmKeyPointGlobal[lvl].ptr(7)[nX /*nIdxGlobal*/] = _cvgmKeyPointCurr.ptr(2)[kp_idx];
				//insert cosine angle between point normal and the visual ray
				_cvgmKeyPointGlobal[lvl].ptr(8)[nX /*nIdxGlobal*/] = fCosAngleNew;

				_cvgmKeyPointGlobal[lvl].ptr(9)[nX /*nIdxGlobal*/] = _cvgmKeyPointCurr.ptr(3)[kp_idx];
				_cvgmKeyPointGlobal[lvl].ptr(10)[nX /*nIdxGlobal*/] = _cvgmKeyPointCurr.ptr(4)[kp_idx];
				_cvgmKeyPointGlobal[lvl].ptr(11)[nX /*nIdxGlobal*/] = _cvgmKeyPointCurr.ptr(5)[kp_idx];
			}
		}//if occupied
		return;
	}//mainFunc()

};

__global__ void kernelGetOccupiedVoxels(SDevIntegrateGlobalVolume sDIGV_ ){
	sDIGV_.getOccupiedVoxels();
}

std::vector<int> cuda_get_occupied_vg ( const GpuMat*  feature_volume_idx_ ,
									 const float* fVoxelSize_, const int nFeatureScale_,
									 GpuMat* ptr_pts_world_, const vector<short3>& vResolutions_,
									 GpuMat* ptr_feature_idx_)
{
    SDevIntegrateGlobalVolume sDIGV;

	float fVoxelSize[5];
	short3 nGridResolution[5];
	int nMaxFeatures[5];
	for (int i = 0; i< nFeatureScale_; i++){
		fVoxelSize[i] = fVoxelSize_[i]; // voxel size in meter
		nGridResolution[i] = vResolutions_[i]; //128,64,32
		nMaxFeatures[i] = ptr_pts_world_[i].cols;

		//in
		sDIGV._cvgmFeatureVolumeIdx[i] = feature_volume_idx_[i];

		//out
		sDIGV._voxel_centre_world[i] = ptr_pts_world_[i];
		sDIGV._feature_idx[i] = ptr_feature_idx_[i]; //for debug purpose
	} 

	cudaSafeCall(cudaMemcpyToSymbol(__fVoxelSize, fVoxelSize, sizeof(float)*nFeatureScale_));
	cudaSafeCall(cudaMemcpyToSymbol(__nGridResolution, nGridResolution, sizeof(short3)*nFeatureScale_));
	cudaSafeCall(cudaMemcpyToSymbol(__nMaxFeatures, nMaxFeatures, sizeof(int)*nFeatureScale_));
	cudaSafeCall(cudaMemcpyToSymbol(__nFeatureScale, &nFeatureScale_, sizeof(int)));

	GpuMat total; total.create(1, nFeatureScale_, CV_32SC1); total.setTo(0);
	sDIGV._total = total;

	//int* total;   cudaMallocManaged(&total, nFeatureScale_ * sizeof(int));
	//for (int i = 0; i < nFeatureScale_; i++){
	//	total[i] = 0;
	//}

    dim3 block (16, 8, 8);
    dim3 grid (1, 1, 1);
    grid.x = cv::cuda::device::divUp ( feature_volume_idx_[0].cols, block.x );
	grid.y = cv::cuda::device::divUp(feature_volume_idx_[0].cols, block.y);
	grid.z = cv::cuda::device::divUp(feature_volume_idx_[0].cols, block.z);

    kernelGetOccupiedVoxels<<<grid,block>>>( sDIGV );
	//cudaSafeCall(cudaDeviceSynchronize()); 
	cudaSafeCall(cudaGetLastError());

	Mat cpu_total;  total.download(cpu_total);
	std::vector<int> vTotal; 
	for (int i = 0; i < nFeatureScale_; i++ )
		vTotal.push_back( cpu_total.ptr<int>()[i] ); 

	//cudaFree(total);
    
	return vTotal;//is not an accurate way to count total number
}

struct SDevFillinOctree{
        
	//lvl 0+1
    PtrStepSz<int>   _cvgmFeatureVolumeIdx[5];
		
	
	__device__ __forceinline__ bool
	isEmpty(const int3& n3Grid_,  PtrStepSz<int>& cvgmFeatureVolumeIdx_, const int lvl_ ) const
	{
		return cvgmFeatureVolumeIdx_.ptr( __nGridResolution[lvl_].x * n3Grid_.y + n3Grid_.x)[n3Grid_.z] < 0;
	}

	__device__ __forceinline__ int3
	getOctreeIdx(const int3& n3Grid_ ) const
	{
		return make_int3( n3Grid_.x/2, n3Grid_.y/2, n3Grid_.z/2 );
	}

	__device__ __forceinline__ bool
	setOctree(const int3& n3Grid_,  PtrStepSz<int>& cvgmFeatureVolumeIdx_, const int lvl_ ) const
	{
		return cvgmFeatureVolumeIdx_.ptr( __nGridResolution[lvl_].x * n3Grid_.y + n3Grid_.x)[n3Grid_.z] ++;
	}
};

__global__ void kernelIntegrateFeatureIntoGlobalVolume(SDevIntegrateGlobalVolume sDIGV_, int* total_, int* replace_){
	sDIGV_.insert_features_into_volume(total_, replace_);
}

__global__ void kernelIntegrateFeatureIntoGlobalVolume2(SDevIntegrateGlobalVolume sDIGV_){
	sDIGV_.insert_features_into_volume2();
}

//nOffset_ shows how many features have been stored in global feature sets;
void cuda_integrate_features ( const pcl::device::Intr& intr_, const pcl::device::Mat33& RwInv_, const float3& Cw_, int nFeatureScale_, const short3& resolution_,
										const cv::cuda::GpuMat& cvgmVolume_, const float3& volume_size, const float fTruncDistanceM_, const float& fVoxelSize_,
										const float fFeatureVoxelSize_[], const vector<short3>& vResolution_, GpuMat* pcvgmFeatureVolumeIdx_,
										const GpuMat& cvgmKeyPointCurr_, const GpuMat& cvgmDescriptorCurr_, const GpuMat& gpu_key_array_2d_, const int nEffectiveKeyPoints_,
										vector<int>* p_vOffset_, vector<GpuMat>* pcvgmGlobalKeyPoint_, vector<GpuMat>* pcvgmGlobalDescriptor_)
{
	SDevIntegrateGlobalVolume sdIGV;
	sdIGV._cvgmVolume = cvgmVolume_;

	sdIGV._key_array_2d = gpu_key_array_2d_;
	//frame features and 3D data	
	sdIGV._cvgmDescriptorCurr = cvgmDescriptorCurr_;
    sdIGV._cvgmKeyPointCurr = cvgmKeyPointCurr_;
	//camera parameters

	float fVoxelSize[5];
	short3 nGridResolution[5];
	int nMaxFeatures[5];
	for (int i = 0; i < nFeatureScale_; i++)	{
		fVoxelSize[i] = fFeatureVoxelSize_[i]; // feature voxel size in meter
		nGridResolution[i] = vResolution_[i]; //128,64,32
		nMaxFeatures[i] = (*pcvgmGlobalKeyPoint_)[i].cols;

		sdIGV._nOffset[i] = (*p_vOffset_)[i];
		sdIGV._cvgmKeyPointGlobal[i] = (*pcvgmGlobalKeyPoint_)[i];
		sdIGV._cvgmDescriptorGlobal[i] = (*pcvgmGlobalDescriptor_)[i];
		sdIGV._cvgmFeatureVolumeIdx[i] = pcvgmFeatureVolumeIdx_[i];
	}

	cudaSafeCall(cudaMemcpyToSymbol(__fVoxelSize, fVoxelSize, sizeof(float)*nFeatureScale_));
	cudaSafeCall(cudaMemcpyToSymbol(__nGridResolution, nGridResolution, sizeof(short3)*nFeatureScale_));
	cudaSafeCall(cudaMemcpyToSymbol(__nMaxFeatures, nMaxFeatures, sizeof(int)*nFeatureScale_));
	cudaSafeCall(cudaMemcpyToSymbol(__nFeatureScale, &nFeatureScale_, sizeof(int)));
	cudaSafeCall(cudaMemcpyToSymbol(__nEffectiveKeyPoints, &nEffectiveKeyPoints_, sizeof(int)));


	int* total;   cudaMallocManaged(&total, nFeatureScale_ * sizeof(int));
	int* replace; cudaMallocManaged(&replace, nFeatureScale_ * sizeof(int));

	int block_size_x = 10;
	int block_size_y = 15;
	cudaSafeCall(cudaMemcpyToSymbol(__block_size_x, &block_size_x, sizeof(int)));
	cudaSafeCall(cudaMemcpyToSymbol(__block_size_y, &block_size_y, sizeof(int)));
	cudaSafeCall(cudaMemcpyToSymbol(__Rcurr, &RwInv_, sizeof(Mat33)));
	cudaSafeCall(cudaMemcpyToSymbol(__tcurr, &Cw_, sizeof(float3)));
	cudaSafeCall(cudaMemcpyToSymbol(__intr, &intr_, sizeof(pcl::device::Intr)));
	cudaSafeCall(cudaMemcpyToSymbol(__VOLUME, &resolution_, sizeof(short3)));
	cudaSafeCall(cudaMemcpyToSymbol(__cell_size, &fVoxelSize_, sizeof(float)));
	float inv_cell_size = 1.f / fVoxelSize_;
	cudaSafeCall(cudaMemcpyToSymbol(__inv_cell_size, &inv_cell_size, sizeof(float)));
	cudaSafeCall(cudaMemcpyToSymbol(__cols, &(gpu_key_array_2d_.cols), sizeof(int)));
	cudaSafeCall(cudaMemcpyToSymbol(__rows, &(gpu_key_array_2d_.rows), sizeof(int)));
	cudaSafeCall(cudaMemcpyToSymbol(__volume_size, &volume_size, sizeof(float3)));
	float time_step = fVoxelSize_*2.f;
	cudaSafeCall(cudaMemcpyToSymbol(__time_step, &time_step, sizeof(float)));
	dim3 block(1, 1, 1);
	dim3 grid(1, 1, 1);
	block.x = cv::cuda::device::divUp(gpu_key_array_2d_.cols, block_size_x);
	block.y = cv::cuda::device::divUp(gpu_key_array_2d_.rows, block_size_y);

	for (sdIGV._block_c = 0; sdIGV._block_c < block_size_x; sdIGV._block_c++)
		for (sdIGV._block_r = 0; sdIGV._block_r < block_size_y; sdIGV._block_r++){
			for (int i = 0; i < nFeatureScale_; i++){
				total[i] = 0;
			}

			kernelIntegrateFeatureIntoGlobalVolume << < grid, block >> >(sdIGV, total, replace);
			cudaSafeCall(cudaDeviceSynchronize());
			cudaSafeCall(cudaGetLastError());

			for (int i = 0; i < nFeatureScale_; i++){
				total[i] = min((*pcvgmGlobalKeyPoint_)[i].cols, total[i]);
				sdIGV._nOffset[i] += total[i];
			}
		}

	cudaFree(total);
	cudaFree(replace);

	for (int i = 0; i < nFeatureScale_; i++)
		(*p_vOffset_)[i] = sdIGV._nOffset[i];

	return;
}

__global__ void kernel_calc_vg_idx_n_saliency(PtrStepSz<short2> volume_, PtrStepSz<float3> pts_curr_, PtrStepSz<float3> nls_curr_,
	PtrStepSz<float> key_points_curr_, PtrStepSz<float2> distance_curr_, PtrStepSz<float2> saliency_,
	PtrStepSz<int> inliers_, PtrStepSz<int> volume_idx_, PtrStepSz<uchar> counter_){

	const int c = blockDim.x * blockIdx.x + threadIdx.x;	if (c >= __nTotalInliers) return;
	int idx = inliers_.ptr()[c]; if (idx >= __nEffectiveKeyPoints) return;
	//const int x = __float2int_rd(key_points_curr_.ptr(0)[idx]);
	//const int y = __float2int_rd(key_points_curr_.ptr(1)[idx]);
	const int lvl = ((int*)key_points_curr_.ptr(9))[idx]; if (lvl >= __nFeatureScale) return;
	float3 pt = pts_curr_.ptr()[idx];
	float3 dir = pt;
	pt = __Rcurr*pt +__tcurr;
	float3 nl = nls_curr_.ptr()[idx];
	//printf("c\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\n", c, x, y, lvl, nl.x, nl.y, nl.z, pt.x, pt.y, pt.z);
	int vx = __float2int_rd(pt.x / __fVoxelSize[lvl] ); // round to negative infinity
	int vy = __float2int_rd(pt.y / __fVoxelSize[lvl] ); // no 0.5 because of the setup of the voxels
	int vz = __float2int_rd(pt.z / __fVoxelSize[lvl] );

	if (vx >= 0 && vy >= 0 && vz >= 0 && vx < __nGridResolution[lvl].x && vy < __nGridResolution[lvl].y && vz < __nGridResolution[lvl].z && !isnan(nl.x) && !isnan(nl.y) && !isnan(nl.z)){ //if feature locates inside the feature volume
		dir *= __frsqrt_rn( dot3<float, float3>(dir, dir) ); //get visual ray
		//float ratio = (distance_curr_.ptr()[idx].y - distance_curr_.ptr()[idx].x) / distance_curr_.ptr()[idx].y;
		//key_points_curr_.ptr(6)[idx] = ratio;
		float ratio = key_points_curr_.ptr(6)[idx];
		//printf("ratio %f \n", ratio);
		saliency_.ptr()[idx] = make_float2(ratio, dot3<float, float3>(dir, nl)); // saliency is indexed the same to key_points_curr_
		volume_idx_.ptr()[c] = __nGridResolution[lvl].z * (__nGridResolution[lvl].x * vy + vx) + vz; //store volume coordinates
		counter_.ptr()[c] = 1;//counter ++
	}
	else{
		inliers_.ptr()[c] = -1;
	}
	return;
}

__global__ void kernel_non_max(PtrStepSz<float> key_points_curr_,
	 PtrStepSz<int> key_idx_, PtrStepSz<uchar> counter_, PtrStepSz<float2> saliency_, PtrStepSz<int> volume_coordinate_){
	int tid = threadIdx.x;
	const int c = blockDim.x * blockIdx.x + tid;	if (c >= __nTotal) return;
	__shared__ int V[1024], K[1024]; float2 Sa[1024];
	__shared__ uchar counter[1024];
	V[tid] = volume_coordinate_.ptr()[c];
	K[tid] = key_idx_.ptr()[c];
	Sa[tid] = saliency_.ptr()[K[tid]];
	counter[tid] = 1;
	__syncthreads();

	if (tid > 0 && tid < 1024){
		if (V[tid - 1] == V[tid] && V[tid] != V[tid + 1]){ // find the pattern: x, x, y
			int i = tid; int tid_best = i;
			while (i>0 && V[i - 1] == V[i]){
				if (Sa[i - 1].x > Sa[tid_best].x && Sa[i-1].y > Sa[tid_best].y){ //i-1 is better than i
					counter[tid_best] = 0;
					K[tid_best] = -1;
					tid_best = i - 1;
				}
				else{
					counter[i - 1] = 0;
					K[i - 1] = -1;
				}
				//printf("id\t%d\tloop\t%d\t%d\t%d\n", c, V[i - 1], V[i], V[i + 1]);
				i--;
			}
		}
	}
	else if (tid == 1024){
		if (V[tid - 1] == V[tid]){ // find the patter x, x;
			int i = tid; int tid_best = i;
			while (i>0 && V[i - 1] == V[i]){
				if (Sa[i - 1].x > Sa[tid_best].x && Sa[i - 1].y > Sa[tid_best].y){ //i-1 is better than i
					//Sa[tid_best] = make_float2(Min, Min);
					counter[tid_best] = 0;
					K[tid_best] = -1;
					tid_best = i - 1;
				}
				else{
					//Sa[i - 1] = make_float2(Min, Min);
					counter[i - 1] = 0;
					K[i - 1] = -1;
				}
				//printf("id\t%d\tloop\t%d\t%d\t%d\n", c, V[i - 1], V[i], V[i + 1]);
				i--;
			}
		}
	}
			
	__syncthreads();
	//volume_coordinate_.ptr()[c] = V[tid];
	key_idx_.ptr()[c] = K[tid];
	counter_.ptr()[c] = counter[tid];
	return;
}

void cuda_nonmax_suppress_n_integrate(const pcl::device::Intr& intr_, const pcl::device::Mat33& RwInv_, const float3& Cw_, const short3& resolution_,
											const GpuMat& volume_, const float3& volume_size, const float& fVoxelSize_,
											const float fFeatureVoxelSize_[], int nFeatureScale_, const vector<short3>& vResolution_, GpuMat* feature_volume_,
											const GpuMat& pts_curr_, const GpuMat& nls_curr_,
											GpuMat& key_points_curr_, const GpuMat& descriptors_curr_, const GpuMat& distance_curr_, const int nEffectiveKeyPoints_,
											GpuMat& gpu_inliers_, const int nTotalInliers_, GpuMat* p_volume_coordinate_, GpuMat* p_counter_,
											vector<int>* p_vOffset_, vector<GpuMat>* pcvgmGlobalKeyPoint_, vector<GpuMat>* pcvgmGlobalDescriptor_){
	//pts_curr_, nls_curr_, is 1-D list of pts and normal of current frame
	//upload all parameters onto GPU constant memory
	float fVoxelSize[5];
	short3 nGridResolution[5];
	for (int i = 0; i<nFeatureScale_; i++)	{
		fVoxelSize[i] = fFeatureVoxelSize_[i]; // feature voxel size in meter
		nGridResolution[i] = vResolution_[i]; //128,64,32
	}

	cudaSafeCall(cudaMemcpyToSymbol(__fVoxelSize, fVoxelSize, sizeof(float)*nFeatureScale_));
	cudaSafeCall(cudaMemcpyToSymbol(__nGridResolution, nGridResolution, sizeof(short3)*nFeatureScale_));
	cudaSafeCall(cudaMemcpyToSymbol(__nFeatureScale, &nFeatureScale_, sizeof(int)));
	cudaSafeCall(cudaMemcpyToSymbol(__nEffectiveKeyPoints, &nEffectiveKeyPoints_, sizeof(int)));
	cudaSafeCall(cudaMemcpyToSymbol(__nTotalInliers, &nTotalInliers_, sizeof(int)));

	cudaSafeCall(cudaMemcpyToSymbol(__intr, &intr_, sizeof(pcl::device::Intr))); 
	cudaSafeCall(cudaMemcpyToSymbol(__Rcurr, &RwInv_, sizeof(Mat33)));
	cudaSafeCall(cudaMemcpyToSymbol(__tcurr, &Cw_, sizeof(float3))); 
	cudaSafeCall(cudaMemcpyToSymbol(__VOLUME, &resolution_, sizeof(short3)));
	cudaSafeCall(cudaMemcpyToSymbol(__cell_size, &fVoxelSize_, sizeof(float))); 
	float inv_cell_size = 1.f / fVoxelSize_;
	cudaSafeCall(cudaMemcpyToSymbol(__inv_cell_size, &inv_cell_size, sizeof(float))); 
	cudaSafeCall(cudaMemcpyToSymbol(__volume_size, &volume_size, sizeof(float3))); 
	cudaSafeCall(cudaMemcpyToSymbol(__time_step, &fVoxelSize_, sizeof(float))); 

	//saliency score
	GpuMat saliency; saliency.create(1, nEffectiveKeyPoints_, CV_32FC2);
	p_volume_coordinate_->setTo(-1);
	p_counter_->setTo(uchar(0));
	
	//1. calc the voxel idx and sailency scores and filter out features outside the feature VG
	dim3 block(1024, 1);
	dim3 grid(1, 1);
	grid.x = cv::cuda::device::divUp(nTotalInliers_, block.x);//Note that gpu_inliers_ stores the idx of key points in key_points_curr
	kernel_calc_vg_idx_n_saliency <<< grid, block >>> (volume_, pts_curr_, nls_curr_, key_points_curr_, distance_curr_, saliency, gpu_inliers_, *p_volume_coordinate_, *p_counter_);
	//cudaSafeCall(cudaDeviceSynchronize());
	//cudaSafeCall(cudaGetLastError());

	int nTotal = cuda::sum(*p_counter_)[0]; if (nTotal == 0) return;//cout << nTotal << endl;//total features inside the volume

	{
		//2. sort to remove the useless features which are outside the feature volume
		thrust::device_ptr<int> K((int*)p_volume_coordinate_->data);
		thrust::device_ptr<int> V((int*)gpu_inliers_.data);
		thrust::sort_by_key(K, K + nTotalInliers_, V, thrust::greater<int>());
	}

	////3. non-max suppression by selecting the most salient one if multiple features competing for a single voxel in the volume.
	////   set the idx of removed key points as -1
	nTotal = nTotal > 1024 ? 1024 : nTotal;
	p_counter_->setTo(uchar(0));
	block.x = 1024;
	grid.x = cv::cuda::device::divUp( nTotal, block.x);
	cudaSafeCall(cudaMemcpyToSymbol(__nTotal, &nTotal, sizeof(int))); //number of features after 1.
	kernel_non_max <<< grid, block >>> (key_points_curr_, gpu_inliers_, *p_counter_, saliency, *p_volume_coordinate_);
	//cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaGetLastError());

	nTotal = cuda::sum(*p_counter_)[0]; //cout << nTotal << endl;
	{
		//2. sort to remove the useless features which are outside the feature volume
		thrust::device_ptr<int> K((int*)gpu_inliers_.data);
		thrust::device_ptr<int> V((int*)p_volume_coordinate_->data);
		thrust::sort_by_key(K, K + nTotalInliers_, V, thrust::greater<int>());
	}
	//{
		//Mat cpu_counter; p_counter_->download(cpu_counter);
		//Mat inliers; gpu_inliers_.download(inliers);
		//Mat v_indx; p_volume_coordinate_->download(v_indx);

		//for (int i = 0; i < nTotal; i++){
		//	cout << inliers.ptr<int>()[i] << "\t" << v_indx.ptr<int>()[i] << endl;
		//}
	//}
	

	SDevIntegrateGlobalVolume sdIGV;
	sdIGV._cvgmVolume = volume_;

	sdIGV._inliers = gpu_inliers_;
	sdIGV._feature_volume_coordinate = *p_volume_coordinate_;
	//frame features and 3D data	
	sdIGV._cvgmKeyPointCurr = key_points_curr_;
	sdIGV._cvgmDescriptorCurr = descriptors_curr_;
	//camera parameters

	int nMaxFeatures[5];
	for (int i = 0; i < nFeatureScale_; i++)	{
		nMaxFeatures[i] = (*pcvgmGlobalKeyPoint_)[i].cols;

		sdIGV._nOffset[i] = (*p_vOffset_)[i];
		sdIGV._cvgmKeyPointGlobal[i] = (*pcvgmGlobalKeyPoint_)[i];
		sdIGV._cvgmDescriptorGlobal[i] = (*pcvgmGlobalDescriptor_)[i];
		sdIGV._cvgmFeatureVolumeIdx[i] = feature_volume_[i];
	}

	cudaSafeCall(cudaMemcpyToSymbol(__nTotal, &nTotal, sizeof(int)));
	cudaSafeCall(cudaMemcpyToSymbol(__nMaxFeatures, nMaxFeatures, sizeof(int)*nFeatureScale_));

	GpuMat total; total.create(1, nFeatureScale_, CV_32SC1); total.setTo(0);
	sdIGV._total = total;

	block.x = 512;
	block.y = 1;
	grid.x = cv::cuda::device::divUp(nTotal, block.x);
	kernelIntegrateFeatureIntoGlobalVolume2 <<< grid, block >>>(sdIGV);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());

	Mat cpu_total; total.download(cpu_total);
	for (int i = 0; i < nFeatureScale_; i++){
		cpu_total.ptr<int>()[i] = min((*pcvgmGlobalKeyPoint_)[i].cols, cpu_total.ptr<int>()[i]);
		sdIGV._nOffset[i] += cpu_total.ptr<int>()[i];
	}

	for (int i = 0; i < nFeatureScale_; i++)
		(*p_vOffset_)[i] = sdIGV._nOffset[i];

	return;
}

__constant__ float __dist_thre;
__constant__ float __cos_visual_angle_thre;
__constant__ float __cos_normal_angle_thre;
__constant__ int   __appearance_matched;
__constant__ matrix3_cmf __Rw;
__constant__ float3 __Tw;

__global__ void kernal_refine_inliers(PtrStepSz<float3> pts_w_, PtrStepSz<float3> nls_w_, PtrStepSz<float3> pts_c_, PtrStepSz<float3> nls_c_, PtrStepSz<int> selected_inliers_, PtrStepSz<uchar> counter_){
	using namespace pcl::device;
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < __appearance_matched){
		float3 pc_ = __Rw * pts_w_.ptr()[i] + __Tw;
		float3 pc = pts_c_.ptr()[i];
		float3 d = pc_ - pc;
		float dist = __fsqrt_rn(dot3<float, float3>(d, d));
		float3 nc = __Rw * nls_w_.ptr()[i];
		float cos_angle_normal = dot3<float, float3>(nc, nls_c_.ptr()[i]);
		pc *= rsqrtf(dot3<float, float3>(pc, pc));
		pc_ *= rsqrtf(dot3<float, float3>(pc_, pc_));
		float cos_visual_angle = dot3<float, float3>(pc, pc_);
		if (dist < __dist_thre && cos_visual_angle > __cos_visual_angle_thre && cos_angle_normal > __cos_normal_angle_thre){
			counter_.ptr()[i] = uchar(1);
			selected_inliers_.ptr()[i] = i;
		}
	}

	return;
}

int cuda_refine_inliers(float fDistThre_, float CosNormalAngleThre_, float CosVisualAngleThre_, int nAppearanceMatched_, const matrix3_cmf& Rw_, const float3& Tw_,
	const GpuMat& pts_global_reloc_, const GpuMat& nls_global_reloc_, const GpuMat& pts_curr_reloc_, const GpuMat& nls_curr_reloc_,
	GpuMat* p_relined_inliers_)
{
	if (nAppearanceMatched_ <= 0) return 0;
	cudaSafeCall(cudaMemcpyToSymbol(__dist_thre, &fDistThre_, sizeof(float)));
	cudaSafeCall(cudaMemcpyToSymbol(__cos_visual_angle_thre, &CosVisualAngleThre_, sizeof(float)));
	cudaSafeCall(cudaMemcpyToSymbol(__cos_normal_angle_thre, &CosNormalAngleThre_, sizeof(float)));
	cudaSafeCall(cudaMemcpyToSymbol(__appearance_matched, &nAppearanceMatched_, sizeof(int)));
	cudaSafeCall(cudaMemcpyToSymbol(__Rw, &Rw_, sizeof(matrix3_cmf)));
	cudaSafeCall(cudaMemcpyToSymbol(__Tw, &Tw_, sizeof(float3)));
	p_relined_inliers_->setTo(-1);
	GpuMat counter; counter.create(p_relined_inliers_->size(), CV_8UC1); counter.setTo(uchar(0));

	dim3 block(1, 1);
	dim3 grid(1, 1);

	block.x = 512;
	grid.x = cv::cuda::device::divUp(nAppearanceMatched_, block.x);

	kernal_refine_inliers << < grid, block >> >(pts_global_reloc_, nls_global_reloc_, pts_curr_reloc_, nls_curr_reloc_, *p_relined_inliers_, counter);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());

	thrust::device_ptr<int> V((int*)p_relined_inliers_->data);
	thrust::sort(V, V + p_relined_inliers_->cols, thrust::greater<int>());
	int nInliers = cuda::sum(counter)[0];
	return nInliers;
}


}//device
}//pcl
