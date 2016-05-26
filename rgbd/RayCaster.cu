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


#define EXPORT
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#endif
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda/common.hpp>


#include "pcl/limits.hpp"
#include "pcl/device.hpp"
#include "pcl/vector_math.hpp"
#include "RayCaster.cuh"

#include "assert.h"
#define INFO
#include "OtherUtil.hpp"
#include <iostream> 
#include <limits>

namespace pcl{  namespace device  {
	using namespace std;
	using namespace cv;
	using namespace cv::cuda;
	using namespace cv::cuda::device;
	using namespace pcl::device;

struct RayCasterParam {
	Mat33 __Rcurr;
	float3 __tcurr; //Cw camera center
	float __time_step;
	float __cell_size;
	float __half_cell_size;
	float __inv_cell_size;
	float __half_inv_cell_size;
	int __rows;
	int __cols;
	short3 __VOLUME;
	short3 __VOLUME_m_1;
	short3 __VOLUME_m_2;
	Intr __intr;
	float3 __volume_max_size_m_1_cell;
};

__constant__ RayCasterParam __param;

__device__ __forceinline__ float
getMinTime(const float3& dir)
{
	float txmin = ((dir.x > 0 ? 0.f : __param.__volume_max_size_m_1_cell.x) - __param.__tcurr.x) / dir.x;
	float tymin = ((dir.y > 0 ? 0.f : __param.__volume_max_size_m_1_cell.y) - __param.__tcurr.y) / dir.y;
	float tzmin = ((dir.z > 0 ? 0.f : __param.__volume_max_size_m_1_cell.z) - __param.__tcurr.z) / dir.z;

	return fmaxf(fmaxf(txmin, tymin), tzmin);
}

__device__ __forceinline__ float
getMaxTime(const float3& dir)
{
	float txmax = ((dir.x > 0 ? __param.__volume_max_size_m_1_cell.x : 0.f) - __param.__tcurr.x) / dir.x;
	float tymax = ((dir.y > 0 ? __param.__volume_max_size_m_1_cell.y : 0.f) - __param.__tcurr.y) / dir.y;
	float tzmax = ((dir.z > 0 ? __param.__volume_max_size_m_1_cell.z : 0.f) - __param.__tcurr.z) / dir.z;

	return fminf(fminf(txmax, tymax), tzmax);
}


#define max_val 300 

struct RayCaster
{
	enum { CTA_SIZE_X = 32, CTA_SIZE_Y = 8, CTA_SIZE = CTA_SIZE_X * CTA_SIZE_Y };
	
	struct SDevPlus {
		__forceinline__ __device__ double operator () (const double &lhs, const volatile double& rhs) const {
			return (lhs + rhs);
		}
	};

	mutable PtrStepSz<double> _cvgmBuf;
	mutable PtrStepSz<double> _cvgmEn;
	int* _total;

	PtrStepSz<short2> _cvgmVolume;
	mutable PtrStepSz<float3> nmap;
	mutable PtrStepSz<float3> vmap;
	mutable PtrStepSz<uchar>  _mask;
	mutable PtrStepSz<uchar>  _mask2;

	__device__ __forceinline__ float3
	get_ray_next (int x, int y) const
	{
		//float3 ray_next;
		//ray_next.x = (x - intr.cx) / intr.fx;
		//ray_next.y = (y - intr.cy) / intr.fy;
		//ray_next.z = 1;
		//return ray_next;
		return make_float3((x - __param.__intr.cx) / __param.__intr.fx, (y - __param.__intr.cy) / __param.__intr.fy, 1.f);
	}

	__device__ __forceinline__ bool
	checkInds (const int3& g) const
	{
		return (g.x > 1 && g.y > 1 && g.z > 1 && g.x < __param.__VOLUME_m_2.x && g.y < __param.__VOLUME_m_2.y && g.z < __param.__VOLUME_m_2.z );
	}

    __device__ __forceinline__ float
    readTsdf (int x, int y, int z) const
    {
		return unpack_tsdf (_cvgmVolume.ptr (__param.__VOLUME.x * y + x)[z]);
    }
	__device__ __forceinline__ float
	readTsdf(const int3& g) const//this version consumes less registers
	{
		return unpack_tsdf(_cvgmVolume.ptr(__param.__VOLUME.x * g.y + g.x)[g.z]);
	}

    __device__ __forceinline__ int3
    getVoxel (const float3& point) const
    {
		//int vx = __float2int_rd (point.x * __param.__inv_cell_size);        // round to negative infinity
		//int vy = __float2int_rd (point.y * __param.__inv_cell_size);
		//int vz = __float2int_rd (point.z * __param.__inv_cell_size);

		return make_int3( __float2int_rd(point.x * __param.__inv_cell_size), 
						  __float2int_rd(point.y * __param.__inv_cell_size), 
						  __float2int_rd(point.z * __param.__inv_cell_size));
    }

    __device__ __forceinline__ float
    interpolateTrilinearyOrigin (const float3& dir, float time) const
    {
		return interpolateTrilineary (__param.__tcurr + dir * time);
    }

    __device__ __forceinline__ float
    interpolateTrilineary (const float3& point) const
    {
		float a = point.x * __param.__inv_cell_size;
		float b = point.y * __param.__inv_cell_size;
		float c = point.z * __param.__inv_cell_size;

		int3 g = make_int3( __float2int_rd(a), 
							__float2int_rd(b), 
							__float2int_rd(c));//get voxel coordinate

		if (g.x<1 || g.y<1 || g.z<1 || g.x >__param.__VOLUME_m_2.x || g.y > __param.__VOLUME_m_2.y || g.z > __param.__VOLUME_m_2.z) return pcl::device::numeric_limits<float>::quiet_NaN();

		g.x = (point.x < g.x * __param.__cell_size + __param.__half_cell_size) ? (g.x - 1.f) : g.x;
		g.y = (point.y < g.y * __param.__cell_size + __param.__half_cell_size) ? (g.y - 1.f) : g.y;
		g.z = (point.z < g.z * __param.__cell_size + __param.__half_cell_size) ? (g.z - 1.f) : g.z;

		a -= (g.x + 0.5f);
		b -= (g.y + 0.5f);
		c -= (g.z + 0.5f);
		int row = __param.__VOLUME.x * g.y + g.x;
		return  unpack_tsdf(_cvgmVolume.ptr(row)                         [g.z])     * (1 - a) * (1 - b) * (1 - c) +
				unpack_tsdf(_cvgmVolume.ptr(row + __param.__VOLUME.x)    [g.z])     * (1 - a) * b       * (1 - c) +
				unpack_tsdf(_cvgmVolume.ptr(row + 1)                     [g.z])     * a       * (1 - b) * (1 - c) +
				unpack_tsdf(_cvgmVolume.ptr(row + __param.__VOLUME.x + 1)[g.z])     * a       * b       * (1 - c) +
				unpack_tsdf(_cvgmVolume.ptr(row)                         [g.z + 1]) * (1 - a) * (1 - b) * c +
				unpack_tsdf(_cvgmVolume.ptr(row + __param.__VOLUME.x)    [g.z + 1]) * (1 - a) * b       * c +
				unpack_tsdf(_cvgmVolume.ptr(row + 1)                     [g.z + 1]) * a       * (1 - b) * c +
				unpack_tsdf(_cvgmVolume.ptr(row + __param.__VOLUME.x + 1)[g.z + 1]) * a       * b       * c;
    }

	__device__ __forceinline__ void ray_casting() const
	{
		const int x = threadIdx.x + blockIdx.x * blockDim.x;
		const int y = threadIdx.y + blockIdx.y * blockDim.y;
		//the organization of the volume is described in the following links
		//https://docs.google.com/drawings/d/1lkw9jnNsVifIc42aDCtqnMEqc53FAlJsEIk3VBMfCN0/edit
		//The x index is major, then y index, z index is the last.
		//Thus, voxel (x,y,z) is stored starting at location
		//( x + y*resolution_x + z*resolution_x*resolution_y ) * (bitpix/8)

		if (x >= __param.__cols || y >= __param.__rows) return;

		//float3 ray_dir = __Rcurr * get_ray_next(x, y);
		float3 ray_dir;
		//ray_dir = get_ray_next(x, y);
		ray_dir.x = (x - __param.__intr.cx) / __param.__intr.fx;
		ray_dir.y = (y - __param.__intr.cy) / __param.__intr.fy;
		ray_dir.z = 1.f;

		ray_dir = __param.__Rcurr * ray_dir;
		ray_dir *= rsqrtf(ray_dir.x * ray_dir.x + ray_dir.y * ray_dir.y + ray_dir.z * ray_dir.z); //normalize ray_dir;

		// computer time when entry and exit volume
		float time_start_volume = fmaxf(getMinTime(ray_dir), 0.f);
		//float time_start_volume = fmax(fmax(fmax(((ray_dir.x > 0 ? 0.f : __volume_size.x) - __tcurr.x) / ray_dir.x, ((ray_dir.y > 0 ? 0.f : __volume_size.y) - __tcurr.y) / ray_dir.y), ((ray_dir.z > 0 ? 0.f : __volume_size.z) - __tcurr.z) / ray_dir.z), 0.f);
		float time_exit_volume = getMaxTime(ray_dir);

		if (time_start_volume >= time_exit_volume){
			vmap.ptr(y)[x] = nmap.ptr(y)[x] = make_float3(pcl::device::numeric_limits<float>::quiet_NaN(), pcl::device::numeric_limits<float>::quiet_NaN(), pcl::device::numeric_limits<float>::quiet_NaN());
			return;
		}

		int3 g = getVoxel(__param.__tcurr + ray_dir * time_start_volume);
		if (!(g.x >= 0 && g.y >= 0 && g.z >= 0 && g.x < __param.__VOLUME.x && g.y < __param.__VOLUME.y && g.z <__param.__VOLUME.z)){
			g.x = fmaxf(0, fminf(g.x, __param.__VOLUME.x - 1));
			g.y = fmaxf(0, fminf(g.y, __param.__VOLUME.y - 1));
			g.z = fmaxf(0, fminf(g.z, __param.__VOLUME.z - 1));
		}

		//float tsdf = readTsdf(g); 
		float3 n;
		n.x/*tsdf*/ = unpack_tsdf(_cvgmVolume.ptr(__param.__VOLUME.x * g.y + g.x)[g.z]);//read tsdf at g

		//infinite loop guard
		bool is_found = false;
		for (; time_start_volume < time_exit_volume; time_start_volume += __param.__time_step)
		{
			n.y/*tsdf_prev*/ = n.x;

			g = getVoxel(__param.__tcurr + ray_dir * (time_start_volume + __param.__time_step));	if (!(g.x >= 0 && g.y >= 0 && g.z >= 0 && g.x < __param.__VOLUME.x && g.y < __param.__VOLUME.y && g.z < __param.__VOLUME.z))  break; //get next g
			n.x/*tsdf*/ = unpack_tsdf(_cvgmVolume.ptr(__param.__VOLUME.x * g.y + g.x)[g.z]); //read tsdf at g

			if (btl::device::isnan(n.y/*tsdf_prev*/) || btl::device::isnan(n.x/*tsdf*/) || n.y/*tsdf_prev*/ == n.x/*tsdf*/ || n.y/*tsdf_prev*/ < 0.f && n.x/*tsdf*/ >= 0.f)  continue;

			if (n.y/*tsdf_prev*/ >= 0.f && n.x/*tsdf*/ < 0.f)           //zero crossing
			{
				n.x/*tsdf*/ = interpolateTrilinearyOrigin(ray_dir, time_start_volume + __param.__time_step); if (btl::device::isnan(n.x/*tsdf*/)) continue; //get more accurate tsdf & tsdf_prev Ftdt
				n.y/*tsdf_prev*/ = interpolateTrilinearyOrigin(ray_dir, time_start_volume);               if (btl::device::isnan(n.y/*tsdf_prev*/)) continue; //Ft

				//float Ts = time_start_volume - time_step * tsdf_prev / (tsdf - tsdf_prev);
				float3 vertex_found = __param.__tcurr + ray_dir * (time_start_volume - __param.__time_step * n.y/*tsdf_prev*/ / (n.x/*tsdf*/ - n.y/*tsdf_prev*/));

				n.x = interpolateTrilineary(make_float3(vertex_found.x + __param.__cell_size, vertex_found.y, vertex_found.z)) - interpolateTrilineary(make_float3(vertex_found.x - __param.__cell_size, vertex_found.y, vertex_found.z));
				n.y = interpolateTrilineary(make_float3(vertex_found.x, vertex_found.y + __param.__cell_size, vertex_found.z)) - interpolateTrilineary(make_float3(vertex_found.x, vertex_found.y - __param.__cell_size, vertex_found.z));
				n.z = interpolateTrilineary(make_float3(vertex_found.x, vertex_found.y, vertex_found.z + __param.__cell_size)) - interpolateTrilineary(make_float3(vertex_found.x, vertex_found.y, vertex_found.z - __param.__cell_size));
				float inv_len = rsqrtf(n.x*n.x + n.y*n.y + n.z*n.z); if (fabsf(inv_len) > 100000.f) continue;
				n *= inv_len; if (dot3<float, float3>(n, ray_dir) >= -0.01f) continue; //exclude the points whose normal and the viewing direction are smaller than 98 degree ( 180 degree when the surface directly facing the camera )

				nmap.ptr(y)[x] = n;
				vmap.ptr(y)[x] = vertex_found;
				is_found = true;
				break;
			}//if (tsdf_prev > 0.f && tsdf < 0.f) 
		}//for (; time_start_volume < time_exit_volume; time_start_volume += time_step)
		if (!is_found){
			vmap.ptr(y)[x] = nmap.ptr(y)[x] = make_float3(pcl::device::numeric_limits<float>::quiet_NaN(), pcl::device::numeric_limits<float>::quiet_NaN(), pcl::device::numeric_limits<float>::quiet_NaN());
		}
		return;
	}//ray_casting()
	__device__ __forceinline__ void ray_casting_dda() const
	{
		const int x = threadIdx.x + blockIdx.x * blockDim.x;
		const int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= __param.__cols || y >= __param.__rows) return;

		float3 ray_dir;
		ray_dir.x = (x - __param.__intr.cx) / __param.__intr.fx;
		ray_dir.y = (y - __param.__intr.cy) / __param.__intr.fy;
		ray_dir.z = 1.f;

		ray_dir = __param.__Rcurr * ray_dir;
		ray_dir *= rsqrtf(ray_dir.x * ray_dir.x + ray_dir.y * ray_dir.y + ray_dir.z * ray_dir.z); //normalize ray_dir;
		
		// computer time when entry and exit volume
		//float tEntr = getMinTime(ray_dir);
		float tx = fabs(ray_dir.x) < __param.__cell_size ? 0.f : ((ray_dir.x > __param.__cell_size ? __param.__cell_size : __param.__volume_max_size_m_1_cell.x) - __param.__tcurr.x) / ray_dir.x;
		float ty = fabs(ray_dir.y) < __param.__cell_size ? 0.f : ((ray_dir.y > __param.__cell_size ? __param.__cell_size : __param.__volume_max_size_m_1_cell.y) - __param.__tcurr.y) / ray_dir.y;
		tx = tx > ty ? tx : ty;
		ty /*tz*/= fabs(ray_dir.z) < __param.__cell_size ? 0.f : ((ray_dir.z > __param.__cell_size ? __param.__cell_size : __param.__volume_max_size_m_1_cell.z) - __param.__tcurr.z) / ray_dir.z;
		tx = tx > ty ? tx : ty;
		float tEntr = tx > 0.f ? tx : 0.f; 

		//float tExit = getMaxTime(ray_dir);
		tx = fabs(ray_dir.x) < __param.__cell_size ? __param.__volume_max_size_m_1_cell.x : ((ray_dir.x > __param.__cell_size ? __param.__volume_max_size_m_1_cell.x : __param.__cell_size) - __param.__tcurr.x) / ray_dir.x; 
		ty = fabs(ray_dir.y) < __param.__cell_size ? __param.__volume_max_size_m_1_cell.y : ((ray_dir.y > __param.__cell_size ? __param.__volume_max_size_m_1_cell.y : __param.__cell_size) - __param.__tcurr.y) / ray_dir.y;
		tx = tx < ty ? tx : ty;
		ty = fabs(ray_dir.z) < __param.__cell_size ? __param.__volume_max_size_m_1_cell.z : ((ray_dir.z > __param.__cell_size ? __param.__volume_max_size_m_1_cell.z : __param.__cell_size) - __param.__tcurr.z) / ray_dir.z;
		tx = tx < ty ? tx : ty;
		//printf("[%d, %d]:\t Enter %f, Exit %f [%f, %f, %f]\n ", x, y, tEntr, tx, ray_dir.x, ray_dir.y, ray_dir.z );

		if (tEntr >= tx/*tExit*/){
			return;
		}

		int StepX = ray_dir.x > 0 ? 1 : -1;
		int StepY = ray_dir.y > 0 ? 1 : -1;
		int StepZ = ray_dir.z > 0 ? 1 : -1;

		float tMaxX, tMaxY, tMaxZ;
		tMaxX = ray_dir.x*tEntr + __param.__tcurr.x; //for now tMaxX,Y,Z stores the coordinate of enter point
		tMaxY = ray_dir.y*tEntr + __param.__tcurr.y;
		tMaxZ = ray_dir.z*tEntr + __param.__tcurr.z;
		int X, Y, Z;
		X = __float2int_rd(tMaxX*__param.__inv_cell_size);	X = X < __param.__VOLUME_m_1.x ? X : __param.__VOLUME_m_2.x; X = X > 1 ? X : 1;
		Y = __float2int_rd(tMaxY*__param.__inv_cell_size);	Y = Y < __param.__VOLUME_m_1.y ? Y : __param.__VOLUME_m_2.y; Y = Y > 1 ? Y : 1;
		Z = __float2int_rd(tMaxZ*__param.__inv_cell_size);	Z = Z < __param.__VOLUME_m_1.z ? Z : __param.__VOLUME_m_2.z; Z = Z > 1 ? Z : 1;

		tMaxX =  (fabs(ray_dir.x) < __param.__half_cell_size) ? max_val  : ((X + StepX) * __param.__cell_size - tMaxX) / ray_dir.x + tEntr; 
		tMaxY =  (fabs(ray_dir.y) < __param.__half_cell_size) ? max_val  : ((Y + StepY) * __param.__cell_size - tMaxY) / ray_dir.y + tEntr; 
		tMaxZ =  (fabs(ray_dir.z) < __param.__half_cell_size) ? max_val  : ((Z + StepZ) * __param.__cell_size - tMaxZ) / ray_dir.z + tEntr; 
		//printf("[%d, %d]:\t Max [%f, %f, %f]\n", x, y, tMaxX, tMaxY, tMaxZ);
		float tDeltaX = StepX * __param.__cell_size / ray_dir.x; 
		float tDeltaY = StepY * __param.__cell_size / ray_dir.y; 
		float tDeltaZ = StepZ * __param.__cell_size / ray_dir.z; 
		//printf("[%d, %d]:\t ray [%f, %f, %f] Max [%f, %f, %f] Delta [%f, %f, %f] XYZ [%d, %d, %d]\n ", x, y, ray_dir.x, ray_dir.y, ray_dir.z, tMaxX, tMaxY, tMaxZ, tDeltaX, tDeltaY, tDeltaZ, X, Y, Z);

		float inv_len;
		float3 n;
		tx /*tsdf*/ = unpack_tsdf( _cvgmVolume.ptr(__param.__VOLUME.x * Y + X)[Z] );//read tsdf at g
		//infinite loop guard
		while ( X > 0 && Y > 0 && Z > 0 && X < __param.__VOLUME_m_1.x && Y < __param.__VOLUME_m_1.y && Z < __param.__VOLUME_m_1.z )//(tMaxX < tExit || tMaxY < tExit || tMaxZ < tExit)
		{
			ty /*tsdf_prev*/ = tx /*tsdf*/;
			if (tMaxX < tMaxY){
				if (tMaxX < tMaxZ){
					X += StepX;
					inv_len = tMaxX;
					tMaxX += tDeltaX;
				}
				else {
					Z += StepZ;
					inv_len = tMaxZ;
					tMaxZ += tDeltaZ;
				}
			}
			else {
				if (tMaxY < tMaxZ){
					Y += StepY;
					inv_len = tMaxY;
					tMaxY += tDeltaY;
				}
				else {
					Z += StepZ;
					inv_len = tMaxZ;
					tMaxZ += tDeltaZ;
				}
			}

			tx/*tsdf*/ = unpack_tsdf(_cvgmVolume.ptr(__param.__VOLUME.x * Y + X)[Z]); //read tsdf 

			if ( (ty/*tsdf_prev*/ >= 0.f && tx/*tsdf*/ < 0.f) || (ty/*tsdf_prev*/ == 0.f && tx/*tsdf*/ == 0.f))           //zero crossing
			{
				float3 vertex_found = __param.__tcurr + ray_dir * inv_len;
				//printf("[%d, %d]:\t [%f, %f, %f]\n ", x, y, vertex_found.x, vertex_found.y, vertex_found.z);
				n.x = interpolateTrilineary(make_float3(vertex_found.x + __param.__half_cell_size, vertex_found.y, vertex_found.z)) - interpolateTrilineary(make_float3(vertex_found.x - __param.__half_cell_size, vertex_found.y, vertex_found.z));
				n.y = interpolateTrilineary(make_float3(vertex_found.x, vertex_found.y + __param.__half_cell_size, vertex_found.z)) - interpolateTrilineary(make_float3(vertex_found.x, vertex_found.y - __param.__half_cell_size, vertex_found.z));
				n.z = interpolateTrilineary(make_float3(vertex_found.x, vertex_found.y, vertex_found.z + __param.__half_cell_size)) - interpolateTrilineary(make_float3(vertex_found.x, vertex_found.y, vertex_found.z - __param.__half_cell_size));
				nmap.ptr(y)[x] = n * rsqrtf(n.x*n.x + n.y*n.y + n.z*n.z); //if (fabsf(inv_len) > 100000.f) continue;
				vmap.ptr(y)[x] = vertex_found;
				break;
			}//if (tsdf_prev > 0.f && tsdf < 0.f) 
		}//while 

		return;
	}//ray_casting2()

	__device__ __forceinline__ void	calc_mask_tsdf(){
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x < __param.__cols && y < __param.__rows) { // we cant use x >= cols || y >= rows because __syncthreads
			_mask.ptr(y)[x] = uchar(0);
			float3 Vw = vmap.ptr(y)[x];
			if (!btl::device::isnan(Vw.x) && !btl::device::isnan(Vw.y) && !btl::device::isnan(Vw.z)){
				float dist = norm<float, float3>(Vw);
				Vw = __param.__Rcurr*Vw + __param.__tcurr;//Rcurr = Rw'; tcurr = cw;  i.e. transform V from camera to world
				//printf("[%d, %d]:\t [%f, %f, %f]\n ", x, y, Vw.x, Vw.y, Vw.z);
				int3 g = getVoxel(Vw);
				if (g.x > 1 && g.y > 1 && g.z > 1 && g.x < __param.__VOLUME_m_2.x && g.y < __param.__VOLUME_m_2.y && g.z < __param.__VOLUME_m_2.z )
				{
					float TSDF = interpolateTrilineary(Vw); //readTsdf(Vw.x, Vw.y, Vw.z);
					if (!btl::device::isnan(TSDF) && fabs(TSDF) < .49f){//
						//printf("[%d, %d]:\t %f [%d, %d, %d] [%f, %f, %f]\n ", x, y, TSDF, g.x, g.y, g.z, Vw.x, Vw.y, Vw.z);
						//printf("[%f, %f, %f; %f, %f, %f;%f, %f, %f;]\n", __param.__Rcurr.data[0].x, __param.__Rcurr.data[0].y, __param.__Rcurr.data[0].z, __param.__Rcurr.data[1].x, __param.__Rcurr.data[1].y, __param.__Rcurr.data[1].z, __param.__Rcurr.data[2].x, __param.__Rcurr.data[2].y, __param.__Rcurr.data[2].z);
						_mask.ptr(y)[x] = uchar(255);
					}
				}
				else{
					_mask.ptr(y)[x] = uchar(128);
				}
			}
		}
		return;
	}//calc_mask_tsdf()

	__device__ __forceinline__ void	calc_energy_tsdf(){
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		float fE = 0.f; // it has to be there for all threads, otherwise, some thread will add un-initialized fE into total energy. 
		if (x < __param.__cols && y < __param.__rows) { // we cant use x >= cols || y >= rows because __syncthreads
			float3 Vw = vmap.ptr(y)[x];
			if (!btl::device::isnan(Vw.x) && !btl::device::isnan(Vw.y) && !btl::device::isnan(Vw.z)){
				Vw = __param.__Rcurr*Vw + __param.__tcurr;//Rcurr = Rw'; tcurr = cw;  i.e. transform V from camera to world
				if ( Vw.x < __param.__cell_size || Vw.y < __param.__cell_size || Vw.z < __param.__cell_size || Vw.x > __param.__volume_max_size_m_1_cell.x || Vw.y > __param.__volume_max_size_m_1_cell.y || Vw.z > __param.__volume_max_size_m_1_cell.z ){
					_mask2.ptr(y)[x] = uchar(1);
				}
				else{
					float TSDF = interpolateTrilineary(Vw); //readTsdf(Vw.x, Vw.y, Vw.z);
					if ( TSDF < .99f && TSDF > -.99f){
						fE = fabs(TSDF);
						//printf("[%d, %d]:\t %f\n", x, y, fE);
						_mask.ptr(y)[x] = uchar(1);
					}
				}
			}
			else{
				_mask2.ptr(y)[x] = uchar(1);
			}
		}

		__shared__ double smem[CTA_SIZE]; // CTA_SIZE is 32*8 == the number of threads in the block
		int nThrID = Block::flattenedThreadId();
		smem[nThrID] = double(fE); //fill all the shared memory
		__syncthreads();

		Block::reduce<CTA_SIZE>(smem, SDevPlus());
		if (nThrID == 0) _cvgmEn.ptr()[blockIdx.x + gridDim.x * blockIdx.y] = smem[0];
		return;
	}//calc_energy_tsdf()

	__device__ __forceinline__ void	accumulate_alignment_energy_tsdf()	{
		const int x = threadIdx.x + blockIdx.x * blockDim.x;
		const int y = threadIdx.y + blockIdx.y * blockDim.y;

		float row[7] = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };

		if (x < __param.__cols && y < __param.__rows) {
			float3 Vw = vmap.ptr(y)[x];
			if (!btl::device::isnan(Vw.x) && !btl::device::isnan(Vw.y) && !btl::device::isnan(Vw.z)){
				Vw = __param.__Rcurr*Vw + __param.__tcurr;//Rcurr = Rw'; tcurr = cw;  i.e. transform V from camera to world
				//get integer volume grid coordinate g
				row[3] = Vw.x * __param.__inv_cell_size;
				row[4] = Vw.y * __param.__inv_cell_size;
				row[5] = Vw.z * __param.__inv_cell_size;

				if (row[3] > 1.f && row[4] > 1.f && row[5] > 1.f && row[3] < __param.__VOLUME_m_2.x && row[4] < __param.__VOLUME_m_2.y && row[5] < __param.__VOLUME_m_2.z ) {
					row[6] = interpolateTrilineary(Vw); //TSDF

					if (row[6] < .99f && row[6]>.99f){
						//dTSDF_dxyz
						row[3] = (interpolateTrilineary(make_float3(Vw.x - __param.__half_cell_size, Vw.y, Vw.z)) - interpolateTrilineary(make_float3(Vw.x + __param.__half_cell_size, Vw.y, Vw.z))) * __param.__inv_cell_size;
						row[4] = (interpolateTrilineary(make_float3(Vw.x, Vw.y - __param.__half_cell_size, Vw.z)) - interpolateTrilineary(make_float3(Vw.x, Vw.y + __param.__half_cell_size, Vw.z))) * __param.__inv_cell_size;
						row[5] = (interpolateTrilineary(make_float3(Vw.x, Vw.y, Vw.z - __param.__half_cell_size)) - interpolateTrilineary(make_float3(Vw.x, Vw.y, Vw.z + __param.__half_cell_size))) * __param.__inv_cell_size;

						row[0] = Vw.y * row[5] - Vw.z * row[4]; //v1.y * v2.z - v1.z * v2.y
						row[1] = Vw.z * row[3] - Vw.x * row[5]; //v1.z * v2.x - v1.x * v2.z
						row[2] = Vw.x * row[4] - Vw.y * row[3]; //v1.x * v2.y - v1.y * v2.x //cross(Vw, dTSDF_dxyz);
					}//if (!isnan(row[6]) && fabsf(row[6]) < .99f){
					else{
						row[3] = row[4] = row[5] = row[6] = 0.f;
					}
				}//if (row[3] > 1 && row[4] > 1 && row[5] > 1 && row[3] < __param.__VOLUME.x - 2 && row[4] < __param.__VOLUME.y - 2 && row[5] < __param.__VOLUME.z - 2) {
				else{
					row[3] = row[4] = row[5] = row[6] = 0.f;
				}
			}//if (!isnan(V.x) && !isnan(V.y) && !isnan(V.z))
		}//if (x < __param.__cols && y < __param.__rows) {


		__shared__ double smem[CTA_SIZE]; // CTA_SIZE is 32*8 == the number of threads in the block
		int nThrID = Block::flattenedThreadId();

		int nShift = 0;
		for (int i = 0; i < 6; ++i){ //_nRows
	#pragma unroll
			for (int j = i; j < 7; ++j){ // _nCols + b
				__syncthreads();
				smem[nThrID] = row[i] * row[j]; //fill all the shared memory
				__syncthreads();

				Block::reduce<CTA_SIZE>(smem, SDevPlus()); //reduce to thread 0;
				if (nThrID == 0) _cvgmBuf.ptr(nShift++)[blockIdx.x + gridDim.x * blockIdx.y] = smem[0]; //nShift < 27 = 21 + 6, upper triangle of 6x6
			}//for
		}//for
		return;
	}//icp_volume_2_frm()
};//RayCaster

__global__ void rayCastKernelDDA (const RayCaster rc) {
	rc.ray_casting_dda();
}

__global__ void rayCastKernel(const RayCaster rc) {
	rc.ray_casting();
}

//get VMap and NMap in world
void cuda_ray_cast ( const pcl::device::Intr& intr, const pcl::device::Mat33& RwInv_, const float3& Cw_, bool bFineCast_,
               const float fTruncDistanceM_, const float& fVoxelSize_, const short3& resolution_, const float3& dimensions_,
               const GpuMat& cvgmVolume_, GpuMat* pVMap_, GpuMat* pNMap_)
{
	RayCasterParam param;
	param.__cols = pVMap_->cols;
	param.__rows = pVMap_->rows;
	param.__cell_size = fVoxelSize_;
	param.__half_cell_size = fVoxelSize_ *0.5f;
	param.__VOLUME = resolution_;
	param.__VOLUME_m_1 = make_short3(resolution_.x - 1, resolution_.y - 1, resolution_.z - 1);
	param.__VOLUME_m_2 = make_short3(resolution_.x - 2, resolution_.y - 2, resolution_.z - 2);
	param.__Rcurr = RwInv_;
	param.__tcurr = Cw_; 
	param.__intr = intr;
	param.__inv_cell_size = 1.f / fVoxelSize_;
	//param.__volume_max_size_m_1_cell = make_float3(fVolumeSizeM_ - fVoxelSize_, fVolumeSizeM_ - fVoxelSize_, fVolumeSizeM_ - fVoxelSize_);
	param.__volume_max_size_m_1_cell = dimensions_;

	//if (!bFineCast_)
	//	param.__time_step = fVoxelSize_ * 8.f; //fTruncDistanceM_ * 0.4f; 
	//else
	param.__time_step = fVoxelSize_ * 4.f;
	cudaSafeCall(cudaMemcpyToSymbol(__param, &param, sizeof(RayCasterParam))); //copy host memory to constant memory on the device.

	RayCaster rc;
	pVMap_->setTo(std::numeric_limits<float>::quiet_NaN());
	pNMap_->setTo(std::numeric_limits<float>::quiet_NaN());
	rc._cvgmVolume = cvgmVolume_;
	rc.vmap = *pVMap_;
	rc.nmap = *pNMap_;

	dim3 block (RayCaster::CTA_SIZE_X, RayCaster::CTA_SIZE_Y);
	dim3 grid (divUp (pVMap_->cols, block.x), device::divUp (pVMap_->rows, block.y));

	rayCastKernel<<<grid, block>>>(rc);
	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaGetLastError());

	return;
}

template<class float_type>
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
		const float_type *beg = _cvgmBuf.ptr(blockIdx.x); // 27 * # of blocks in previous kernel launch
		const float_type *end = beg + length;

		int tid = threadIdx.x;

		float_type sum = 0.f;
		for (const float_type *t = beg + tid; t < end; t += STRIDE)
			sum += *t;

		__shared__ float_type smem[CTA_SIZE];

		smem[tid] = sum;
		__syncthreads();

		Block::reduce<CTA_SIZE>(smem, RayCaster::SDevPlus());

		if (tid == 0) pOutput[blockIdx.x] = smem[0];
	}//operator ()
};//STranformReduction

__global__ void kernel_icp_volume_2_frm_tsdf(RayCaster rc) {
	rc.accumulate_alignment_energy_tsdf();
}

__global__ void kernel_calc_energy_tsdf(RayCaster rc) {
	rc.calc_energy_tsdf();
}

__global__ void kernel_calc_mask_tsdf(RayCaster rc) {
	rc.calc_mask_tsdf();
}

__global__ void kernelTransformEstimator(STranformReduction<double> sTR) {
	sTR();
}


}// device
}// pcl


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

