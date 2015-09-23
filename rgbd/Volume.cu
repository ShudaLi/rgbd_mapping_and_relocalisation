#define EXPORT
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#endif

#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda/common.hpp>
#include "pcl/limits.hpp"
#include "pcl/device.hpp"
#include "pcl/vector_math.hpp"
#include "Volume.cuh"

namespace btl{ namespace device
{
using namespace pcl::device;
using namespace cv::cuda;

__constant__ short3 __VOLUME;
//the organization of the volume is described in the following links
//https://docs.google.com/drawings/d/1lkw9jnNsVifIc42aDCtqnMEqc53FAlJsEIk3VBMfCN0/edit
__global__ void kernel_init(cv::cuda::PtrStepSz<short2> _volume){
	short x = threadIdx.x + blockIdx.x * blockDim.x;
	short y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < __VOLUME.x && y < __VOLUME.y)
	{
		short2 *pos = _volume.ptr(y*__VOLUME.x + x);

#pragma unroll
		for (int z = 0; z < __VOLUME.z; z++, pos++)
		{
			pack_tsdf(1.f, 0, *pos);
		}
	}
}//init_volume   

void cuda_init_tsdf(GpuMat* p_volume_, short3 resolution_)
{
	cudaSafeCall(cudaMemcpyToSymbol(__VOLUME, &resolution_, sizeof(short3))); //copy host memory to constant memory on the device.

	dim3 block(32, 32);
	dim3 grid(1, 1, 1);
	grid.x = cv::cuda::device::divUp(resolution_.x, block.x);
	grid.y = cv::cuda::device::divUp(resolution_.y, block.y);
	kernel_init << <grid, block >> >(*p_volume_);
	//cudaSafeCall ( cudaGetLastError () );
	//cudaSafeCall (cudaDeviceSynchronize ());
}//initVolume()

struct TsdfParam {
	short __WEIGHT;
	short __MAX_WEIGHT;
	short __MAX_WEIGHT_SHORT;
	float __epsilon;
	float __trunc_dist;
	float __trunc_dist_inv;
	Intr __intr;
	Mat33 __Rcurr_inv;
	float3 __tcurr;
	float3 __cell_size;
	short3 __VOLUME;
};

__constant__ TsdfParam __param;

__global__ void kernel_integrate(const cv::cuda::PtrStepSz<float> scaled_depth_, cv::cuda::PtrStepSz<short2> tsdf_)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	//the organization of the volume is described in the following links
	//https://docs.google.com/drawings/d/1lkw9jnNsVifIc42aDCtqnMEqc53FAlJsEIk3VBMfCN0/edit
	//The x index is major, then y index, z index is the last.
	//Thus, voxel (x,y,z) is stored starting at location
	//( x + y*resolution_x + z*resolution_x*resolution_y ) * (bitpix/8)

	if (x >= __param.__VOLUME.x || y >= __param.__VOLUME.y)    return;

	float v_g_x = (x + 0.5f) * __param.__cell_size.x - __param.__tcurr.x; // vw - Cw: voxel center in the world and camera center in the world
	float v_g_y = (y + 0.5f) * __param.__cell_size.y - __param.__tcurr.y;
	float v_g_z = (0 + 0.5f) * __param.__cell_size.z - __param.__tcurr.z;

	float v_g_part_norm = v_g_x * v_g_x + v_g_y * v_g_y; // get the distance from the voxel to the camera center

	float v_x = (__param.__Rcurr_inv.data[0].x * v_g_x + __param.__Rcurr_inv.data[0].y * v_g_y + __param.__Rcurr_inv.data[0].z * v_g_z) * __param.__intr.fx; //vc = Rw( vw - Cw ); voxel center in camera coordinate
	float v_y = (__param.__Rcurr_inv.data[1].x * v_g_x + __param.__Rcurr_inv.data[1].y * v_g_y + __param.__Rcurr_inv.data[1].z * v_g_z) * __param.__intr.fy; //p = K*vc; project the vc onto the image
	float v_z = (__param.__Rcurr_inv.data[2].x * v_g_x + __param.__Rcurr_inv.data[2].y * v_g_y + __param.__Rcurr_inv.data[2].z * v_g_z);

	float z_scaled = 0;

	float Rcurr_inv_0_z_scaled = __param.__Rcurr_inv.data[0].z * __param.__cell_size.z * __param.__intr.fx;
	float Rcurr_inv_1_z_scaled = __param.__Rcurr_inv.data[1].z * __param.__cell_size.z * __param.__intr.fy;

	//float tranc_dist_inv = 1.0f / __tranc_dist;

	short2* pos = tsdf_.ptr(y * __param.__VOLUME.x + x);
	//int elem_step = volume.step * __param.__VOLUME_X / sizeof(short2);// VOLUME_X * VOLUMEX 

#pragma unroll
	for (int z = 0; z < __param.__VOLUME.z;
		++z,
		v_g_z += __param.__cell_size.z,
		z_scaled += __param.__cell_size.z,
		v_x += Rcurr_inv_0_z_scaled,
		v_y += Rcurr_inv_1_z_scaled,
		pos++)
	{
		float inv_z = 1.0f / (v_z + __param.__Rcurr_inv.data[2].z * z_scaled);
		if (inv_z < 0) continue;

		// project to current cam
		float2 fcoo = {
			v_x * inv_z,
			v_y * inv_z
		};
		int2 coo = {
			__float2int_rd(fcoo.x + __param.__intr.cx + .5f),
			__float2int_rd(fcoo.y + __param.__intr.cy + .5f)
		};

		if (coo.x >= 0 && coo.y >= 0 && coo.x < scaled_depth_.cols && coo.y < scaled_depth_.rows)         //6
		{
			float Dp_scaled = scaled_depth_.ptr(coo.y)[coo.x]; //meters
			float angle_to_principle_axis = fabsf(fcoo.x) + fabsf(fcoo.y);
			float tsdf_curr = Dp_scaled - __fsqrt_rd(v_g_z * v_g_z + v_g_part_norm); //__fsqrt_rd sqrtf
			if (Dp_scaled != 0 && tsdf_curr >= -__param.__trunc_dist)//&& tsdf_curr <= __param.__trunc_dist) //meters//
			{
				tsdf_curr *= __param.__trunc_dist_inv; tsdf_curr = tsdf_curr > 1.f ? 1.f : tsdf_curr; // tsdf_curr \in [-1,1]

				//read and unpack
				short weight_prev; float tsdf_prev;
				unpack_tsdf(*pos, tsdf_prev, weight_prev);

				//constant weight
				if (false)
				{
					short weight_curr = 1;
					tsdf_curr *= weight_curr; tsdf_curr += weight_prev * tsdf_prev; tsdf_curr /= (weight_prev + weight_curr);
					weight_curr += weight_prev; weight_curr = weight_curr > __param.__MAX_WEIGHT ? __param.__MAX_WEIGHT : weight_curr;
					pack_tsdf(tsdf_curr, weight_curr, *pos);
				}
				else if (false)//narrow linear weight
				{
					short weight_curr = short(__param.__WEIGHT*(1.f - fabs(tsdf_curr)) / Dp_scaled / angle_to_principle_axis + .5f);
					tsdf_curr *= weight_curr; tsdf_curr += weight_prev * tsdf_prev; tsdf_curr /= (weight_prev + weight_curr);
					weight_curr += weight_prev; weight_curr = weight_curr > __param.__MAX_WEIGHT ? __param.__MAX_WEIGHT : weight_curr;
					pack_tsdf(tsdf_curr, weight_curr, *pos);
				}
				else //linear weight
				{
					short weight_curr = short(__param.__WEIGHT*fmax(1.f, 1.f + tsdf_curr) + .5f);
					tsdf_curr *= weight_curr; tsdf_curr += weight_prev * tsdf_prev; tsdf_curr /= (weight_prev + weight_curr);
					weight_curr += weight_prev; weight_curr = weight_curr > __param.__MAX_WEIGHT ? __param.__MAX_WEIGHT : weight_curr;
					pack_tsdf(tsdf_curr, weight_curr, *pos);
				}
			}
		}
	}       // for(int z = 0; z < VOLUME_Z; ++z)
}      // __global__

void cuda_integrate_depth(cv::cuda::GpuMat& cvgmDepthScaled_,
	const float fVoxelSize_, const float fTruncDistanceM_,
	const Mat33& Rw_, const float3& Cw_,
	const Intr& intr, const short3& resolution_,
	cv::cuda::GpuMat* pcvgmVolume_)
{
	TsdfParam param;
	param.__WEIGHT = 1 << 9; //512
	param.__MAX_WEIGHT = 1 << 14; //16384
	param.__cell_size = make_float3(fVoxelSize_, fVoxelSize_, fVoxelSize_);
	param.__epsilon = fVoxelSize_;
	param.__intr = intr;
	param.__Rcurr_inv = Rw_;
	param.__tcurr = Cw_;
	param.__trunc_dist = fTruncDistanceM_;
	param.__trunc_dist_inv = 1.f / fTruncDistanceM_;
	param.__VOLUME = resolution_;
	cudaSafeCall(cudaMemcpyToSymbol(__param, &param, sizeof(TsdfParam))); //copy host memory to constant memory on the device.

	dim3 block(16, 8);
	dim3 grid(cv::cuda::device::divUp(resolution_.x, block.x), cv::cuda::device::divUp(resolution_.y, block.y));

	kernel_integrate << <grid, block >> >(cvgmDepthScaled_, *pcvgmVolume_);


	//cudaSafeCall(cudaMemcpyToSymbol(__VOLUME, &resolution_, sizeof(short3))); //copy host memory to constant memory on the device.
	//print_volume << <grid, block >> >(*pcvgmVolume_);
	//cudaSafeCall(cudaGetLastError());
	//cudaSafeCall(cudaDeviceSynchronize());

	return;
}



}//device
}//btl