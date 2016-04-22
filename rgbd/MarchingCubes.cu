
#define EXPORT
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda/common.hpp>

#include "pcl/device.hpp"
#include "pcl/limits.hpp"
#include "pcl/device.hpp"
#include "pcl/vector_math.hpp"
#include "MarchingCubes.cuh"
#include "thrust/device_ptr.h"
#include "thrust/scan.h"
#include <iostream>
#include <algorithm>
using namespace std;

namespace pcl
{
	namespace device
	{
		//texture<int, 1, cudaReadModeElementType> edgeTex;
		texture<int, 1, cudaReadModeElementType> triTex;
		texture<int, 1, cudaReadModeElementType> numVertsTex;
	}
}

void
pcl::device::bindTextures(const GpuMat& edgeBuf, const GpuMat& triBuf, const GpuMat& numVertsBuf)
{
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
	//cudaSafeCall(cudaBindTexture(0, edgeTex, edgeBuf, desc) );
	cudaSafeCall(cudaBindTexture(0, triTex, triBuf.ptr<int>(), desc));
	cudaSafeCall(cudaBindTexture(0, numVertsTex, numVertsBuf.ptr<int>(), desc));
}
void pcl::device::unbindTextures()
{
	//cudaSafeCall( cudaUnbindTexture(edgeTex) );
	cudaSafeCall(cudaUnbindTexture(numVertsTex));
	cudaSafeCall(cudaUnbindTexture(triTex));
}

namespace pcl
{
	namespace device
	{
		struct MCParam {
			short3 __resolution;
			short3 __resolution_m_2;
			float __cell_size;
			float __inv_cell_size;
			float __half_cell_size;
			float3 __volume_size;
		};

		__constant__ MCParam __param;

		__device__ int global_count = 0;
		__device__ int output_count;
		__device__ unsigned int blocks_done = 0;

		struct CubeIndexEstimator
		{
			PtrStepSz<short2> volume;

			static __device__ __forceinline__ float isoValue() { return 0.f; }

			__device__ __forceinline__ void
				readTsdf(int x, int y, int z, float& tsdf, short& weight) const
			{
				//unpack_tsdf(volume.ptr(__param.__resolution.y * z + y)[x], tsdf, weight);
				unpack_tsdf(volume.ptr(__param.__resolution.x * y + x)[z], tsdf, weight);
			}

			__device__ __forceinline__ int
				computeCubeIndex(int x, int y, int z, float f[8]) const
			{
				short weight;
				readTsdf(x, y, z, f[0], weight); if (weight == 0) return 0;
				readTsdf(x + 1, y, z, f[1], weight); if (weight == 0) return 0;
				readTsdf(x + 1, y + 1, z, f[2], weight); if (weight == 0) return 0;
				readTsdf(x, y + 1, z, f[3], weight); if (weight == 0) return 0;
				readTsdf(x, y, z + 1, f[4], weight); if (weight == 0) return 0;
				readTsdf(x + 1, y, z + 1, f[5], weight); if (weight == 0) return 0;
				readTsdf(x + 1, y + 1, z + 1, f[6], weight); if (weight == 0) return 0;
				readTsdf(x, y + 1, z + 1, f[7], weight); if (weight == 0) return 0;

				// calculate flag indicating if each vertex is inside or outside isosurface
				int cubeindex;
				cubeindex = int(f[0] < isoValue());
				cubeindex += int(f[1] < isoValue()) * 2;
				cubeindex += int(f[2] < isoValue()) * 4;
				cubeindex += int(f[3] < isoValue()) * 8;
				cubeindex += int(f[4] < isoValue()) * 16;
				cubeindex += int(f[5] < isoValue()) * 32;
				cubeindex += int(f[6] < isoValue()) * 64;
				cubeindex += int(f[7] < isoValue()) * 128;

				return cubeindex;
			}

			__device__ __forceinline__ float
				interpolateTrilineary(const float3& Xw_) const
			{
				float a = Xw_.x * __param.__inv_cell_size;
				float b = Xw_.y * __param.__inv_cell_size;
				float c = Xw_.z * __param.__inv_cell_size;

				int3 g = make_int3(__float2int_rd(a), //round down to negative infinity
					__float2int_rd(b),
					__float2int_rd(c));//get voxel coordinate

				if (g.x<1 || g.y<1 || g.z<1 || g.x >__param.__resolution_m_2.x || g.y > __param.__resolution_m_2.y || g.z > __param.__resolution_m_2.z) return pcl::device::numeric_limits<float>::quiet_NaN();

				g.x = (Xw_.x < g.x * __param.__cell_size + __param.__half_cell_size) ? (g.x - 1.f) : g.x;
				g.y = (Xw_.y < g.y * __param.__cell_size + __param.__half_cell_size) ? (g.y - 1.f) : g.y;
				g.z = (Xw_.z < g.z * __param.__cell_size + __param.__half_cell_size) ? (g.z - 1.f) : g.z;

				a -= (g.x + 0.5f);
				b -= (g.y + 0.5f);
				c -= (g.z + 0.5f);
				int row = __param.__resolution.x * g.y + g.x;
				return  unpack_tsdf(volume.ptr(row)[g.z])     * (1 - a) * (1 - b) * (1 - c) +
					unpack_tsdf(volume.ptr(row + __param.__resolution.x)[g.z])     * (1 - a) * b       * (1 - c) +
					unpack_tsdf(volume.ptr(row + 1)[g.z])     * a       * (1 - b) * (1 - c) +
					unpack_tsdf(volume.ptr(row + __param.__resolution.x + 1)[g.z])     * a       * b       * (1 - c) +
					unpack_tsdf(volume.ptr(row)[g.z + 1]) * (1 - a) * (1 - b) * c +
					unpack_tsdf(volume.ptr(row + __param.__resolution.x)[g.z + 1]) * (1 - a) * b       * c +
					unpack_tsdf(volume.ptr(row + 1)[g.z + 1]) * a       * (1 - b) * c +
					unpack_tsdf(volume.ptr(row + __param.__resolution.x + 1)[g.z + 1]) * a       * b       * c;
			}

		};

		struct OccupiedVoxels : public CubeIndexEstimator
		{
			enum
			{
				CTA_SIZE_X = 32,
				CTA_SIZE_Y = 8,
				CTA_SIZE = CTA_SIZE_X * CTA_SIZE_Y,

				WARPS_COUNT = CTA_SIZE / Warp::WARP_SIZE
			};

			mutable int* voxels_indeces;
			mutable int* vetexes_number;
			int max_size;

			__device__ __forceinline__ void
				operator () () const
			{
				int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
				int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

				if (__all(x >= __param.__resolution.x) || __all(y >= __param.__resolution.y))
					return;

				int ftid = Block::flattenedThreadId();
				int warp_id = Warp::id();
				int lane_id = Warp::laneId();

				volatile __shared__ int warps_buffer[WARPS_COUNT];

				for (int z = 0; z < __param.__resolution.z - 1; z++)
				{
					int numVerts = 0;
					if (x + 1 < __param.__resolution.x && y + 1 < __param.__resolution.y)
					{
						float field[8];
						int cubeindex = computeCubeIndex(x, y, z, field);

						// read number of vertices from texture
						numVerts = (cubeindex == 0 || cubeindex == 255) ? 0 : tex1Dfetch(numVertsTex, cubeindex);
					}

					int total = __popc(__ballot(numVerts > 0));
					if (total == 0)
						continue;

					if (lane_id == 0)
					{
						int old = atomicAdd(&global_count, total);
						warps_buffer[warp_id] = old;
					}
					int old_global_voxels_count = warps_buffer[warp_id];

					int offs = Warp::binaryExclScan(__ballot(numVerts > 0));

					if (old_global_voxels_count + offs < max_size && numVerts > 0)
					{
						voxels_indeces[old_global_voxels_count + offs] = __param.__resolution.y * __param.__resolution.x * z + __param.__resolution.x * y + x;
						vetexes_number[old_global_voxels_count + offs] = numVerts;
					}

					bool full = old_global_voxels_count + total >= max_size;

					if (full)
						break;

				} /* for(int z = 0; z < VOLUME_Z - 1; z++) */


				/////////////////////////
				// prepare for future scans
				if (ftid == 0)
				{
					unsigned int total_blocks = gridDim.x * gridDim.y * gridDim.z;
					unsigned int value = atomicInc(&blocks_done, total_blocks);

					//last block
					if (value == total_blocks - 1)
					{
						output_count = min(max_size, global_count);
						blocks_done = 0;
						global_count = 0;
					}
				}
			} /* operator () */
		};
		__global__ void getOccupiedVoxelsKernel(const OccupiedVoxels ov) { ov(); }
	}
}

int
pcl::device::getOccupiedVoxels(const GpuMat& volume, short3 resolution_, GpuMat& occupied_voxels)
{
	MCParam param;
	param.__resolution = resolution_;
	cudaSafeCall(cudaMemcpyToSymbol(__param, &param, sizeof(MCParam))); //copy host memory to constant memory on the device.

	OccupiedVoxels ov;
	ov.volume = volume;

	//occupied_voxels.create(2, 65532, CV_32SC1);

	ov.voxels_indeces = occupied_voxels.ptr<int>(0);
	ov.vetexes_number = occupied_voxels.ptr<int>(1);
	ov.max_size = occupied_voxels.cols;

	dim3 block(OccupiedVoxels::CTA_SIZE_X, OccupiedVoxels::CTA_SIZE_Y);
	dim3 grid(cv::cuda::device::divUp(resolution_.x, block.x), cv::cuda::device::divUp(resolution_.y, block.y));

	//cudaFuncSetCacheConfig(getOccupiedVoxelsKernel, cudaFuncCachePreferL1);
	//printFuncAttrib(getOccupiedVoxelsKernel);

	getOccupiedVoxelsKernel <<<grid, block >>>(ov);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());

	int size;
	cudaSafeCall(cudaMemcpyFromSymbol(&size, output_count, sizeof(size)));
	cout << "occupied " << size << endl;
	return size;
}

int
pcl::device::computeOffsetsAndTotalVertexes(GpuMat& occupied_voxels)
{
	if (occupied_voxels.cols==0) return 0;
	thrust::device_ptr<int> beg(occupied_voxels.ptr<int>(1));
	thrust::device_ptr<int> end = beg + occupied_voxels.cols;

	thrust::device_ptr<int> out(occupied_voxels.ptr<int>(2));
	thrust::exclusive_scan(beg, end, out);

	//int lastElement, lastScanElement;
	cv::Mat lastElement, lastScanElement;
	occupied_voxels.row(1).colRange(occupied_voxels.cols - 1, occupied_voxels.cols).download(lastElement);
	occupied_voxels.row(2).colRange(occupied_voxels.cols - 1, occupied_voxels.cols).download(lastScanElement);

	//GpuMat last_elem(1, 1, CV_32SC1, occupied_voxels.ptr(1) + occupied_voxels.cols - 1);
	//GpuMat last_scan(1, 1, CV_32SC1, occupied_voxels.ptr(2) + occupied_voxels.cols - 1);
	//last_elem.download(lastElement);
	//last_scan.download(lastScanElement);

	return lastElement.ptr<int>()[0] + lastScanElement.ptr<int>()[0];
}


namespace pcl
{
	namespace device
	{
		struct TrianglesGenerator : public CubeIndexEstimator
		{
			enum { CTA_SIZE = 128, MAX_GRID_SIZE_X = 65536 };

			const int* occupied_voxels;
			const int* vertex_ofssets;
			int voxels_count;

			mutable float3 *vertices;
			mutable float3 *normals;

			__device__ __forceinline__ float3
				getNodeCoo(int x, int y, int z) const
			{
				float3 coo = make_float3(x, y, z);
				coo += 0.5f;                 //shift to volume cell center;

				coo.x *= __param.__cell_size;
				coo.y *= __param.__cell_size;
				coo.z *= __param.__cell_size;

				return coo;
			}

			__device__ __forceinline__ float3
				getNormal(int x, int y, int z) const
			{
				float3 G;
				float f1, f2; short weight;
				readTsdf(x + 1, y, z, f1, weight);
				readTsdf(x - 1, y, z, f2, weight);
				G.x = (f1 - f2)*__param.__inv_cell_size*0.5f;

				readTsdf(x, y + 1, z, f1, weight);
				readTsdf(x, y - 1, z, f2, weight);
				G.y = (f1 - f2)*__param.__inv_cell_size*0.5f;

				readTsdf(x, y, z + 1, f1, weight);
				readTsdf(x, y, z - 1, f2, weight);
				G.z = (f1 - f2)*__param.__inv_cell_size*0.5f;

				float inv_len = rsqrtf(G.x*G.x + G.y*G.y + G.z*G.z);
				return G*inv_len;
			}

			__device__ __forceinline__ float3
				vertex_interp(float3 p0, float3 p1, float f0, float f1) const
			{
				float t = (isoValue() - f0) / (f1 - f0 + 1e-15f);
				float x = p0.x + t * (p1.x - p0.x);
				float y = p0.y + t * (p1.y - p0.y);
				float z = p0.z + t * (p1.z - p0.z);
				return make_float3(x, y, z);
			}

		__device__ __forceinline__ void
		operator () () const
		{
			int tid = threadIdx.x;
			int idx = (blockIdx.y * MAX_GRID_SIZE_X + blockIdx.x) * CTA_SIZE + tid;


			if (idx >= voxels_count)
				return;

			int voxel = occupied_voxels[idx];

			int z = voxel / (__param.__resolution.x * __param.__resolution.y);
			int y = (voxel - z * __param.__resolution.x * __param.__resolution.y) / __param.__resolution.x;
			int x = (voxel - z * __param.__resolution.x * __param.__resolution.y) - y * __param.__resolution.x;

			if (x <= 0 || y <= 0 || z <= 0 || x >= __param.__resolution_m_2.x || y >= __param.__resolution_m_2.y || z >= __param.__resolution_m_2.z){
				__syncthreads();
				return;
			}

			float f[8];
			int cubeindex = computeCubeIndex(x, y, z, f);

			// calculate cell vertex positions
			float3 v[8];
			v[0] = getNodeCoo(x, y, z);
			v[1] = getNodeCoo(x + 1, y, z);
			v[2] = getNodeCoo(x + 1, y + 1, z);
			v[3] = getNodeCoo(x, y + 1, z);
			v[4] = getNodeCoo(x, y, z + 1);
			v[5] = getNodeCoo(x + 1, y, z + 1);
			v[6] = getNodeCoo(x + 1, y + 1, z + 1);
			v[7] = getNodeCoo(x, y + 1, z + 1);

			// find the vertices where the surface intersects the cube
			// use shared memory to avoid using local
			__shared__ float3 vertlist[12][CTA_SIZE];

			vertlist[0][tid] = vertex_interp(v[0], v[1], f[0], f[1]);
			vertlist[1][tid] = vertex_interp(v[1], v[2], f[1], f[2]);
			vertlist[2][tid] = vertex_interp(v[2], v[3], f[2], f[3]);
			vertlist[3][tid] = vertex_interp(v[3], v[0], f[3], f[0]);
			vertlist[4][tid] = vertex_interp(v[4], v[5], f[4], f[5]);
			vertlist[5][tid] = vertex_interp(v[5], v[6], f[5], f[6]);
			vertlist[6][tid] = vertex_interp(v[6], v[7], f[6], f[7]);
			vertlist[7][tid] = vertex_interp(v[7], v[4], f[7], f[4]);
			vertlist[8][tid] = vertex_interp(v[0], v[4], f[0], f[4]);
			vertlist[9][tid] = vertex_interp(v[1], v[5], f[1], f[5]);
			vertlist[10][tid] = vertex_interp(v[2], v[6], f[2], f[6]);
			vertlist[11][tid] = vertex_interp(v[3], v[7], f[3], f[7]);

			__syncthreads();

			// output triangle vertices
			int numVerts = tex1Dfetch(numVertsTex, cubeindex);

			for (int i = 0; i < numVerts; i += 3)
			{
				int index = vertex_ofssets[idx] + i;

				int v1 = tex1Dfetch(triTex, (cubeindex * 16) + i + 0);
				int v2 = tex1Dfetch(triTex, (cubeindex * 16) + i + 1);
				int v3 = tex1Dfetch(triTex, (cubeindex * 16) + i + 2);

				float3 vv1 = vertlist[v1][tid];
				float3 vv2 = vertlist[v2][tid];
				float3 vv3 = vertlist[v3][tid];

				float3 p1 = vv1 - vv3;
				float3 p2 = vv2 - vv3;

				float3 n;
				n.x = p1.y*p2.z - p1.z*p2.y;
				n.y = p1.z*p2.x - p1.x*p2.z;
				n.z = p1.x*p2.y - p1.y*p2.x;

				n *= rsqrtf(n.x*n.x + n.y*n.y + n.z*n.z);

				float3 avg = (vv1 + vv2 + vv3) / 3.f;
				avg = avg + n*0.01f;
				float tsdf = interpolateTrilineary(avg);
				if (tsdf >= 1.f || tsdf <= -1.f || tsdf != tsdf){
					avg.x = avg.y = avg.z = 0.f;
					store_point(vertices, index + 0, avg);
					store_point(vertices, index + 1, avg);
					store_point(vertices, index + 2, avg);

					store_point(normals, index / 3, avg);
					return;
				}
				if (tsdf > 0){
					;
				}
				else{
					n.x *= -1;
					n.y *= -1;
					n.z *= -1;
				}
				
				store_point(vertices, index + 0, vv1);
				store_point(vertices, index + 1, vv2);
				store_point(vertices, index + 2, vv3);

				store_point(normals, index / 3, n);
			}
		}
			__device__ __forceinline__ void
				store_point(float3 *ptr, int index, const float3& point) const {
				ptr[index] = make_float3(point.x, point.y, point.z);
			}
		};
		__global__ void
			trianglesGeneratorKernel(const TrianglesGenerator tg) { tg(); }
	}
}

void
pcl::device::generateTriangles(const GpuMat& volume, const GpuMat& occupied_voxels, const float3& volume_size, const short3 resolution, GpuMat& vertices, GpuMat& normals)
{
	vertices.setTo(cv::Scalar(0.f, 0.f, 0.f));
	normals.setTo(cv::Scalar(0.f, 0.f, 0.f));
	MCParam param;
	param.__volume_size = volume_size;
	param.__resolution = resolution;
	param.__resolution_m_2 = make_short3(resolution.x-2,resolution.y-2,resolution.z-2);
	param.__cell_size = volume_size.x / resolution.x;
	param.__half_cell_size = param.__cell_size*0.5f;
	param.__inv_cell_size = 1.f / param.__cell_size;

	cudaSafeCall(cudaMemcpyToSymbol(__param, &param, sizeof(MCParam))); //copy host memory to constant memory on the device.

	int device;
	cudaSafeCall(cudaGetDevice(&device));

	cudaDeviceProp prop;
	cudaSafeCall(cudaGetDeviceProperties(&prop, device));

	int block_size = prop.major < 2 ? 96 : TrianglesGenerator::CTA_SIZE; // please see TrianglesGenerator::CTA_SIZE

	typedef TrianglesGenerator Tg;
	Tg tg;

	tg.volume = volume;
	tg.occupied_voxels = occupied_voxels.ptr<int>(0);
	tg.vertex_ofssets = occupied_voxels.ptr<int>(2);
	tg.voxels_count = occupied_voxels.cols;
	tg.vertices = vertices.ptr<float3>();
	tg.normals = normals.ptr<float3>();

	int blocks_num = cv::cuda::device::divUp(tg.voxels_count, block_size);

	dim3 block(block_size);
	dim3 grid(std::min(blocks_num, int(Tg::MAX_GRID_SIZE_X)), cv::cuda::device::divUp(blocks_num, Tg::MAX_GRID_SIZE_X));

	trianglesGeneratorKernel <<< grid, block >>>(tg);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}