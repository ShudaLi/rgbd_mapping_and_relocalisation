#ifndef BTL_CUDA_MC_HEADER
#define BTL_CUDA_MC_HEADER
#include "DllExportDef.h"

namespace pcl { namespace device {
	using namespace cv::cuda;
	using namespace pcl::device;
	// Marching cubes implementation

	/** \brief Binds marching cubes tables to texture references */
	void
		DLL_EXPORT bindTextures(const GpuMat& edgeBuf, const GpuMat& triBuf, const GpuMat& numVertsBuf);

	/** \brief Unbinds */
	void
		DLL_EXPORT unbindTextures();

	/** \brief Scans tsdf volume and retrieves occuped voxes
	* \param[in] volume tsdf volume
	* \param[out] occupied_voxels buffer for occuped voxels. The function fulfills first row with voxel ids and second row with number of vertextes.
	* \return number of voxels in the buffer
	*/
	int
		DLL_EXPORT getOccupiedVoxels(const GpuMat& volume, short3 resolution_, GpuMat& occupied_voxels);

	/** \brief Computes total number of vertexes for all voxels and offsets of vertexes in final triangle array
	* \param[out] occupied_voxels buffer with occuped voxels. The function fulfills 3nd only with offsets
	* \return total number of vertexes
	*/
	int
		DLL_EXPORT computeOffsetsAndTotalVertexes(GpuMat& occupied_voxels);

	/** \brief Generates final triangle array
	* \param[in] volume tsdf volume
	* \param[in] occupied_voxels occuped voxel ids (first row), number of vertexes(second row), offsets(third row).
	* \param[in] volume_size volume size in meters
	* \param[out] output triangle array
	*/
	void
		DLL_EXPORT generateTriangles(const GpuMat& volume, const GpuMat& occupied_voxels, const float3& volume_size, const short3 resolution, GpuMat& vertices, GpuMat& normals); //DeviceArray<float4>

}//device
}//btl

#endif