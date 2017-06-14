//Copyright(c) 2016 Shuda Li[lishuda1980@gmail.com]
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
#define _USE_MATH_DEFINES
#define INFO
#define DEFAULT_TRIANGLES_BUFFER_SIZE 
//gl
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <numeric>
#include <experimental/filesystem>
//nifti
#include <nifti1_io.h>   /* directly include I/O library functions */
//stl
#ifdef __gnu_linux__
#include <sys/types.h>
#include <sys/stat.h>
#elif _WIN32
#include <direct.h>
#else 
#error "OS not supported!"
#endif
#include <vector>
#include <fstream>
#include <list>
#include <limits>
#include <cstring>
//opencv
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda/common.hpp>
//eigen
#include <Eigen/Core>
#include <sophus/se3.hpp>
//nifti
#include <nifti1_io.h>
#define MIN_HEADER_SIZE 348
#define NII_HEADER_SIZE 352

//self
#include "OtherUtil.hpp"
#include "Converters.hpp"
#include "EigenUtil.hpp"
#include "GLUtil.hpp"
#include <map>
#include "Camera.h"
#include "Kinect.h"
#include "RGBDFrame.h"
#include "pcl/internal.h"
#include "CubicGrids.h"
#include "CudaLib.cuh"
#include "Volume.cuh"
#include "RayCaster.cuh"
#include "FeatureVolume.cuh"
#include "ICPFrame2Frame.cuh"
#include "MarchingCubes.cuh"

namespace btl{ namespace geometry
{

using namespace cv;
using namespace cv::cuda;
using namespace btl::kinect;
using namespace std;
using namespace pcl::device;

CCubicGrids::CCubicGrids(ushort usVolumeResolutionX_, ushort usVolumeResolutionY_, ushort usVolumeResolutionZ_, float fVolumeSizeMX_, int nFeatureScale_, const string& featureName_)
	:_VolumeResolution(make_short3(usVolumeResolutionX_, usVolumeResolutionY_, usVolumeResolutionZ_)),  _nFeatureScale(nFeatureScale_)
{
	_uVolumeLevelXY = _VolumeResolution.x*_VolumeResolution.y;
	_uVolumeTotal = _uVolumeLevelXY*_VolumeResolution.z;
	_fVoxelSizeM = fVolumeSizeMX_ / _VolumeResolution.x;
	_fTruncateDistanceM = _fVoxelSizeM*10;
	_fVolumeSizeM = make_float3(fVolumeSizeMX_, _fVoxelSizeM * _VolumeResolution.y, _fVoxelSizeM * _VolumeResolution.z);
	
	_quadratic = gluNewQuadric();                // Create A Pointer To The Quadric Object ( NEW )
	// Can also use GLU_NONE, GLU_FLAT
	gluQuadricNormals(_quadratic, GLU_SMOOTH); // Create Smooth Normals
	gluQuadricTexture(_quadratic, GL_TRUE);   // Create Texture Coords ( NEW )

	setFeatureName(featureName_);
	_gpu_YXxZ_tsdf.create(_uVolumeLevelXY, _VolumeResolution.z, CV_16SC2);//y*x rows,z cols
	btl::device::cuda_init_tsdf(&_gpu_YXxZ_tsdf, _VolumeResolution);

	_gpu_feature_volume_coordinate.create(1, 1500, CV_32SC1);
	_gpu_counter.create(1, 1500, CV_8UC1);
	//allocate volumetric grids (VG)
	_vFeatureVoxelSizeM.clear();
	_vFeatureResolution.clear();
	for (int i = 0; i < _nFeatureScale; i++)	{
		_vFeatureResolution.push_back(make_short3(_VolumeResolution.x >> (2 + i), _VolumeResolution.y >> (2 + i), _VolumeResolution.z >> (2 + i)));
		_gpu_YXxZ_vg_idx[i].create(_vFeatureResolution[i].x * _vFeatureResolution[i].y, _vFeatureResolution[i].z, CV_32SC1);
		_gpu_YXxZ_vg_idx[i].setTo(-1);

		int s1 = 1 << i * 2;
		_gpu_pts[i].create(1, 32000 / s1, CV_32FC3);
		_gpu_feature_idx[i].create(1, 32000 / s1, CV_32SC1);
		int s2 = 4 << i;
		_fFeatureVoxelSizeM[i] = _fVoxelSizeM*s2;
		_vFeatureVoxelSizeM.push_back(_fVoxelSizeM*s2);
	}
	//allocate key points
	int descriptor_bytes = 64;

	_nKeyPointDimension = 12;
	_vTotalGlobal.clear();

	_gpu_global_descriptors.resize(_nFeatureScale);
	_gpu_global_3d_key_points.resize(_nFeatureScale);
	_vTotalGlobal.resize(_nFeatureScale);
	for (int i = 0; i < _nFeatureScale; i++){
		int s = 1 << i * 2;
		int length = 320000 / s;
		_gpu_global_3d_key_points[i].create(_nKeyPointDimension, length, CV_32FC1); _gpu_global_3d_key_points[i].setTo(Scalar::all(numeric_limits<float>::quiet_NaN()));
		_gpu_global_descriptors[i].create(length, descriptor_bytes, CV_8UC1);
		_vTotalGlobal[i] = 0;//At the first frame, no features will be inserted into the global feature set.
	}
}
CCubicGrids::~CCubicGrids(void)
{
	_gpu_YXxZ_tsdf.release();
	for(int i=0; i<3; i++){
		_gpu_YXxZ_vg_idx[i].release();
	}
	
	//releaseVBOPBO();
	//if(_pGL) _pGL->releaseVBO(_uVBO,_pResourceVBO);
}

void CCubicGrids::releaseVBOPBO()
{
	//release VBO
	cudaSafeCall( cudaGraphicsUnregisterResource( _pResourceVBO ) );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );
	glDeleteBuffers( 1, &_uVBO );

	//release PBO
	// unregister this buffer object with CUDA
	//http://rickarkin.blogspot.co.uk/2012/03/use-pbo-to-share-buffer-between-cuda.html
	//cudaSafeCall( cudaGraphicsUnregisterResource( _pResourcePBO ) );
	//glDeleteBuffers(1, &_uPBO);
}

void CCubicGrids::setFeatureName(const string& FeatureName_){
	if (!FeatureName_.compare("BRISK")){
		_nFeatureName = btl::BRISKC;
	}
	else{
		_nFeatureName = btl::BRISKC;
		cout << "Unsupported features" << endl;
	}

}

void CCubicGrids::integrateDepth(const CRGBDFrame& cFrame_, const Intr& intr_){
	//Note: the point cloud int cFrame_ must be transformed into world before calling it, i.e. it integrate a VMap NMap in world to the volume in world
	Eigen::Matrix3f eimfRw = cFrame_._Rw.transpose();//device cast do the transpose implicitly because eimcmRwCur is col major by default.
	pcl::device::Mat33& devRw = pcl::device::device_cast<pcl::device::Mat33> (eimfRw);
	Eigen::Vector3f Cw = - cFrame_._Rw.transpose() *cFrame_._Tw ; //get camera center in world coordinate
	float3& devCw = pcl::device::device_cast<float3> (Cw);

	btl::device::cuda_integrate_depth(*cFrame_._acvgmShrPtrPyrDepths[0],
									_fVoxelSizeM,_fTruncateDistanceM, 
									devRw, devCw,//camera parameters,
									intr_, _VolumeResolution,
									&_gpu_YXxZ_tsdf);

	return;
}

void CCubicGrids::integrateFeatures( const pcl::device::Intr& cCam_, const Matrix3f& Rw_, const Vector3f& Tw_, int nFeatureScale_,
												  GpuMat& pts_curr_, GpuMat& nls_curr_,
												  GpuMat& gpu_key_points_curr_, const GpuMat& gpu_descriptor_curr_, const GpuMat& gpu_distance_curr_, const int nEffectiveKeypoints_,
												  GpuMat& gpu_inliers_, const int nTotalInliers_){

	using namespace btl::device;

	//Note: the point cloud int cFrame_ must be transformed into world before calling it, i.e. it integrate a VMap NMap in world to the volume in world
	//get VMap and NMap in world
	Matrix3f tmp = Rw_;
	pcl::device::Mat33& devRwCurTrans = pcl::device::device_cast<pcl::device::Mat33> ( tmp );	//device cast do the transpose implicitly because eimcmRwCur is col major by default.
	//Cw = -Rw'*Tw
	Eigen::Vector3f eivCwCur = - Rw_.transpose() * Tw_ ;
	float3& devCwCur = pcl::device::device_cast<float3> (eivCwCur);
	
	//locate the volume coordinate for each feature, if the feature falls outside the volume, just remove it.
	cuda_nonmax_suppress_n_integrate ( _intrinsics, devRwCurTrans, devCwCur, _VolumeResolution,
								_gpu_YXxZ_tsdf, _fVolumeSizeM, _fVoxelSizeM,
								_fFeatureVoxelSizeM, _nFeatureScale, _vFeatureResolution, _gpu_YXxZ_vg_idx,
								pts_curr_, nls_curr_, gpu_key_points_curr_, gpu_descriptor_curr_, gpu_distance_curr_, nEffectiveKeypoints_,
								gpu_inliers_, nTotalInliers_, &_gpu_feature_volume_coordinate, &_gpu_counter,
								&_vTotalGlobal, &_gpu_global_3d_key_points, &_gpu_global_descriptors);

	return;
}

void CCubicGrids::rayCast(CRGBDFrame* pVirtualFrame_, bool bForDisplay_, bool bFineCast_, int lvl /*= 0*/  ) const {
	//get VMap and NMap in world
	pcl::device::Mat33& devRwCurTrans = pcl::device::device_cast<pcl::device::Mat33> (pVirtualFrame_->_Rw);	//device cast do the transpose implicitly because eimcmRwCur is col major by default.
	//Cw = -Rw'*Tw
	Eigen::Vector3f CwCur = - pVirtualFrame_->_Rw.transpose() * pVirtualFrame_->_Tw ;
	float3& devCwCur = pcl::device::device_cast<float3> (CwCur);
	
	pcl::device::cuda_ray_cast(_intrinsics(lvl), devRwCurTrans, devCwCur, bFineCast_, _fTruncateDistanceM, _fVoxelSizeM, 
						 _VolumeResolution, _fVolumeSizeM,
						 _gpu_YXxZ_tsdf,
						 &*pVirtualFrame_->_acvgmShrPtrPyrPts[lvl],&*pVirtualFrame_->_acvgmShrPtrPyrNls[lvl] );
	
	//if (bForDisplay_)
	{
		//down-sampling
		pVirtualFrame_->_acvgmShrPtrPyrPts[lvl]->download(*pVirtualFrame_->_acvmShrPtrPyrPts[lvl]); 
		pVirtualFrame_->_acvgmShrPtrPyrNls[lvl]->download(*pVirtualFrame_->_acvmShrPtrPyrNls[lvl]);
		for (short s = 1; s < pVirtualFrame_->pyrHeight(); s++){
			btl::device::cuda_resize_map(false, *pVirtualFrame_->_acvgmShrPtrPyrPts[s - 1], &*pVirtualFrame_->_acvgmShrPtrPyrPts[s]);
			btl::device::cuda_resize_map(true, *pVirtualFrame_->_acvgmShrPtrPyrNls[s - 1], &*pVirtualFrame_->_acvgmShrPtrPyrNls[s]);
			pVirtualFrame_->_acvgmShrPtrPyrPts[s]->download(*pVirtualFrame_->_acvmShrPtrPyrPts[s]);
			pVirtualFrame_->_acvgmShrPtrPyrNls[s]->download(*pVirtualFrame_->_acvmShrPtrPyrNls[s]);
		}//for each pyramid level
	}
	
	return;
}

void CCubicGrids::renderBoxGL() const
{
	// x axis
	glColor3f(1.f, .0f, .0f);
	//top
	glBegin(GL_LINE_LOOP);
	glVertex3f(0.f, 0.f, 0.f);
	glVertex3f(_fVolumeSizeM.x, 0.f, 0.f);
	glVertex3f(_fVolumeSizeM.x, 0.f, _fVolumeSizeM.z);
	glVertex3f(0.f, 0.f, _fVolumeSizeM.z);
	glEnd();
	//bottom
	glBegin(GL_LINE_LOOP);
	glVertex3f(0.f, _fVolumeSizeM.y, 0.f);
	glVertex3f(_fVolumeSizeM.x, _fVolumeSizeM.y, 0.f);
	glVertex3f(_fVolumeSizeM.x, _fVolumeSizeM.y, _fVolumeSizeM.z);
	glVertex3f(0.f, _fVolumeSizeM.y, _fVolumeSizeM.z);
	glEnd();
	//middle
	glBegin(GL_LINES);
	glVertex3f(0.f, 0.f, 0.f);
	glVertex3f(0.f, _fVolumeSizeM.y, 0.f);
	glEnd();
	glBegin(GL_LINES);
	glVertex3f(_fVolumeSizeM.x, 0.f, 0.f);
	glVertex3f(_fVolumeSizeM.x, _fVolumeSizeM.y, 0.f);
	glEnd();
	glBegin(GL_LINES);
	glVertex3f(_fVolumeSizeM.x, 0.f, _fVolumeSizeM.z);
	glVertex3f(_fVolumeSizeM.x, _fVolumeSizeM.y, _fVolumeSizeM.z);
	glEnd();
	glBegin(GL_LINES);
	glVertex3f(0.f, 0.f, _fVolumeSizeM.z);
	glVertex3f(0.f, _fVolumeSizeM.y, _fVolumeSizeM.z);
	glEnd();
}

//render all voxels if lvl_ <0 otherwise render lvl_ voxels
void CCubicGrids::renderOccupiedVoxels(btl::gl_util::CGLUtil::tp_ptr pGL_,int lvl_) {
	_vOccupied = btl::device::cuda_get_occupied_vg(_gpu_YXxZ_vg_idx, _fFeatureVoxelSizeM, _nFeatureScale, _gpu_pts, _vFeatureResolution, _gpu_feature_idx);
	float fRadius = 0.004f;
	glLineWidth(1.f);
	//glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
	//glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
	//glEnable(GL_LIGHTING);

	glDisable(GL_LIGHTING);
	
	for(int i=0; i < _vOccupied.size(); i++)
	{
		if( i!= lvl_ ){
			continue;
		}
		if( _vOccupied[i] == 0 ) continue;
		Mat pts; _gpu_pts[i].colRange(0,_vOccupied[i]).download(pts);
		if (i==0) {
			glColor4f(1.f,0.f,0.f,0.9f);
		}
		else if( i== 1) {
			glColor4f(1.f,0.4f,.4f,0.9f);
		}
		else if( i== 2){
			glColor4f(1.f,.5f,0.f,0.9f);
		}
		else if( i== 3){
			glColor4f(1.f,0.69f,.4f,0.9f);
		}
		else {
			glColor4f(1.f,1.f,0.f,0.9f);
		}
		//glBegin(GL_POINTS); 
		for (int n=0; n< _vOccupied[i]; n++ ) {
			float3 pt =  pts.ptr<float3>()[n];
			//glVertex3f ( pt.x, pt.y, pt.z );  
			glMatrixMode ( GL_MODELVIEW );
			glPushMatrix();
			glTranslatef ( pt.x, pt.y, pt.z );  
			pGL_->renderVoxelGL2(_fFeatureVoxelSizeM[i]);
			//gluSphere ( _quadratic, _fFeatureVoxelSizeM[i]/4.f, 12, 12 );
			glPopMatrix();
		}
		//glEnd();
	}

	return;
}

void CCubicGrids::loadNIFTI( const string& strPath_ )
{
	cout << ("Load tsdf...") << endl;
	//for tsdf
	{
		string volume_name = strPath_ + string("tsdf_volume.nii.gz");
		nifti_image* ptr_nim = nifti_image_read( volume_name.c_str() , 1 ) ;
		if (!ptr_nim) cout << ("load nifti volume incorrectly") << endl;

		string weight_name = strPath_ + string("tsdf_weight.nii.gz");
		nifti_image* ptr_weight = nifti_image_read(weight_name.c_str(), 1);
		if (!ptr_weight) cout << ("load nifti weight incorrectly") << endl;

		vector<Mat> vMats;

		if (_gpu_YXxZ_tsdf.type() == CV_16SC2){
			vMats.push_back(Mat(_uVolumeLevelXY, _VolumeResolution.z, CV_16SC1));
			vMats.push_back(Mat(_uVolumeLevelXY, _VolumeResolution.z, CV_16SC1));
			memcpy((void*)vMats[0].data, ptr_nim->data, sizeof(short)*_uVolumeTotal);
			memcpy((void*)vMats[1].data, ptr_weight->data, sizeof(short)*_uVolumeTotal);
		}
		else{
			vMats.push_back(Mat(_uVolumeLevelXY, _VolumeResolution.z, CV_32FC1));
			vMats.push_back(Mat(_uVolumeLevelXY, _VolumeResolution.z, CV_32FC1));
			memcpy((void*)vMats[0].data, ptr_nim->data, sizeof(float)*_uVolumeTotal);
			memcpy((void*)vMats[1].data, ptr_weight->data, sizeof(float)*_uVolumeTotal);
		}

		cv::Mat cvmVolume;
		cv::merge(vMats, cvmVolume);
		_gpu_YXxZ_tsdf.upload(cvmVolume);
		nifti_image_free(ptr_nim);
		nifti_image_free(ptr_weight);
	}
	cout << ("Load feature idx...") << endl;
	//for feature idx
	for (int i = 0; i<_nFeatureScale; i++)
	{
		cout << (i) << endl;
		ostringstream convert;   // stream used for the conversion
		convert << i;      // insert the textual representation of 'Number' in the characters in the stream
		string volume_name = strPath_ + string("feature_idx_")  + convert.str() + string(".nii.gz"); 
		nifti_image* ptr_nim = nifti_image_read( volume_name.c_str() , 1 ) ;
		if( !ptr_nim ) cout << ("load nifti file incorrectly") << endl;

		Mat cvmFeatureIdx(_vFeatureResolution[i].x * _vFeatureResolution[i].y, _vFeatureResolution[i].z, CV_32SC1);
		cvmFeatureIdx.setTo(Scalar(-1));
		//copy data 
		memcpy( (void*)cvmFeatureIdx.data, ptr_nim->data, sizeof(int) * _vFeatureResolution[i].x * _vFeatureResolution[i].y * _vFeatureResolution[i].z );

		_gpu_YXxZ_vg_idx[i].upload(cvmFeatureIdx);
		nifti_image_free( ptr_nim ) ;
	}
	//for display
	//btl::device::fill_in_octree( _gpu_YXxZ_volumetric_feature_idx );

	return;
}

void CCubicGrids::storeNIFTI(const string& strPath_) const
{
	std::experimental::filesystem::path dir(strPath_.c_str());
	if(std::experimental::filesystem::create_directories(dir)) {
		std::cout << "Success" << "\n";
	}

	{
		cv::Mat cvmVolume;
		_gpu_YXxZ_tsdf.download(cvmVolume);
		vector<cv::Mat> vMats;
		int dims[] = { 3, _VolumeResolution.x, _VolumeResolution.y, _VolumeResolution.z , 0, 0, 0, 0 };
		nifti_image *ptr_nim;
		nifti_image *ptr_weight;
		if (cvmVolume.type() == CV_16SC2){
			vMats.push_back(Mat(_uVolumeLevelXY, _VolumeResolution.z, CV_16SC1));
			vMats.push_back(Mat(_uVolumeLevelXY, _VolumeResolution.z, CV_16SC1));
			ptr_nim = nifti_make_new_nim(dims, NIFTI_TYPE_INT16, 1);
			ptr_weight = nifti_make_new_nim(dims, NIFTI_TYPE_INT16, 1);
		}
		else{
			vMats.push_back(Mat(_uVolumeLevelXY, _VolumeResolution.z, CV_32FC1));
			vMats.push_back(Mat(_uVolumeLevelXY, _VolumeResolution.z, CV_32FC1));
			ptr_nim = nifti_make_new_nim(dims, NIFTI_TYPE_FLOAT32, 1);
			ptr_weight = nifti_make_new_nim(dims, NIFTI_TYPE_FLOAT32, 1);
		}
		split( cvmVolume, vMats );

		ptr_nim->xyz_units = NIFTI_UNITS_MM;
		ptr_weight->xyz_units = NIFTI_UNITS_MM;

		float f_voxel_size_in_mm = _fVoxelSizeM*1000.f;
		ptr_nim->pixdim[1] = f_voxel_size_in_mm; 
		ptr_nim->pixdim[2] = f_voxel_size_in_mm; 
		ptr_nim->pixdim[3] = f_voxel_size_in_mm; 
		ptr_nim->dx = f_voxel_size_in_mm;
		ptr_nim->dy = f_voxel_size_in_mm;
		ptr_nim->dz = f_voxel_size_in_mm;
		ptr_nim->scl_slope = 0;

		ptr_weight->pixdim[1] = f_voxel_size_in_mm;
		ptr_weight->pixdim[2] = f_voxel_size_in_mm;
		ptr_weight->pixdim[3] = f_voxel_size_in_mm;
		ptr_weight->dx = f_voxel_size_in_mm;
		ptr_weight->dy = f_voxel_size_in_mm;
		ptr_weight->dz = f_voxel_size_in_mm;
		ptr_weight->scl_slope = 0;

		//copy data 
		if (cvmVolume.type() == CV_16SC2){
			memcpy(ptr_nim->data, (void*)vMats[0].data, sizeof(short)*_uVolumeTotal);
			ptr_nim->cal_max = 32767;
			ptr_nim->cal_min = -32767;
			ptr_nim->nbyper = 2; //bytes per voxel.

			memcpy(ptr_weight->data, (void*)vMats[1].data, sizeof(short)*_uVolumeTotal);
			ptr_weight->cal_max = 32767;
			ptr_weight->cal_min = 0;
			ptr_weight->nbyper = 2; //bytes per voxel.
		}
		else{
			memcpy(ptr_nim->data, (void*)vMats[0].data, sizeof(float)*_uVolumeTotal);
			ptr_nim->cal_max = 1.0f;
			ptr_nim->cal_min = -1.f;
			ptr_nim->nbyper = 4; //bytes per voxel.

			memcpy(ptr_weight->data, (void*)vMats[1].data, sizeof(float)*_uVolumeTotal);
			ptr_weight->cal_max = 1.0f;
			ptr_weight->cal_min = -1.f;
			ptr_weight->nbyper = 4; //bytes per voxel.
		}
		//ptr_nim->data = (void*) vMats[0].data; 


		{
			ptr_nim->nvox = _uVolumeTotal;
		
			ptr_nim->qform_code = 4;
			ptr_nim->qoffset_x = -f_voxel_size_in_mm * _VolumeResolution.x / 2;
			ptr_nim->qoffset_y = -f_voxel_size_in_mm * _VolumeResolution.y / 2;
			ptr_nim->qoffset_z =  f_voxel_size_in_mm * _VolumeResolution.z / 2;
			ptr_nim->qfac = -1.f;

			ptr_nim->quatern_b = 0.f;
			ptr_nim->quatern_c = 0.f;
			ptr_nim->quatern_d = 0.f;

			string volume_name = strPath_ + string("tsdf_volume");
			int ll = strlen(volume_name.c_str());
			char* tmpstr = nifti_makebasename(volume_name.c_str());
			ptr_nim->fname = (char *)calloc(1, ll + 8); 
			strcpy(ptr_nim->fname, tmpstr); //ll + 8, 
			ptr_nim->iname = (char *)calloc(1, ll + 8); 
			strcpy(ptr_nim->iname, tmpstr); //ll + 8, 
			free(tmpstr);
			strcat(ptr_nim->fname, ".nii");
			strcat(ptr_nim->iname, ".nii");
			strcat(ptr_nim->fname, ".gz");
			strcat(ptr_nim->iname, ".gz");

			nifti_image_write(ptr_nim);
			nifti_image_free(ptr_nim);
		}

		{
			ptr_weight->nvox = _uVolumeTotal;

			ptr_weight->qform_code = 4;
			ptr_weight->qoffset_x = -f_voxel_size_in_mm * _VolumeResolution.x / 2;
			ptr_weight->qoffset_y = -f_voxel_size_in_mm * _VolumeResolution.y / 2;
			ptr_weight->qoffset_z =  f_voxel_size_in_mm * _VolumeResolution.z / 2;
			ptr_weight->qfac = -1.f;

			ptr_weight->quatern_b = 0.f;
			ptr_weight->quatern_c = 0.f;
			ptr_weight->quatern_d = 0.f;

			string volume_name = strPath_ + string("tsdf_weight");
			int ll = strlen(volume_name.c_str());
			char* tmpstr = nifti_makebasename(volume_name.c_str());
			ptr_weight->fname = (char *)calloc(1, ll + 8); 
			strcpy(ptr_weight->fname, tmpstr); //ll + 8,
			ptr_weight->iname = (char *)calloc(1, ll + 8); 
			strcpy(ptr_weight->iname, tmpstr); //ll + 8,
			free(tmpstr);
			strcat(ptr_weight->fname, ".nii");
			strcat(ptr_weight->iname, ".nii");
			strcat(ptr_weight->fname, ".gz");
			strcat(ptr_weight->iname, ".gz");

			nifti_image_write(ptr_weight);
			nifti_image_free(ptr_weight);
		}
	}
	for(int i=0; i<_nFeatureScale; i++)
	{
		cv::Mat cvmVolumeIdx;
		_gpu_YXxZ_vg_idx[i].download(cvmVolumeIdx);
		int VolumeTotal = _vFeatureResolution[i].x * _vFeatureResolution[i].y * _vFeatureResolution[i].z;
		int dims[] = { 3, _vFeatureResolution[i].x, _vFeatureResolution[i].y, _vFeatureResolution[i].z, 0, 0, 0, 0 };
		nifti_image *ptr_nim = nifti_make_new_nim(dims, NIFTI_TYPE_INT32, 1);
		ptr_nim->xyz_units = NIFTI_UNITS_MM;

		float f_voxel_size_in_mm = _fFeatureVoxelSizeM[i]*1000.f;
		ptr_nim->pixdim[1] = f_voxel_size_in_mm; 
		ptr_nim->pixdim[2] = f_voxel_size_in_mm; 
		ptr_nim->pixdim[3] = f_voxel_size_in_mm; 
		ptr_nim->dx = f_voxel_size_in_mm;
		ptr_nim->dy = f_voxel_size_in_mm;
		ptr_nim->dz = f_voxel_size_in_mm;

		ptr_nim->scl_slope = 0;

		//copy data 
		memcpy(ptr_nim->data, (void*)cvmVolumeIdx.data,sizeof(int)*VolumeTotal);
		//ptr_nim->data = (void*) vMats[0].data; 
		ptr_nim->nvox = VolumeTotal;
		ptr_nim->nbyper = 4; //bytes per voxel.
		ptr_nim->cal_max = std::numeric_limits<int>::max();
		ptr_nim->cal_min = -2;

		ptr_nim->qform_code = 4;
		ptr_nim->qoffset_x = -f_voxel_size_in_mm * _vFeatureResolution[i].x / 2;
		ptr_nim->qoffset_y = -f_voxel_size_in_mm * _vFeatureResolution[i].y / 2;
		ptr_nim->qoffset_z =  f_voxel_size_in_mm * _vFeatureResolution[i].z / 2;

		ptr_nim->qfac = -1.f;

		ptr_nim->quatern_b = 0.f;
		ptr_nim->quatern_c = 0.f;
		ptr_nim->quatern_d = 0.f;

		ostringstream convert;   // stream used for the conversion
		convert << i;      // insert the textual representation of 'Number' in the characters in the stream
		string volume_name = strPath_ + string("feature_idx_")  + convert.str(); 

		int ll = strlen(volume_name.c_str()) ;
		char* tmpstr = nifti_makebasename(volume_name.c_str());
		ptr_nim->fname = (char *)calloc(1,ll+8) ; strcpy(ptr_nim->fname,tmpstr) ;
		ptr_nim->iname = (char *)calloc(1,ll+8) ; strcpy(ptr_nim->iname,tmpstr) ;
		free(tmpstr);
		strcat(ptr_nim->fname,".nii") ;
		strcat(ptr_nim->iname,".nii") ;
		strcat(ptr_nim->fname,".gz");
		strcat(ptr_nim->iname,".gz");

		nifti_image_write( ptr_nim ) ;
		nifti_image_free( ptr_nim ) ;
	}

	return;
}

void CCubicGrids::storeGlobalFeatures(const string& strPath_) const
{
	string global_features = strPath_ + string("GlobalFeatures.yml");
	cv::FileStorage storage(global_features, cv::FileStorage::WRITE);
	for (int i = 0; i < _nFeatureScale; i++)
	{
		Mat key_point_global;
		Mat descriptors_global;
		_gpu_global_3d_key_points[i].download(key_point_global);
		_gpu_global_descriptors[i].download(descriptors_global);
		//store features
		ostringstream convert;   // stream used for the conversion
		convert << i;
		storage << "TotalGlobalFeatures" + convert.str() << _vTotalGlobal[i];
		storage << "KeyPoints" + convert.str() << key_point_global;
		storage << "Descriptors" + convert.str() << descriptors_global;
	}
	storage.release();
	return;
}

void CCubicGrids::loadGlobalFeatures(const string& strPath_)
{
	string global_features = strPath_ + string("GlobalFeatures.yml");
	cv::FileStorage storage(global_features, cv::FileStorage::READ);
	for (int i = 0; i < _nFeatureScale; i++)
	{
		Mat key_point_global;
		Mat descriptors_global;
		ostringstream convert;   // stream used for the conversion
		convert << i;

		storage["TotalGlobalFeatures" + convert.str()] >> _vTotalGlobal[i];
		storage["KeyPoints" + convert.str()] >> key_point_global;
		storage["Descriptors" + convert.str()] >> descriptors_global;

		_gpu_global_3d_key_points[i].upload(key_point_global);
		_gpu_global_descriptors[i].upload(descriptors_global);
	}
	storage.release();
	return;
}

void CCubicGrids::displayAllGlobalFeatures(int lvl_, bool bRenderSpheres_) const{
	// Generate the data

	for (int i = 0; i < _nFeatureScale; i++) {
		if (lvl_ != -1 && i != lvl_){
			continue;
		}

		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glDisable(GL_LIGHTING);
		Mat cvmKeyPointGlobal;
		_gpu_global_3d_key_points[i].colRange(0, _vTotalGlobal[i]).download(cvmKeyPointGlobal);

		if (true){
			//if (bRenderSpheres_){
			for (int nIdx = 0; nIdx < cvmKeyPointGlobal.cols; nIdx++){
				if (cvmKeyPointGlobal.ptr<float>(6)[nIdx] < 0.f) continue;
				float fX = cvmKeyPointGlobal.ptr<float>(0)[nIdx];
				float fY = cvmKeyPointGlobal.ptr<float>(1)[nIdx];
				float fZ = cvmKeyPointGlobal.ptr<float>(2)[nIdx];
				float fS = 0.005f;// cvmKeyPointGlobal.ptr<float>(7)[nIdx] / .f;
				float fAS = cvmKeyPointGlobal.ptr<float>(7)[nIdx];
				int l = int( fAS / _fVoxelSizeM );
				int nP = 12;
				if (i <= 1) {
					glColor4f(1.f, 0.f, 0.f, 1.f);
				}
				else if (i <= 2) {
					glColor4f(1.f, .5f, 0.f, 1.f);
				}
				else if (i <= 3){
					glColor4f(1.f, 0.69f, .4f, 1.f);
				}
				else if (i <= 4){
					glColor4f(1.f, 0.69f, .4f, 1.f);
				}
				else {
					glColor4f(1.f, 1.f, 0.f, 1.f);
				}
				//glBegin(GL_POINTS);
				//glVertex3f(fX, fY, fZ);
				//glEnd();
				glMatrixMode(GL_MODELVIEW);
				glPushMatrix();
				glTranslatef(fX, fY, fZ);
				gluSphere(_quadratic, fS, nP, nP);
				glPopMatrix();
			}
		}
		else{//render normal
			glDisable(GL_LIGHTING);
			glDisable(GL_BLEND);
			glLineWidth(1.f);
			glPointSize(3.f);
			for (int nIdx = 0; nIdx < cvmKeyPointGlobal.cols; nIdx++){
				if (cvmKeyPointGlobal.ptr<float>(6)[nIdx] < 0.f) continue;
				float fX = cvmKeyPointGlobal.ptr<float>(0)[nIdx];
				float fY = cvmKeyPointGlobal.ptr<float>(1)[nIdx];
				float fZ = cvmKeyPointGlobal.ptr<float>(2)[nIdx];
				float fS = cvmKeyPointGlobal.ptr<float>(7)[nIdx] / 2.f;

				glBegin(GL_POINTS);
				glVertex3f(fX, fY, fZ);
				glEnd();
				//normal
				glColor4f(1.f, 0.f, 0.f, 1.f);
				glBegin(GL_LINES);
				glVertex3f(fX, fY, fZ);
				float fNX = cvmKeyPointGlobal.ptr<float>(3)[nIdx];
				float fNY = cvmKeyPointGlobal.ptr<float>(4)[nIdx];
				float fNZ = cvmKeyPointGlobal.ptr<float>(5)[nIdx];
				glVertex3f(fX + fS*fNX, fY + fS*fNY, fZ + fS*fNZ);
				//glNormal3f ( fNX, fNY, fNZ );  
				glEnd();
				//draw main direction
				glColor4f(0.f, 1.f, 0.f, 1.f);
				glBegin(GL_LINES);
				glVertex3f(fX, fY, fZ);
				float fMDX = cvmKeyPointGlobal.ptr<float>(9)[nIdx];
				float fMDY = cvmKeyPointGlobal.ptr<float>(10)[nIdx];
				float fMDZ = cvmKeyPointGlobal.ptr<float>(11)[nIdx];
				glVertex3f(fX + fS*fMDX, fY + fS*fMDY, fZ + fS*fMDZ);
				glEnd();
				Vector3f nn(fNX, fNY, fNZ);
				Vector3f v1(fMDX, fMDY, fMDZ);
				Vector3f v2 = nn.cross(v1);
				glColor4f(0.f, 0.f, 1.f, 1.f);
				glBegin(GL_LINES);
				glVertex3f(fX, fY, fZ);
				glVertex3f(fX + fS*v2(0), fY + fS*v2(1), fZ + fS*v2(2));
				glEnd();
			}
		}
	}

	return;
}

double CCubicGrids::icpFrm2Frm(const CRGBDFrame::tp_ptr pCurFrame_, const CRGBDFrame::tp_ptr pPrevFrame_, const short asICPIterations_[], Eigen::Matrix3f* peimRw_, Eigen::Vector3f* peivTw_, Eigen::Vector4i* pActualIter_) const
{
	//initialize R,T from input.
	Eigen::Matrix3f& eimRwCurTmp = *peimRw_;
	Eigen::Vector3f& eivTwCurTmp = *peivTw_;

	const float fDistThreshold = 0.25f; //meters works for the desktop non-stationary situation.
	const float fCosAngleThres = (float) cos (60.f * M_PI / 180.f);
		
	//get R,T of previous 
	Eigen::Matrix3f eimrmRwPrev = pPrevFrame_->_Rw.transpose();//because by default eimrmRwPrev is column major
	Eigen::Vector3f eivTwPrev = pPrevFrame_->_Tw;
	const pcl::device::Mat33&  devRwPrev = pcl::device::device_cast<pcl::device::Mat33> (eimrmRwPrev);
	const float3& devTwPrev = pcl::device::device_cast<float3> (eivTwPrev);
	//the following two lines ensure that once eimRwCurTmp and eivTwCurTmp are updated, devRwCurTrans and devTwCur are updated implicitly.
	pcl::device::Mat33& devRwCurTrans = pcl::device::device_cast<pcl::device::Mat33> (eimRwCurTmp);
	float3& devTwCur = pcl::device::device_cast<float3> (eivTwCurTmp);

	Eigen::Matrix3f eimRwCurBest = eimRwCurTmp;
	Eigen::Vector3f eivTwCurBest = eivTwCurTmp;
	//from low resolution to high
	//double dEnergyThresh[4]={0.000,0.00000,0.0000000,0.0000};
	double dMinEnergy = numeric_limits<double>::max();
	for (short sPyrLevel = pCurFrame_->pyrHeight() - 1; sPyrLevel >= 0; sPyrLevel--){
		// for each pyramid level we have a min energy and corresponding best R t
		if (asICPIterations_[sPyrLevel] > 0){
			dMinEnergy = btl::device::calc_energy_icp_fr_2_fr(_intrinsics(sPyrLevel),
				fDistThreshold, fCosAngleThres,
				devRwCurTrans, devTwCur, devRwPrev, devTwPrev, *pCurFrame_->_acvgmShrPtrPyrDepths[sPyrLevel],
				*pPrevFrame_->_acvgmShrPtrPyrPts[sPyrLevel], *pPrevFrame_->_acvgmShrPtrPyrNls[sPyrLevel],
				*pCurFrame_->_acvgmShrPtrPyrPts[sPyrLevel], *pCurFrame_->_acvgmShrPtrPyrNls[sPyrLevel], *pCurFrame_->_pry_mask[sPyrLevel]);
		}

		for ( short sIter = 0; sIter < asICPIterations_[sPyrLevel]; ++sIter ) {
			//get R and T
			cuda::GpuMat cvgmSumBuf = btl::device::cuda_icp_fr_2_fr(_intrinsics(sPyrLevel),
																	fDistThreshold,fCosAngleThres,
																	devRwCurTrans, devTwCur, devRwPrev, devTwPrev, *pCurFrame_->_acvgmShrPtrPyrDepths[sPyrLevel],
																	*pPrevFrame_->_acvgmShrPtrPyrPts[sPyrLevel], *pPrevFrame_->_acvgmShrPtrPyrNls[sPyrLevel],
																	*pCurFrame_->_acvgmShrPtrPyrPts[sPyrLevel],      *pCurFrame_->_acvgmShrPtrPyrNls[sPyrLevel] );
			extractRTFromBuffer(cvgmSumBuf, &eimRwCurTmp, &eivTwCurTmp); // Note that devRwCurTrans, devTwCur are updated as well
			double dEnergy = btl::device::calc_energy_icp_fr_2_fr ( _intrinsics(sPyrLevel),
																	fDistThreshold, fCosAngleThres,
																	devRwCurTrans, devTwCur, devRwPrev, devTwPrev, *pCurFrame_->_acvgmShrPtrPyrDepths[sPyrLevel],
																	*pPrevFrame_->_acvgmShrPtrPyrPts[sPyrLevel], *pPrevFrame_->_acvgmShrPtrPyrNls[sPyrLevel],
																	*pCurFrame_->_acvgmShrPtrPyrPts[sPyrLevel], *pCurFrame_->_acvgmShrPtrPyrNls[sPyrLevel], *pCurFrame_->_pry_mask[sPyrLevel]);			
			if (dEnergy < dMinEnergy) {
				dMinEnergy = dEnergy;
				eimRwCurBest = eimRwCurTmp;
				eivTwCurBest = eivTwCurTmp;
			}
		}//for each iteration
		eimRwCurTmp = eimRwCurBest;// Note that devRwCurTrans, devTwCur are updated as well
		eivTwCurTmp = eivTwCurBest;
	}//for pyrlevel
	//short sPyrLevel = 0;
	//dMinEnergy = btl::device::calc_energy_icp_fr_2_fr(_intrinsics(sPyrLevel),
	//													fDistThreshold, fCosAngleThres,
	//													devRwCurTrans, devTwCur, devRwPrev, devTwPrev, *pCurFrame_->_acvgmShrPtrPyrDepths[sPyrLevel],
	//													*pPrevFrame_->_acvgmShrPtrPyrPts[sPyrLevel], *pPrevFrame_->_acvgmShrPtrPyrNls[sPyrLevel],
	//													*pCurFrame_->_acvgmShrPtrPyrPts[sPyrLevel], *pCurFrame_->_acvgmShrPtrPyrNls[sPyrLevel], *pCurFrame_->_pry_mask[sPyrLevel]);

	return dMinEnergy;
}

double CCubicGrids::verifyPoseHypothesesAndRefine(CRGBDFrame::tp_ptr pCurFrame_, CRGBDFrame::tp_ptr pPrevFrame_, vector<Eigen::Affine3f>& v_k_hypothese_poses, int nICPMethod_, int nStage_,
	Matrix3f* pRw_, Vector3f* pTw_, int* p_best_idx_){
	//assert the hypotheses list is not empty
	double dICPEnergy = numeric_limits<double>::max();
	if (v_k_hypothese_poses.empty()){
		//using the ICP to refine each hypotheses and store their alignment score
		//do ICP as one of pose hypotheses
		Matrix3f eimRw; Vector3f eivTw;
		//refine R,t
		Eigen::Vector4i eivIter;
		pPrevFrame_->getRnt(&*pRw_, &*pTw_);//use previous R,t as initial pose
		{
			short asICPIterations[4] = { 3, 2, 1, 1};
			dICPEnergy = icpFrm2Frm(pCurFrame_, pPrevFrame_, asICPIterations, &*pRw_, &*pTw_, &eivIter);
		}
		pCurFrame_->setRTw(*pRw_, *pTw_);
		*p_best_idx_ = -1;
		return dICPEnergy;
	}

	//using the ICP to refine each hypotheses and store their alignment score
	vector<Matrix3f> v_refined_R;
	vector<Vector3f> v_refined_t;
	Mat energy_ICP;
	energy_ICP.create(1, v_k_hypothese_poses.size(), CV_64FC1); energy_ICP.setTo(numeric_limits<double>::max());

	for (int i = 0; i < v_k_hypothese_poses.size(); i++)
	{
		const short asICPIterations[4] = { 0, 0, 1, 1 };
		Matrix3f eimRw; Vector3f eivTw;
		btl::utility::convertPrj2Rnt(v_k_hypothese_poses[i], &eimRw, &eivTw);

		float s = eimRw.sum();
		if (fabs(s) < 0.0001 || std::isnan<float>(s)){
			energy_ICP.at<double>(0, i) = numeric_limits<double>::max();
			v_refined_R.push_back(Matrix3f::Zero());
			v_refined_t.push_back(Vector3f::Zero());
			continue;
		}

		//ICP -- refine R,t
		Eigen::Vector4i eivIter;
		switch (nICPMethod_){
		case btl::Frm_2_Frm_ICP:
			if (nStage_ == btl::Relocalisation_Only){
				//get virtual frame as previous, if current frame is a lost one, no previous frame can be used for refinement, 
				//therefore a virtual frame is required here.
				pPrevFrame_->setRTw(eimRw, eivTw);
				rayCast(&*pPrevFrame_,1);
			}
			energy_ICP.at<double>(0, i) = icpFrm2Frm(pCurFrame_, pPrevFrame_, asICPIterations, &eimRw, &eivTw, &eivIter);
			break;
		default:
			cout << ("Failure - unrecognized ICP method.") << endl;
			break;
		}
		if (energy_ICP.at<double>(0, i) != energy_ICP.at<double>(0, i) || energy_ICP.at<double>(0, i) < 0 )
			energy_ICP.at<double>(0, i) = numeric_limits<double>::max();

		v_refined_R.push_back(eimRw);
		v_refined_t.push_back(eivTw);
	}

	//use the best hypotheses
	//Mat SortedIdx;
	//cv::sortIdx(energy_ICP, SortedIdx, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
	int best_k;// = SortedIdx.at<int>(0, 0);
	for (best_k = 0; best_k < v_k_hypothese_poses.size(); best_k++){
		dICPEnergy = energy_ICP.at<double>(0, best_k);
		if (dICPEnergy >= 0.f) break;
	}
	for (int i = best_k + 1; i < v_k_hypothese_poses.size(); i++){
		if (energy_ICP.at<double>(0, i) < dICPEnergy && energy_ICP.at<double>(0, i) >= 0.f) {
			best_k = i;
			dICPEnergy = energy_ICP.at<double>(0, i);
		}
	}

	best_k >= v_k_hypothese_poses.size() ? 0 : best_k;
	*p_best_idx_ = best_k;
	//show refined pose
	*pRw_ = v_refined_R[best_k];
	*pTw_ = v_refined_t[best_k];
	pCurFrame_->setRTw(*pRw_, *pTw_);

	//final refinement
	if (true){
		double dICPEnergyFinal = numeric_limits<double>::max();
		Matrix3f eimRw = *pRw_; Vector3f eivTw = *pTw_;
		float s = eimRw.sum();
		if (fabs(s) > 0.0001 && !std::isnan<float>(s)){
			//ICP -- refine R,t
			Eigen::Vector4i eivIter;
			switch (nICPMethod_){
			case Frm_2_Frm_ICP:
				//if (nStage_ == btl::Relocalisation_Only)
				{
					//get virtual frame as previous, if current frame is a lost one, no previous frame can be used for refinement, 
					//therefore a virtual frame is required here.
					pPrevFrame_->setRTw(eimRw, eivTw);
					rayCast(&*pPrevFrame_);
				}
				{
					const short asICPIterations[4] = { 1, 1, 1, 1 };
					dICPEnergyFinal = icpFrm2Frm(pCurFrame_, pPrevFrame_, asICPIterations, &eimRw, &eivTw, &eivIter);
				}
				break;
			default:
				cout << ("Failure - unrecognized ICP method.") << endl;
				break;
			}
		}
		*pRw_ = eimRw; *pTw_ = eivTw;
		dICPEnergy = dICPEnergyFinal;
	}
	pCurFrame_->setRTw(*pRw_, *pTw_);
	cout << (dICPEnergy) << endl;
	return dICPEnergy;
}

void CCubicGrids::extractRTFromBuffer(const cuda::GpuMat& cvgmSumBuf_, Eigen::Matrix3f* peimRw_, Eigen::Vector3f* peivTw_) const{
	Mat cvmSumBuf;
	cvgmSumBuf_.download(cvmSumBuf);
	double* aHostTmp = (double*)cvmSumBuf.data;
	//declare A and b
	Eigen::Matrix<double, 6, 6, Eigen::RowMajor> A;
	Eigen::Matrix<double, 6, 1> b;
	//retrieve A and b from cvmSumBuf
	short sShift = 0;
	for (int i = 0; i < 6; ++i){   // rows
		for (int j = i; j < 7; ++j) { // cols + b
			double value = aHostTmp[sShift++];
			if (j == 6)       // vector b
				b.data()[i] = value;
			else
				A.data()[j * 6 + i] = A.data()[i * 6 + j] = value;
		}//for each col
	}//for each row
	//checking nullspace
	double dDet = A.determinant();
	if (fabs(dDet) < 1e-15 || dDet != dDet){
		if (dDet != dDet)
			cout << ("Failure -- dDet cannot be qnan. ") << endl;
		//reset ();
		return;
	}//if dDet is rational
	//float maxc = A.maxCoeff();

	Eigen::Matrix<float, 6, 1> result = A.llt().solve(b).cast<float>();
	//Eigen::Matrix<float, 6, 1> result = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);

	float alpha = result(0);
	float beta = result(1);
	float gamma = result(2);

	Eigen::Matrix3f Rinc = (Eigen::Matrix3f)Eigen::AngleAxisf(gamma, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(beta, Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(alpha, Eigen::Vector3f::UnitX());
	Eigen::Vector3f tinc = result.tail<3>();

	//compose
	//eivTwCur   = Rinc * eivTwCur + tinc;
	//eimrmRwCur = Rinc * eimrmRwCur;
	Eigen::Vector3f eivTinv = -peimRw_->transpose()* (*peivTw_);
	Eigen::Matrix3f eimRinv = peimRw_->transpose();
	eivTinv = Rinc * eivTinv + tinc;
	eimRinv = Rinc * eimRinv;
	*peivTw_ = -eimRinv.transpose() * eivTinv;
	*peimRw_ = eimRinv.transpose();
}

void CCubicGrids::gpuMarchingCubes(){
	GpuMat _occupied_voxels, _triangles_buffer, _normals_buffer;
	_occupied_voxels.create(3, 7000000, CV_32SC1);

	Mat cpuEdgeTable(1, 256, CV_32SC1, (void*)edgeTable);
	Mat cpuTriTable(1, 256 * 16, CV_32SC1, (void*)triTable);
	Mat cpuNumVertsTable(1, 256, CV_32SC1, (void*)numVertsTable);
	//cout << cpuNumVertsTable << endl;
	GpuMat eTable(cpuEdgeTable);
	GpuMat tTable(cpuTriTable);
	GpuMat nTable(cpuNumVertsTable);
	pcl::device::bindTextures(GpuMat(cpuEdgeTable), tTable, nTable);
	int active_voxels = pcl::device::getOccupiedVoxels(_gpu_YXxZ_tsdf, _VolumeResolution, _occupied_voxels);
	if (active_voxels == 0){
		pcl::device::unbindTextures();
		return;
	}
	GpuMat occupied_voxels = _occupied_voxels.colRange(0, active_voxels);
	int _total_vertexes = pcl::device::computeOffsetsAndTotalVertexes(occupied_voxels);
	cout << _total_vertexes << endl;
	_triangles_buffer.create(1, _total_vertexes, CV_32FC3);
	_normals_buffer.create(1, _total_vertexes/3, CV_32FC3);
	pcl::device::generateTriangles(_gpu_YXxZ_tsdf, occupied_voxels, _fVolumeSizeM, _VolumeResolution, _triangles_buffer, _normals_buffer);
	_triangles_buffer.colRange(0, _total_vertexes).download(_triangles);
	_normals_buffer.colRange(0, _total_vertexes / 3).download(_normals);
	pcl::device::unbindTextures();

	return;
}

void CCubicGrids::displayTriangles() const{

	glEnable(GL_LIGHTING); /* glEnable(GL_TEXTURE_2D);*/
	float shininess = 15.0f;
	float diffuseColor[3] = { 0.8f, 0.8f, 0.8f };
	float specularColor[4] = { .2f, 0.2f, 0.2f, 1.0f };
	// set specular and shiniess using glMaterial (gold-yellow)
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess); // range 0 ~ 128
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specularColor);
	// set ambient and diffuse color using glColorMaterial (gold-yellow)
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glColor3fv(diffuseColor);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	//glMultMatrixd(_T_fw.inverse().matrix().data());//times with original model view matrix manipulated by mouse or keyboard

	assert(_triangles.cols / 3 == _normals.cols);
	for (int t = 0; t < _normals.cols; t++)
	{
		const float* vv = _triangles.ptr<float>() + t * 3 * 3;
		const float* nn = _normals.ptr<float>() + t * 3;

		glBegin(GL_TRIANGLES);
		glNormal3fv(nn);
		glVertex3fv(vv);
		glVertex3fv(vv + 3);
		glVertex3fv(vv + 6);
		glEnd();
	}

	glPopMatrix();
	return;
}

}//geometry
}//btl
