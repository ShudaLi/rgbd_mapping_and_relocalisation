#ifndef BTL_UTILITY_HEADER
#define BTL_UTILITY_HEADER


#include "OtherUtil.hpp"
#include "EigenUtil.hpp"
#include "CVUtil.hpp"
#include "Converters.hpp"

//#include <pcl/kdtree/kdtree.h>
//#include <pcl/point_types.h>
//#include <pcl/features/normal_3d.h>
#define _USE_MATH_DEFINES
#include <math.h>
#undef _USE_MATH_DEFINES

namespace btl
{
namespace utility
{

/*
template< class T >
void normalEstimationGL( const T* pDepth_, const cv::Mat& cvmRGB_, 
	std::vector<const unsigned char*>* vColor_, std::vector<Eigen::Vector3d>* vPt_, std::vector<Eigen::Vector3d>* vNormal_, 
	std::vector<int>* pvX_= NULL, std::vector<int>* pvY_ = NULL)
{
	BTL_ASSERT(cvmRGB_.type()== CV_8UC3, "CVUtil::normalEstimationGL() Error: the input must be a 3-channel color image.")
	vColor_->clear();
	vPt_->clear();
	vNormal_->clear();

	if(pvX_&&pvY_)
	{
		pvX_->clear();
		pvY_->clear();
	}
	unsigned char* pColor_ = (unsigned char*) cvmRGB_.data;

	Eigen::Vector3d n1, n2, n3, v(0,0,1);

	//calculate normal
	for( int r = 0; r < cvmRGB_.rows; r++ )
	for( int c = 0; c < cvmRGB_.cols; c++ )
	{
		// skip the right and bottom boarder line
		if( c == cvmRGB_.cols-1 || r == cvmRGB_.rows-1 )
		{
			pColor_+=3;
			continue;
		}
		size_t i;
		i = r*cvmRGB_.cols + c;
		size_t ii = i*3;
		Eigen::Vector3d pti  ( pDepth_[ii],-pDepth_[ii+1],-pDepth_[ii+2] );
		size_t i1;
		i1 = i + 1;
		ii = i1*3;
		Eigen::Vector3d pti1 ( pDepth_[ii],-pDepth_[ii+1],-pDepth_[ii+2] );
		size_t j1;
		j1 = i + cvmRGB_.cols;
		ii = j1*3;
		Eigen::Vector3d ptj1 ( pDepth_[ii],-pDepth_[ii+1],-pDepth_[ii+2] );

		if( fabs( pti(2) ) > 0.0000001 && fabs( pti1(2) ) > 0.0000001 && fabs( ptj1(2) ) > 0.0000001 )
		{
			n1 = pti1 - pti;
			n2 = ptj1 - pti;
			n3 = n1.cross(n2);
			double dNorm = n3.norm() ;
			if ( dNorm > SMALL )
			{
				n3/=dNorm;
				if ( v.dot(n3) < 0 )
				{
					//PRINT( n3 );
					n3 = -n3;
				}
				vColor_->push_back(pColor_);
				vPt_->push_back(pti);
				vNormal_->push_back(n3);
				if(pvX_&&pvY_)
				{
					pvX_->push_back(c);
					pvY_->push_back(r);
				}
			}
		}
		pColor_+=3;
	}
	return;
}*/
template< class T >
T norm3( const T* pPtCur_, const T* pPtRef_ ){
	T tTmp0 = pPtCur_[0]-pPtRef_[0];
	T tTmp1 = pPtCur_[1]-pPtRef_[1];
	T tTmp2 = pPtCur_[2]-pPtRef_[2];
	return std::sqrt(tTmp0*tTmp0 + tTmp1*tTmp1 + tTmp2*tTmp2);
}

template< class T >
T norm3( const T* pPtCurCam_, const T* pPtRefWor_, const T* pCurRw_/*col major*/, const T* pCurTw_ ){
	//C = CurRw*W + T
	T tRefCam[3];
	tRefCam[0] = pCurRw_[0]*pPtRefWor_[0] + pCurRw_[3]*pPtRefWor_[1] + pCurRw_[6]*pPtRefWor_[2] + pCurTw_[0];
	tRefCam[1] = pCurRw_[1]*pPtRefWor_[0] + pCurRw_[4]*pPtRefWor_[1] + pCurRw_[7]*pPtRefWor_[2] + pCurTw_[1];
	tRefCam[2] = pCurRw_[2]*pPtRefWor_[0] + pCurRw_[5]*pPtRefWor_[1] + pCurRw_[8]*pPtRefWor_[2] + pCurTw_[2];

	return norm3< T >( pPtCurCam_,tRefCam);
}

/*template< class T >
void normalEstimationGLPCL( const T* pDepth_, const cv::Mat& cvmRGB_, int nKNearest_, std::vector<const unsigned char*>* vColor_, std::vector<Eigen::Vector3d>* vPt_, std::vector<Eigen::Vector3d>* vNormal_ )
{
	vColor_->clear();
	vPt_->clear();
	vNormal_->clear();

	const unsigned char* pColor_ = (const unsigned char*)cvmRGB_.data;

	pcl::PointCloud<pcl::PointXYZ> _cloudNoneZero;
	pcl::PointCloud<pcl::Normal>   _cloudNormals;
	for( unsigned int r = 0; r < cvmRGB_.rows; r++ )
	for( unsigned int c = 0; c < cvmRGB_.cols; c++ )
	{
		size_t i;
		i = r*cvmRGB_.cols + c;
		size_t ii = i*3;
		if( pDepth_[ii+2] > 0.0000001 )
		{
			pcl::PointXYZ point( pDepth_[ii],-pDepth_[ii+1],-pDepth_[ii+2] );
			_cloudNoneZero.push_back(point);
			vColor_->push_back( pColor_ );
		}
		pColor_+=3;
	}

	//get normal using pcl
	pcl::search::KdTree<pcl::PointXYZ>::Ptr pTree (new pcl::search::KdTree<pcl::PointXYZ>());
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;

	// Estimate point normals
	ne.setSearchMethod (pTree);
	ne.setInputCloud (_cloudNoneZero.makeShared());
	ne.setKSearch (nKNearest_);
	ne.compute (_cloudNormals);

	// collect the points and normals

	for (size_t i = 0; i < _cloudNoneZero.points.size (); ++i)
	{
		Eigen::Vector3d eivPt  ( _cloudNoneZero.points[i].x,_cloudNoneZero.points[i].y,_cloudNoneZero.points[i].z );
		Eigen::Vector3d eivNl  ( _cloudNormals.points[i].normal_x,_cloudNormals.points[i].normal_y,_cloudNormals.points[i].normal_z );
		vPt_->push_back(eivPt);
		vNormal_->push_back(eivNl);
	}
	return;
}*/


template< class T >
void normalVotes( const T* pNormal_, const double& dS_, int* pR_, int* pC_, btl::utility::tp_coordinate_convention eCon_ = btl::utility::BTL_GL)
{
	//pNormal[3] is a normal defined in a right-hand reference
	//system with positive-z the elevation, and counter-clockwise from positive-x is
	//the azimuth, 
	//dS_ is the step length in radian
	//*pR_ is the discretized elevation 
	//*pC_ is the discretized azimuth

	//normal follows GL-convention
	const double dNx = pNormal_[0];
	const double dNy = pNormal_[1];
	double dNz = pNormal_[2];
	if(btl::utility::BTL_CV == eCon_) {dNz = -dNz;}

	double dA = atan2(dNy,dNx); //atan2 ranges from -pi to pi
	dA = dA <0 ? dA+2*M_PI :dA; // this makes sure that dA ranging from 0 to 2pi
	double dyx= sqrt( dNx*dNx + dNy*dNy );
	double dE = atan2(dNz,dyx);

	*pC_ = floor(dA/dS_);
	*pR_ = floor(dE/dS_);

}

template< class T >
void avgNormals(const cv::Mat& cvmNls_,const std::vector<unsigned int>& vNlIdx_, Eigen::Vector3d* peivAvgNl_)
{
	//note that not all normals in vNormals_ will be averaged
	*peivAvgNl_ << 0,0,0;
	const float* const pNl = (float*)cvmNls_.data;
	for(std::vector<unsigned int>::const_iterator cit_vNlIdx = vNlIdx_.begin();
		cit_vNlIdx!=vNlIdx_.end(); cit_vNlIdx++)
	{
		unsigned int nOffset = (*cit_vNlIdx)*3;
		*peivAvgNl_+=Eigen::Vector3d(pNl[nOffset],pNl[nOffset+1],pNl[nOffset+2]);
	}
	peivAvgNl_->normalize();
}

template< class T >
bool isNormalSimilar( const T* pNormal1_, const Eigen::Vector3d& eivNormal2_, const T& dCosThreshold_)
{
	//if the angle between eivNormal1_ and eivNormal2_ is larger than dCosThreshold_
	//the two normal is not similar and return false
	T dCos = pNormal1_[0]*eivNormal2_[0] + pNormal1_[1]*eivNormal2_[1] + pNormal1_[2]*eivNormal2_[2];
	if(dCos>dCosThreshold_)
		return true;
	else
		return false;
}
/*
template< class T >
void normalCluster( const std::vector<Eigen::Vector3d>& vNormal_, const std::vector< Eigen::Vector3d>& veivNlCluster_, 
	int nSamples_, std::vector<unsigned int>* pvLabel_)
{
	//normalCluster is an exhaustive function to cluster a vector of normals onto a vector of pre-calculated
	//normal clusters. the labeling will be returned
	//calculate the threshold, ie the angle difference between the cluster center and a normal
	const double dCosThreshold = std::cos(M_PI_2/nSamples_);

	std::vector<Eigen::Vector3d>::const_iterator citNormal = vNormal_.begin();
	pvLabel_->clear();
	pvLabel_->resize(vNormal_.size());
	std::vector<unsigned int>::iterator it = pvLabel_->begin();
	for (;citNormal!=vNormal_.end(); citNormal++)
	{
		std::vector< Eigen::Vector3d>::const_iterator citCluster = veivNlCluster_.begin();
		for (int nClusterIdx=0;citCluster!=veivNlCluster_.end(); citCluster++, nClusterIdx++)
		{
			if(isNormalSimilar< double >(*citCluster,*citNormal,dCosThreshold))
				*it = nClusterIdx;//set label;
		}//for all clusters
	}//for all normals
	return;
}*/
template< class T >
void normalCluster( const cv::Mat& cvmNls_,const std::vector< unsigned int >& vNlIdx_, 
	const Eigen::Vector3d& eivClusterCenter_, 
	const double& dCosThreshold_, const short& sLabel_, cv::Mat* pcvmLabel_, std::vector< unsigned int >* pvNlIdx_ )
{
	//the pvLabel_ must be same length as vNormal_ 
	//with each element assigned with a NEGATIVE value
	const float* pNl = (const float*)cvmNls_.data; 
	short* pLabel = (short*) pcvmLabel_->data;
	for( std::vector< unsigned int >::const_iterator cit_vNlIdx_ = vNlIdx_.begin();
		cit_vNlIdx_!= vNlIdx_.end(); cit_vNlIdx_++ ){
		int nOffset = (*cit_vNlIdx_)*3;
		if( pLabel[*cit_vNlIdx_]<0 && btl::utility::isNormalSimilar< float >(pNl+nOffset,eivClusterCenter_,dCosThreshold_) )	{
			pLabel[*cit_vNlIdx_] = sLabel_;
			pvNlIdx_->push_back(*cit_vNlIdx_);
		}//if
	}
	return;
}

}//utility
}//btl
#endif
