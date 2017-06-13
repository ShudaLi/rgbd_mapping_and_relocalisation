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


#ifndef BTL_Eigen_UTILITY_HEADER
#define BTL_Eigen_UTILITY_HEADER

//eigen-based helpers
#include "OtherUtil.hpp"
#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace btl
{
namespace utility
{
using namespace Eigen;
using namespace btl::utility;


template< class T >
Eigen::Matrix< T, 4, 4 > inver(const Eigen::Matrix< T, 4, 4 >& F_ctw_){
	using namespace Eigen;
	Matrix< T, 4, 4 > F_wtc; F_wtc.setIdentity();

	Matrix< T, 3, 3 > R_trans = F_ctw_.block(0,0,3,3);
	Matrix< T, 3, 1 > Cw = F_ctw_.block(0,3,3,1);

	F_wtc.block(0,0,3,3) = R_trans.transpose();
	F_wtc.block(0,3,3,1) = -R_trans*Cw;
	return F_wtc;
}

template< class T >
void getCwVwFromPrj_Cam2World(const Eigen::Matrix< T, 4, 4 >&  Prj_ctw_,Eigen::Matrix< T, 3, 1 >* pCw_,Eigen::Matrix< T, 3, 1 >* pVw_){
	*pCw_ = Prj_ctw_.template block<3,1>(0,3); //4th column is camera centre
	*pVw_ = Prj_ctw_.template block<3,1>(0,0); //1st column is viewing direction
}

template< class T >
void getRTCVfromModelViewGL ( const Eigen::Matrix< T, 4, 4 >&  mMat_, Eigen::Matrix< T, 3, 3 >* pmR_, Eigen::Matrix< T, 3, 1 >* pvT_ )
{
    (* pmR_) ( 0, 0 ) =  mMat_ ( 0, 0 );   (* pmR_) ( 0, 1 ) =   mMat_ ( 0, 1 );  (* pmR_) ( 0, 2 ) = mMat_ ( 0, 2 );
    (* pmR_) ( 1, 0 ) = -mMat_ ( 1, 0 );   (* pmR_) ( 1, 1 ) = - mMat_ ( 1, 1 );  (* pmR_) ( 1, 2 ) = -mMat_ ( 1, 2 );
    (* pmR_) ( 2, 0 ) = -mMat_ ( 2, 0 );   (* pmR_) ( 2, 1 ) = - mMat_ ( 2, 1 );  (* pmR_) ( 2, 2 ) = -mMat_ ( 2, 2 );
    
	(*pvT_) ( 0 ) = mMat_ ( 0, 3 );
    (*pvT_) ( 1 ) = -mMat_ ( 1, 3 );
    (*pvT_) ( 2 ) = -mMat_ ( 2, 3 );

    return;
}

template< class T >
Eigen::Matrix< T, 4, 4 > setModelViewGLfromPrj(const Eigen::Transform<T, 3, Eigen::Affine> & Prj_)
{
	// column first for pGLMat_[16];
	// row first for Matrix3d;
	// pGLMat_[ 0] =  mR_(0,0); pGLMat_[ 4] =  mR_(0,1); pGLMat_[ 8] =  mR_(0,2); pGLMat_[12] =  vT_(0);
	// pGLMat_[ 1] = -mR_(1,0); pGLMat_[ 5] = -mR_(1,1); pGLMat_[ 9] = -mR_(1,2); pGLMat_[13] = -vT_(1);
	// pGLMat_[ 2] = -mR_(2,0); pGLMat_[ 6] = -mR_(2,1); pGLMat_[10] = -mR_(2,2); pGLMat_[14] = -vT_(2);
	// pGLMat_[ 3] =  0;        pGLMat_[ 7] =  0;        pGLMat_[11] =  0;        pGLMat_[15] = 1;

	Eigen::Matrix< T , 4, 4 > mMat;
	mMat.row( 0 ) =  Prj_.matrix().row( 0 );
	mMat.row( 1 ) = -Prj_.matrix().row( 1 );
	mMat.row( 2 ) = -Prj_.matrix().row( 2 );
	mMat.row( 3 ) =  Prj_.matrix().row( 3 );

	return mMat;
}

template< class T >
Eigen::Matrix< T , 4, 4 > setModelViewGLfromRTCV ( const Eigen::Matrix< T, 3, 3 >& mR_, const Eigen::Matrix< T, 3, 1 >& vT_ )
{
    // column first for pGLMat_[16];
    // row first for Matrix3d;
    // pGLMat_[ 0] =  mR_(0,0); pGLMat_[ 4] =  mR_(0,1); pGLMat_[ 8] =  mR_(0,2); pGLMat_[12] =  vT_(0);
    // pGLMat_[ 1] = -mR_(1,0); pGLMat_[ 5] = -mR_(1,1); pGLMat_[ 9] = -mR_(1,2); pGLMat_[13] = -vT_(1);
    // pGLMat_[ 2] = -mR_(2,0); pGLMat_[ 6] = -mR_(2,1); pGLMat_[10] = -mR_(2,2); pGLMat_[14] = -vT_(2);
    // pGLMat_[ 3] =  0;        pGLMat_[ 7] =  0;        pGLMat_[11] =  0;        pGLMat_[15] = 1;

    Eigen::Matrix< T , 4, 4 > mMat;
    mMat ( 0, 0 ) =  mR_ ( 0, 0 ); mMat ( 0, 1 ) =  mR_ ( 0, 1 ); mMat ( 0, 2 ) =  mR_ ( 0, 2 ); mMat ( 0, 3 ) =  vT_ ( 0 );
    mMat ( 1, 0 ) = -mR_ ( 1, 0 ); mMat ( 1, 1 ) = -mR_ ( 1, 1 ); mMat ( 1, 2 ) = -mR_ ( 1, 2 ); mMat ( 1, 3 ) = -vT_ ( 1 );
    mMat ( 2, 0 ) = -mR_ ( 2, 0 ); mMat ( 2, 1 ) = -mR_ ( 2, 1 ); mMat ( 2, 2 ) = -mR_ ( 2, 2 ); mMat ( 2, 3 ) = -vT_ ( 2 );
    mMat ( 3, 0 ) =  0;            mMat ( 3, 1 ) =  0;            mMat ( 3, 2 ) =  0;            mMat ( 3, 3 ) =  1;
    
    return mMat;
}

template< class T >
Eigen::Matrix< T , 4, 4 > setModelViewGLfromRCCV ( const Eigen::Matrix< T, 3, 3 >& mR_, const Eigen::Matrix< T, 3, 1 >& vC_ )
{
	Eigen::Matrix< T, 3,1> eivT = -mR_.transpose()*vC_;
	return setModelViewGLfromRTCV(mR_,vC_);
}

template< class T1, class T2 >
void unprojectCamera2World ( const int& nX_, const int& nY_, const unsigned short& nD_, const Eigen::Matrix< T1, 3, 3 >& mK_, Eigen::Matrix< T2, 3, 1 >* pVec_ )
{
	//the pixel coordinate is defined w.r.t. opencv camera reference, which is defined as x-right, y-downward and z-forward. It's
	//a right hand system.
	//when rendering the point using opengl's camera reference which is defined as x-right, y-upward and z-backward. the
	//	glVertex3d ( Pt(0), -Pt(1), -Pt(2) );
	if ( nD_ > 400 ) {
		T2 dZ = nD_ / 1000.; //convert to meter
		T2 dX = ( nX_ - mK_ ( 0, 2 ) ) / mK_ ( 0, 0 ) * dZ;
		T2 dY = ( nY_ - mK_ ( 1, 2 ) ) / mK_ ( 1, 1 ) * dZ;
		( *pVec_ ) << dX + 0.0025, dY, dZ + 0.00499814; // the value is esimated using CCalibrateKinectExtrinsics::calibDepth()
		// 0.0025 by experience.
	}
	else {
		( *pVec_ ) << 0, 0, 0;
	}
}

template< class T >
void projectWorld2Camera ( const Eigen::Matrix< T, 3, 1 >& vPt_, const Eigen::Matrix3d& mK_, Eigen::Matrix< short, 2, 1>* pVec_  )
{
	// this is much faster than the function
	// eiv2DPt = mK * vPt; eiv2DPt /= eiv2DPt(2);
	( *pVec_ ) ( 0 ) = short ( mK_ ( 0, 0 ) * vPt_ ( 0 ) / vPt_ ( 2 ) + mK_ ( 0, 2 ) + 0.5 );
	( *pVec_ ) ( 1 ) = short ( mK_ ( 1, 1 ) * vPt_ ( 1 ) / vPt_ ( 2 ) + mK_ ( 1, 2 ) + 0.5 );
}

template< class T >
void convertPrj2Rnt(const Eigen::Transform< T, 3, Eigen::Affine >& Prj_, Eigen::Matrix< T, 3, 3 >* pR_, Eigen::Matrix< T, 3, 1 >* pT_)
{
	*pR_ = Prj_.linear();
	*pT_ = Prj_.translation();
	return;
}
template< class T >
Eigen::Transform< T, 3, Eigen::Affine > convertRnt2Prj(const Eigen::Matrix< T, 3, 3 >& R_, const Eigen::Matrix< T, 3, 1 >& T_)
{
	Eigen::Transform< T, 3, Eigen::Affine > prj;
	prj.setIdentity();
	prj.linear() = R_;
	prj.translation() = T_;
	return prj;
}
template< class T >
void convertPrjInv2RpnC( const Eigen::Matrix< T, 4, 4 >& Prj_, Eigen::Matrix< T, 3, 3 >* pR_trans_, Eigen::Matrix< T, 3, 1 >* pT_)
{
	*pR_trans_ = Prj_.template block<3,3>(0,0);
	*pT_ = Prj_.template block<3,1>(0,3);
	return;
}
template< class T >
Eigen::Matrix< T, 4, 4 > convertRpnC2PrjInv(  const Eigen::Matrix< T, 3, 3 >& R_trans_, const Eigen::Matrix< T, 3, 1 >& C_ )
{
	Eigen::Matrix< T, 4, 4 > prj;
	prj.setIdentity();
	prj.template block<3,3>(0,0) = R_trans_;
	prj.template block<3,1>(0,3) = C_;
	return prj;
}

template< class T > /*Eigen::Matrix<float,-1,-1,0,-1,-1> = Eigen::MatrixXf*/
T aoHorn ( const Eigen::Matrix<T,-1,-1,0,-1,-1> & eimXw_, const Eigen::Matrix<T,-1,-1,0,-1,-1>&  eimXc_, const double dThreSecondIter_, const T dDegree_,
 					   bool bEstimateScale_, Eigen::Matrix< T, 3, 3>* pRw_, Eigen::Matrix< T , 3, 1 >* pTw_, T* pdScale_, cv::Mat* pInliers_ ){
	// A is Ref B is Cur
	// eimB_ = R * eimA_ + T;
	// main references: http://www.mathworks.com/matlabcentral/fileexchange/22422-absolute-orientation
	//CHECK ( 	eimXw_.rows() == 3, " absoluteOrientation() requires the input matrix A_ is a 3 x N matrix. " );
	//CHECK ( 	eimXc_.rows() == 3, " absoluteOrientation() requires the input matrix B_ is a 3 x N matrix. " );
	//CHECK ( 	eimXw_.cols() == eimXc_.cols(), " absoluteOrientation() requires the columns of input matrix A_ and B_ are equal. " );

	//Compute the centroid of each point set

	Eigen::Matrix<T,3,1> eivCentroidA(0,0,0), eivCentroidB(0,0,0); //Matrix<float,3,1,0,3,1> = Vector3f
	for ( int nCount = 0; nCount < eimXw_.cols(); nCount++ ){
		eivCentroidA += eimXw_.col ( nCount );
		eivCentroidB += eimXc_.col ( nCount );
	}
	eivCentroidA /= (float)eimXw_.cols();
	eivCentroidB /= (float)eimXw_.cols();
	//PRINT(eimA_.cols());
	//PRINT( nCount );
	//PRINT( eivCentroidA );
	//PRINT( eivCentroidB );

	//Remove the centroid
	float fSquaredSin = sinf(float(M_PI_4)/45.f*dDegree_); 	fSquaredSin *= fSquaredSin; 
	Eigen::MatrixXi eimMask ( 1, eimXw_.cols() );
	cv::Mat inliers(1,(int)eimXw_.cols(),CV_32SC1);
	Eigen::Vector3f eimD;
	Eigen::Matrix<T,-1,-1,0,-1,-1> An ( 3, eimXw_.cols() ), Bn ( 3, eimXw_.cols() );
	int nInliers1 = 0;
	for ( int nC = 0; nC < eimXw_.cols(); nC++ ){
		An.col ( nC ) = eimXw_.col ( nC ) - eivCentroidA;
		Bn.col ( nC ) = eimXc_.col ( nC ) - eivCentroidB;

		eimD = An.col ( nC ) - Bn.col ( nC );
		float fDist = eimD.squaredNorm();
		float fDistThresh = eimXc_.col(nC).squaredNorm() * fSquaredSin;
		if ( fDist < fDistThresh ){
			inliers.at<int>(0,nInliers1) = nC;
			nInliers1++;
		}
	}

	cv::Mat mask; inliers.colRange(0,nInliers1).copyTo(mask);
	//PRINT(nInliers1);
	//PRINT( An );
	//PRINT( Bn );
	T dE = 0; // final energy
	int nTotalInlier = -1;
	for(int nIter=0; nIter < 2; nIter++){
		if ( nTotalInlier == 0)	{ break; }
		nTotalInlier = 0;
		//Recompute centroid and 
		eivCentroidA.setZero();
		eivCentroidB.setZero();
		for( int i = 0; i< mask.cols; i++){
			int idx = mask.ptr<int>()[i];
			eivCentroidA += eimXw_.col ( idx );
			eivCentroidB += eimXc_.col ( idx );
		}
		eivCentroidA /= (float)mask.cols;
		eivCentroidB /= (float)mask.cols;
		// transform coordinate
		for( int i = 0; i< mask.cols; i++){
			int idx = mask.ptr<int>()[i];
			An.col ( idx ) = eimXw_.col ( idx ) - eivCentroidA;
			Bn.col ( idx ) = eimXc_.col ( idx ) - eivCentroidB;
		}

		//Compute the quaternions
		Eigen::Matrix<T,4,4> M; M.setZero();
		Eigen::Matrix<T,4,4> Ma, Mb;
		for( int i = 0; i< mask.cols; i++){
			int nC = mask.ptr<int>()[i];
			//pure imaginary Shortcuts
			Eigen::Matrix<T,4,1> a(0,0,0,0), b(0,0,0,0);
			a ( 1 ) = An ( 0, nC );
			a ( 2 ) = An ( 1, nC );
			a ( 3 ) = An ( 2, nC );
			b ( 1 ) = Bn ( 0, nC );
			b ( 2 ) = Bn ( 1, nC );
			b ( 3 ) = Bn ( 2, nC );
			//cross products
			Ma << a ( 0 ), -a ( 1 ), -a ( 2 ), -a ( 3 ),
				a ( 1 ),  a ( 0 ),  a ( 3 ), -a ( 2 ),
				a ( 2 ), -a ( 3 ),  a ( 0 ),  a ( 1 ),
				a ( 3 ),  a ( 2 ), -a ( 1 ),  a ( 0 );
			Mb << b ( 0 ), -b ( 1 ), -b ( 2 ), -b ( 3 ),
				b ( 1 ),  b ( 0 ), -b ( 3 ),  b ( 2 ),
				b ( 2 ),  b ( 3 ),  b ( 0 ), -b ( 1 ),
				b ( 3 ), -b ( 2 ),  b ( 1 ),  b ( 0 );
			//Add up
			M += Ma.transpose() * Mb;
		}

		Eigen::EigenSolver <Eigen::Matrix<T,4,4> > eigensolver ( M );
		Eigen::Matrix< std::complex< T >, 4, 1 > v = eigensolver.eigenvalues();

		//find the largest eigenvalue;
		float dLargest = -1000000;
		int n;

		for ( int i = 0; i < 4; i++ ) {
			if ( dLargest < v ( i ).real() ) {
				dLargest = v ( i ).real();
				n = i;
			}
		}

		//PRINT( dLargest );
		//PRINT( n );

		Eigen::Matrix<T,4,1> e;
		e << eigensolver.eigenvectors().col ( n ) ( 0 ).real(),
			 eigensolver.eigenvectors().col ( n ) ( 1 ).real(),
			 eigensolver.eigenvectors().col ( n ) ( 2 ).real(),
			 eigensolver.eigenvectors().col ( n ) ( 3 ).real();

		//PRINT( e );

		Eigen::Matrix<T,4,4>M1, M2, R;
		//Compute the rotation matrix
		M1 <<  e ( 0 ), -e ( 1 ), -e ( 2 ), -e ( 3 ),
			   e ( 1 ),  e ( 0 ),  e ( 3 ), -e ( 2 ),
			   e ( 2 ), -e ( 3 ),  e ( 0 ),  e ( 1 ),
			   e ( 3 ),  e ( 2 ), -e ( 1 ),  e ( 0 );
		M2 <<  e ( 0 ), -e ( 1 ), -e ( 2 ), -e ( 3 ),
			   e ( 1 ),  e ( 0 ), -e ( 3 ),  e ( 2 ),
			   e ( 2 ),  e ( 3 ),  e ( 0 ), -e ( 1 ),
			   e ( 3 ), -e ( 2 ),  e ( 1 ),  e ( 0 );
		R = M1.transpose() * M2;
		( *pRw_ ) = R.block ( 1, 1, 3, 3 );

		//Compute the scale factor if necessary
		if ( bEstimateScale_ ){
			T a = 0, b = 0;
			for( int i = 0; i< mask.cols; i++){
				int nC = mask.ptr<int>()[i];
				a += Bn.col ( nC ).transpose() * ( *pRw_ ) * An.col ( nC );
				b += Bn.col ( nC ).transpose() * Bn.col ( nC );
			}
			//PRINT( a );
			//PRINT( b );
			( *pdScale_ ) = b / a;
		}
		else{
			( *pdScale_ ) = 1;
		}
		//Compute the final translation
		( *pTw_ ) = eivCentroidB - ( *pdScale_ ) * ( *pRw_ ) * eivCentroidA;

		//Compute the residual error
		if(dThreSecondIter_ < 0.) break;
		dE = 0;
		Eigen::Matrix<T,3,1> eivE;
		//recollect inliers
		nTotalInlier = 0;
		for ( int nC = 0; nC < eimXc_.cols(); nC++ ) {
			eivE = eimXc_.col ( nC ) - ( ( *pdScale_ ) * ( *pRw_ ) * eimXw_.col ( nC ) + ( *pTw_ ) );
			T dEN = eivE.norm();
			if( dEN > dThreSecondIter_ ) {
				eimMask(nC) = 0;
			}
			else{
				inliers.at<int>(0,nTotalInlier) = nC;
				nTotalInlier ++;
				dE += dEN;
			}
		}
		inliers.colRange(0,nTotalInlier).copyTo(mask);
		//PRINT(nTotalInlier);
		dE /= nTotalInlier;
	}//for iterations
	mask.copyTo(*pInliers_);
	PRINT(nTotalInlier);
	return dE;
}


enum { 
	HORN = 1, // Horn, B. K. P. (1987). Closed-form solution of absolute orientation using unit quaternions. Journal of the Optical Society of America A, 6(4), 422. doi:10.1364/JOSAA.4.000629
	ARUN = 2  // Arun, S. K., Huang, T. S., & Blostein, S. D. (1987). Least-Squares Fitting of Two 3-D Point Sets. In PAMI (pp. 698\96700).
};
//more stable than absoluteOirentationSimple() when the # of pairs are small. It requires minimum 3 non-linear pairs of points to get R,t
//Xw = Rw * Xc + tw; <==> A = Rw * B + tw;
//absoluteOirentationSimple() requires 4. 
template< class T > /*Eigen::Matrix<float,-1,-1,0,-1,-1> = Eigen::MatrixXf*/
T aoHorn ( const Eigen::Matrix<T,-1,-1,0,-1,-1> & eimXw_, const Eigen::Matrix<T,-1,-1,0,-1,-1>&  eimXc_, Eigen::Matrix< T, 3, 3>* pRw_, Eigen::Matrix< T , 3, 1 >* pTw_ ){
	// A is Ref B is Cur
	// eimB_ = R * eimA_ + T;
	// main references: http://www.mathworks.com/matlabcentral/fileexchange/22422-absolute-orientation
	//CHECK ( 	eimXw_.rows() == 3, " absoluteOrientation() requires the input matrix A_ is a 3 x N matrix. " );
	//CHECK ( 	eimXc_.rows() == 3, " absoluteOrientation() requires the input matrix B_ is a 3 x N matrix. " );
	//CHECK ( 	eimXw_.cols() == eimXc_.cols(), " absoluteOrientation() requires the columns of input matrix A_ and B_ are equal. " );

	//Compute the centroid of each point set
	Eigen::Matrix<T,3,1> eivCentroidA(0,0,0), eivCentroidB(0,0,0); //Matrix<float,3,1,0,3,1> = Vector3f
	for ( int nCount = 0; nCount < eimXw_.cols(); nCount++ ){
		eivCentroidA += eimXw_.col ( nCount );
		eivCentroidB += eimXc_.col ( nCount );
	}
	eivCentroidA /= (float)eimXw_.cols();
	eivCentroidB /= (float)eimXw_.cols();
	//PRINT(eimA_.cols());
	//PRINT( nCount );
	//PRINT( eivCentroidA );
	//PRINT( eivCentroidB );
	T dE = 0; // final energy
	int nTotalInlier = 0;
		
	// transform coordinate
	Eigen::Matrix<T,-1,-1,0,-1,-1> An ( 3, eimXw_.cols() ), Bn ( 3, eimXw_.cols() );
	for ( int nC = 0; nC < eimXw_.cols(); nC++ ){
		An.col ( nC ) = eimXw_.col ( nC ) - eivCentroidA;
		Bn.col ( nC ) = eimXc_.col ( nC ) - eivCentroidB;
	}

	//Compute the quaternions
	Eigen::Matrix<T,4,4> M; M.setZero();
	Eigen::Matrix<T,4,4> Ma, Mb;
	for ( int nC = 0; nC < eimXw_.cols(); nC++ ){
		//pure imaginary Shortcuts
		Eigen::Matrix<T,4,1> a(0,0,0,0), b(0,0,0,0);
		a ( 1 ) = An ( 0, nC );
		a ( 2 ) = An ( 1, nC );
		a ( 3 ) = An ( 2, nC );
		b ( 1 ) = Bn ( 0, nC );
		b ( 2 ) = Bn ( 1, nC );
		b ( 3 ) = Bn ( 2, nC );
		//cross products
		Ma << a ( 0 ), -a ( 1 ), -a ( 2 ), -a ( 3 ),
			a ( 1 ),  a ( 0 ),  a ( 3 ), -a ( 2 ),
			a ( 2 ), -a ( 3 ),  a ( 0 ),  a ( 1 ),
			a ( 3 ),  a ( 2 ), -a ( 1 ),  a ( 0 );
		Mb << b ( 0 ), -b ( 1 ), -b ( 2 ), -b ( 3 ),
			b ( 1 ),  b ( 0 ), -b ( 3 ),  b ( 2 ),
			b ( 2 ),  b ( 3 ),  b ( 0 ), -b ( 1 ),
			b ( 3 ), -b ( 2 ),  b ( 1 ),  b ( 0 );
		//Add up
		M += Ma.transpose() * Mb;
	}

	Eigen::EigenSolver <Eigen::Matrix<T,4,4> > eigensolver ( M );
	Eigen::Matrix< std::complex< T >, 4, 1 > v = eigensolver.eigenvalues();

	//find the largest eigenvalue;
	float dLargest = -1000000;
	int n;

	for ( int i = 0; i < 4; i++ ) {
		if ( dLargest < v ( i ).real() ) {
			dLargest = v ( i ).real();
			n = i;
		}
	}

	//PRINT( dLargest );
	//PRINT( n );

	Eigen::Matrix<T,4,1> e;
	e <<eigensolver.eigenvectors().col ( n ) ( 0 ).real(),
		eigensolver.eigenvectors().col ( n ) ( 1 ).real(),
		eigensolver.eigenvectors().col ( n ) ( 2 ).real(),
		eigensolver.eigenvectors().col ( n ) ( 3 ).real();

	//PRINT( e );

	Eigen::Matrix<T,4,4>M1, M2, R;
	//Compute the rotation matrix
	M1 <<  e ( 0 ), -e ( 1 ), -e ( 2 ), -e ( 3 ),
			e ( 1 ),  e ( 0 ),  e ( 3 ), -e ( 2 ),
			e ( 2 ), -e ( 3 ),  e ( 0 ),  e ( 1 ),
			e ( 3 ),  e ( 2 ), -e ( 1 ),  e ( 0 );
	M2 <<  e ( 0 ), -e ( 1 ), -e ( 2 ), -e ( 3 ),
			e ( 1 ),  e ( 0 ), -e ( 3 ),  e ( 2 ),
			e ( 2 ),  e ( 3 ),  e ( 0 ), -e ( 1 ),
			e ( 3 ), -e ( 2 ),  e ( 1 ),  e ( 0 );
	R = M1.transpose() * M2;
	Eigen::Matrix< T, 3, 3> R_tmp;
	R_tmp = R.block ( 1, 1, 3, 3 );

		
	//Compute the final translation
	Eigen::Matrix< T, 3, 1> T_tmp;
	T_tmp = eivCentroidB - R_tmp * eivCentroidA;

	//Compute the residual error
	dE = 0;
	Eigen::Matrix<T,3,1> eivE;
	//recollect inliers
	for ( int nC = 0; nC < eimXw_.cols(); nC++ ) {
		eivE = eimXc_.col ( nC ) - ( R_tmp * eimXw_.col ( nC ) + T_tmp );
		T dEN = eivE.norm();
		dE += dEN;
	}
	dE /= eimXw_.cols();

	( *pRw_ ) = R_tmp;
	( *pTw_ ) = T_tmp;
	return dE;
}

template< class T > /*Eigen::Matrix<float,-1,-1,0,-1,-1> = Eigen::MatrixXf*/
T aoArun ( const Eigen::Matrix<T,-1,-1,0,-1,-1> & eimXw_, const Eigen::Matrix<T,-1,-1,0,-1,-1>&  eimXc_, Eigen::Matrix< T, 3, 3>* pR_, Eigen::Matrix< T , 3, 1 >* pT_ ){
	// A is from World  B is Local coordinate system
	// eimXc_ = R * eimXw_ + T; //R and t is defined in world and transform a point in world to local
	// main references: http://www.mathworks.com/matlabcentral/fileexchange/22422-absolute-orientation
	using namespace Eigen;
	//CHECK ( 	eimXw_.rows() == 3, " absoluteOrientation() requires the input matrix A_ is a 3 x N matrix. " );
	//CHECK ( 	eimXc_.rows() == 3, " absoluteOrientation() requires the input matrix B_ is a 3 x N matrix. " );
	//CHECK ( 	eimXw_.cols() == eimXc_.cols(), " absoluteOrientation() requires the columns of input matrix A_ and B_ are equal. " );

	//Compute the centroid of each point set
	Eigen::Matrix<double, 3,1> eivCentroidW(0,0,0), eivCentroidC(0,0,0); //Matrix<float,3,1,0,3,1> = Vector3f
	for ( int nCount = 0; nCount < eimXw_.cols(); nCount++ ){
		eivCentroidW += eimXw_.col ( nCount ).template cast<double>();
		eivCentroidC += eimXc_.col ( nCount ).template cast<double>();
	}
	eivCentroidW /= (double)eimXw_.cols();
	eivCentroidC /= (double)eimXw_.cols();
	//PRINT(eimA_.cols());
	//PRINT( nCount );
	//PRINT( eivCentroidA );
	//PRINT( eivCentroidB );

	// transform coordinate
	Eigen::Matrix<T,-1,-1,0,-1,-1> Aw ( 3, eimXw_.cols() ), Ac ( 3, eimXw_.cols() );
	for ( int nC = 0; nC < eimXw_.cols(); nC++ ){
		Aw.col ( nC ) = eimXw_.col ( nC ) - eivCentroidW.template cast<float>();
		Ac.col ( nC ) = eimXc_.col ( nC ) - eivCentroidC.template cast<float>();
	}

	//Compute the quaternions
	Eigen::Matrix<double,3,3> M; M.setZero();
	Eigen::Matrix<T,3,3> N;
	//M=M+x3d_c(i,:)'*x3d_w(i,:);
	for ( int nC = 0; nC < eimXw_.cols(); nC++ ){
		//calc the N = x3d_c(i,:)'*x3d_w(i,:);
		N = Ac.col( nC ) * Aw.col( nC ).transpose();
		M += N.template cast<double>();
	}
	
	JacobiSVD<Eigen::Matrix<double,-1,-1,0,-1,-1> > svd(M, ComputeFullU | ComputeFullV );
	//[U S V]=svd(M);
	//R=U*V';
	Matrix<double, 3, 3> U = svd.matrixU();
	Matrix<double, 3, 3> V = svd.matrixV();
	Matrix<double, 3, 3> R_tmp;
	if( U.determinant()*V.determinant() < 0){
		Matrix<double, 3, 3> I = Matrix<double, 3, 3>::Identity(); I(2, 2) = -1;
		R_tmp = U*I*V.transpose();
	}
	else{
		R_tmp = U*V.transpose();
	}
	//T=ccent'-R*wcent';
	Eigen::Matrix< double, 3, 1> T_tmp = eivCentroidC - R_tmp * eivCentroidW;

	( *pR_ ) = R_tmp.template cast<T>();
	( *pT_ ) = T_tmp.template cast<T>();
	//recollect inliers
	T dE = 0; // final energy
	for ( int nC = 0; nC < eimXw_.cols(); nC++ ) {
		Eigen::Matrix< T, 3, 1> eivE = eimXc_.col ( nC ) - ( ( *pR_ ) * eimXw_.col ( nC ) + ( *pT_ ) );
		dE += eivE.norm();
	}
	dE /= eimXw_.cols();
	
	return dE;
}

template< class T > /*Eigen::Matrix<float,-1,-1,0,-1,-1> = Eigen::MatrixXf*/
T aoRansac ( const Eigen::Matrix<T,-1,-1,0,-1,-1> & eimXw_, const Eigen::Matrix<T,-1,-1,0,-1,-1>&  eimXc_, 
							  const T dist_thre_, const int Iter,
							  Eigen::Matrix< T, 3, 3>* pR_, Eigen::Matrix< T , 3, 1 >* pT_, cv::Mat * pInliers_, int nMethod_ = HORN ){
	// A is from World  B is Local coordinate system
	// eimB_ = R * eimA_ + T; //R and t is defined in world and transform a point in world to local
	// main references: http://www.mathworks.com/matlabcentral/fileexchange/22422-absolute-orientation
	using namespace Eigen;
	using namespace cv;
	//CHECK (	eimXw_.rows() == 3, " absoluteOrientation() requires the input matrix A_ is a 3 x N matrix. " );
	//CHECK (	eimXc_.rows() == 3, " absoluteOrientation() requires the input matrix B_ is a 3 x N matrix. " );
	//CHECK (	eimXw_.cols() == eimXc_.cols(), " absoluteOrientation() requires the columns of input matrix A_ and B_ are equal. " );

	RandomElements<int> re( (int)eimXw_.cols() );
	const int K = 3;
	vector<vector<int> > _v_v_inlier_idx;

	cv::Mat all_votes(1,Iter,CV_32FC1);
	for (int i=0; i < Iter; i++)	{
		//randomly select K candidates
		vector<int> selected_cols;
		re.run(K, &selected_cols);
		Matrix<T,-1,-1,0,-1,-1> eimX_world(3,K),eimX_cam(3,K);
		for (int c=0; c<K; c++) {
			eimX_world.col( c ) = eimXw_.col( selected_cols[c] );
			eimX_cam.col( c )   = eimXc_.col( selected_cols[c] );
		}
		//calc R&t		
		Eigen::Matrix< T, 3, 3> R_tmp;
		Eigen::Matrix< T, 3, 1> T_tmp;
		if( nMethod_ == HORN )
			aoHorn<T>(eimX_world,eimX_cam,&R_tmp,&T_tmp);
		else
			aoArun<T>(eimX_world,eimX_cam,&R_tmp,&T_tmp);
		//collect votes
		int nVotes = 0;
		vector<int> inlier_idx;
		for (int c=0; c < eimXw_.cols(); c++  )	{
			Eigen::Matrix< T, 3, 1> eivE = eimXc_.col ( c ) - ( R_tmp * eimXw_.col ( c ) + T_tmp );
			if ( eivE.norm() < dist_thre_ ){
				inlier_idx.push_back( c );
				nVotes ++;
			}
		}
		all_votes.at<int>( 0,i ) = nVotes;
		_v_v_inlier_idx.push_back( inlier_idx );
	}
	cv::Mat most;
	sortIdx(all_votes,most, CV_SORT_DESCENDING + CV_SORT_EVERY_ROW );
	//collect inliers
	////////////////////////////////////////////////////////////////
	int best_idx = most.at<int>(0,0);
	cout << ( all_votes.at<int>( 0,best_idx) ) << endl;
	vector<int>& v_inlier_idx = _v_v_inlier_idx[best_idx];
	pInliers_->create(1,(int)v_inlier_idx.size(),CV_32SC1); 
	Matrix<T,-1,-1,0,-1,-1> eimX_world_selected( 3,v_inlier_idx.size() ),    eimX_cam_selected(3,v_inlier_idx.size());
	
	for ( int i=0; i< v_inlier_idx.size(); i++)
	{
		eimX_world_selected.col(i) = eimXw_.col( v_inlier_idx[i] );
		eimX_cam_selected.col(i) = eimXc_.col( v_inlier_idx[i] );
		pInliers_->ptr<int>()[i] = v_inlier_idx[i];
	}

	Eigen::Matrix< T, 3, 3> R_tmp;
	Eigen::Matrix< T, 3, 1> T_tmp;
	T dE;
	if( nMethod_ == HORN )
		dE = aoHorn<T>( eimX_world_selected, eimX_cam_selected, &R_tmp, &T_tmp);
	else
		dE = aoArun<T>( eimX_world_selected, eimX_cam_selected, &R_tmp, &T_tmp);

	*pR_ = R_tmp;
	*pT_ = T_tmp;

	return dE;
}

template< class T > /*Eigen::Matrix<float,-1,-1,0,-1,-1> = Eigen::MatrixXf*/
Eigen::Matrix<T,2,1> absoluteOrientationWithNormal ( const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimXw_, const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimNlw_, 
													 const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimXc_, const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimNlc_, Eigen::Matrix< T, 3, 3>* pRw_, Eigen::Matrix< T , 3, 1 >* pTw_ ){
	using namespace Eigen;
	// A is Ref B is Cur
	// eimB_ = R * eimA_ + T;
	// main references: http://www.mathworks.com/matlabcentral/fileexchange/22422-absolute-orientation
	//CHECK ( 	eimXw_.rows() == 3, " absoluteOrientation() requires the input matrix A_ is a 3 x N matrix. " );
	//CHECK ( 	eimXc_.rows() == 3, " absoluteOrientation() requires the input matrix B_ is a 3 x N matrix. " );
	//CHECK ( 	eimXw_.cols() == eimXc_.cols(), " absoluteOrientation() requires the columns of input matrix A_ and B_ are equal. " );

	//Compute the centroid of each point set
	Eigen::Matrix<T,3,1> eivCentroidA(0,0,0), eivCentroidB(0,0,0); //Matrix<float,3,1,0,3,1> = Vector3f
	for ( int nCount = 0; nCount < eimXw_.cols(); nCount++ ){
		eivCentroidA += eimXw_.col ( nCount );
		eivCentroidB += eimXc_.col ( nCount );
	}
	eivCentroidA /= (float)eimXw_.cols();
	eivCentroidB /= (float)eimXw_.cols();
	//PRINT(eimA_.cols());
	//PRINT( nCount );
	//PRINT( eivCentroidA );
	//PRINT( eivCentroidB );
	int nTotalInlier = 0;

	// transform coordinate
	Eigen::Matrix<T,-1,-1,0,-1,-1> An ( 3, eimXw_.cols() ), Bn ( 3, eimXw_.cols() );
	for ( int nC = 0; nC < eimXw_.cols(); nC++ ){
		An.col ( nC ) = eimXw_.col ( nC ) - eivCentroidA;
		Bn.col ( nC ) = eimXc_.col ( nC ) - eivCentroidB;
	}

	//Compute the quaternions
	Eigen::Matrix<T,4,4> M; M.setZero();
	Eigen::Matrix<T,4,4> Ma, Mb;
	T forNormalisation = 0.f;
	for ( int nC = 0; nC < eimXw_.cols(); nC++ ){
		//pure imaginary Shortcuts
		Eigen::Matrix<T,4,1> a(0,0,0,0), b(0,0,0,0);
		a ( 1 ) = An ( 0, nC );
		a ( 2 ) = An ( 1, nC );
		a ( 3 ) = An ( 2, nC );
		b ( 1 ) = Bn ( 0, nC );
		b ( 2 ) = Bn ( 1, nC );
		b ( 3 ) = Bn ( 2, nC );
		forNormalisation += An.col( nC ).norm(); 
		forNormalisation += Bn.col( nC ).norm(); 
		//cross products
		Ma << a ( 0 ), -a ( 1 ), -a ( 2 ), -a ( 3 ),
			a ( 1 ),  a ( 0 ),  a ( 3 ), -a ( 2 ),
			a ( 2 ), -a ( 3 ),  a ( 0 ),  a ( 1 ),
			a ( 3 ),  a ( 2 ), -a ( 1 ),  a ( 0 );
		Mb << b ( 0 ), -b ( 1 ), -b ( 2 ), -b ( 3 ),
			b ( 1 ),  b ( 0 ), -b ( 3 ),  b ( 2 ),
			b ( 2 ),  b ( 3 ),  b ( 0 ), -b ( 1 ),
			b ( 3 ), -b ( 2 ),  b ( 1 ),  b ( 0 );
		//Add up
		M += Ma.transpose() * Mb;
	}

	forNormalisation /= ( An.cols() * 2 );
	//PRINT( forNormalisation );
	for ( int nC = 0; nC < eimNlw_.cols(); nC++ ){
		//pure imaginary Shortcuts
		Eigen::Matrix<T,4,1> a(0,0,0,0), b(0,0,0,0);
		a ( 1 ) = eimNlw_ ( 0, nC );
		a ( 2 ) = eimNlw_ ( 1, nC );
		a ( 3 ) = eimNlw_ ( 2, nC );
		a *= forNormalisation;
		b ( 1 ) = eimNlc_ ( 0, nC );
		b ( 2 ) = eimNlc_ ( 1, nC );
		b ( 3 ) = eimNlc_ ( 2, nC );
		b *= forNormalisation;
		//cross products
		Ma << a ( 0 ), -a ( 1 ), -a ( 2 ), -a ( 3 ),
			a ( 1 ),  a ( 0 ),  a ( 3 ), -a ( 2 ),
			a ( 2 ), -a ( 3 ),  a ( 0 ),  a ( 1 ),
			a ( 3 ),  a ( 2 ), -a ( 1 ),  a ( 0 );
		Mb << b ( 0 ), -b ( 1 ), -b ( 2 ), -b ( 3 ),
			b ( 1 ),  b ( 0 ), -b ( 3 ),  b ( 2 ),
			b ( 2 ),  b ( 3 ),  b ( 0 ), -b ( 1 ),
			b ( 3 ), -b ( 2 ),  b ( 1 ),  b ( 0 );
		//Add up
		M += Ma.transpose() * Mb;
	}

	Eigen::EigenSolver <Eigen::Matrix<T,4,4> > eigensolver ( M );
	Eigen::Matrix< std::complex< T >, 4, 1 > v = eigensolver.eigenvalues();

	//find the largest eigenvalue;
	float dLargest = -1000000;
	int n;

	for ( int i = 0; i < 4; i++ ) {
		if ( dLargest < v ( i ).real() ) {
			dLargest = v ( i ).real();
			n = i;
		}
	}

	//PRINT( dLargest );
	//PRINT( n );

	Eigen::Matrix<T,4,1> e;
	e <<eigensolver.eigenvectors().col ( n ) ( 0 ).real(),
		eigensolver.eigenvectors().col ( n ) ( 1 ).real(),
		eigensolver.eigenvectors().col ( n ) ( 2 ).real(),
		eigensolver.eigenvectors().col ( n ) ( 3 ).real();

	//PRINT( e );

	Eigen::Matrix<T,4,4>M1, M2, R;
	//Compute the rotation matrix
	M1 <<  e ( 0 ), -e ( 1 ), -e ( 2 ), -e ( 3 ),
		e ( 1 ),  e ( 0 ),  e ( 3 ), -e ( 2 ),
		e ( 2 ), -e ( 3 ),  e ( 0 ),  e ( 1 ),
		e ( 3 ),  e ( 2 ), -e ( 1 ),  e ( 0 );
	M2 <<  e ( 0 ), -e ( 1 ), -e ( 2 ), -e ( 3 ),
		e ( 1 ),  e ( 0 ), -e ( 3 ),  e ( 2 ),
		e ( 2 ),  e ( 3 ),  e ( 0 ), -e ( 1 ),
		e ( 3 ), -e ( 2 ),  e ( 1 ),  e ( 0 );
	R = M1.transpose() * M2;
	Eigen::Matrix< T, 3, 3> R_tmp;
	R_tmp = R.block ( 1, 1, 3, 3 );


	//Compute the final translation
	Eigen::Matrix< T, 3, 1> T_tmp;
	T_tmp = eivCentroidB - R_tmp * eivCentroidA;

	//Compute the residual error
	T dE = 0; // final energy
	Eigen::Matrix<T,3,1> eivE;

	//recollect inliers
	for ( int nC = 0; nC < eimXw_.cols(); nC++ ) {
		eivE = eimXc_.col ( nC ) - ( R_tmp * eimXw_.col ( nC ) + T_tmp );
		T dEN = eivE.norm();
		dE += dEN;
	}
	dE /= eimXw_.cols();

	T dA = 0;
	for ( int nC = 0; nC < eimNlw_.cols(); nC++ ) {
		T dEN = eimNlc_.col( nC ).dot( R_tmp * eimNlw_.col ( nC ) );
		dEN = dEN > 1.f ? 1.f : dEN;
		dEN = dEN <-1.f ?-1.f : dEN;
		dA += acos( dEN );
	}
	dA /= eimNlw_.cols();

	( *pRw_ ) = R_tmp;
	( *pTw_ ) = T_tmp;

	Matrix<T,2,1> eivEA( dE, dA );
	return eivEA;
}

template< class T > /*Eigen::Matrix<float,-1,-1,0,-1,-1> = Eigen::MatrixXf*/
Eigen::Matrix<T,2,1> absoluteOrientationWithNormalnMainDirection ( const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimXw_, const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimNlw_, const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimMDw_, 
											  const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimXc_, const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimNlc_, const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimMDc_, 
											  Eigen::Matrix< T, 3, 3>* pRw_, Eigen::Matrix< T , 3, 1 >* pTw_ ){
	using namespace Eigen;
	// A is Ref B is Cur
	// eimB_ = R * eimA_ + T;
	// main references: http://www.mathworks.com/matlabcentral/fileexchange/22422-absolute-orientation
	//CHECK ( 	eimXw_.rows() == 3, " absoluteOrientation() requires the input matrix A_ is a 3 x N matrix. " );
	//CHECK ( 	eimXc_.rows() == 3, " absoluteOrientation() requires the input matrix B_ is a 3 x N matrix. " );
	//CHECK ( 	eimXw_.cols() == eimXc_.cols(), " absoluteOrientation() requires the columns of input matrix A_ and B_ are equal. " );

	//Compute the centroid of each point set
	Eigen::Matrix<T,3,1> eivCentroidA(0,0,0), eivCentroidB(0,0,0); //Matrix<float,3,1,0,3,1> = Vector3f
	for ( int nCount = 0; nCount < eimXw_.cols(); nCount++ ){
		eivCentroidA += eimXw_.col ( nCount );
		eivCentroidB += eimXc_.col ( nCount );
	}
	eivCentroidA /= (float)eimXw_.cols();
	eivCentroidB /= (float)eimXw_.cols();
	//PRINT(eimA_.cols());
	//PRINT( nCount );
	//PRINT( eivCentroidA );
	//PRINT( eivCentroidB );
	int nTotalInlier = 0;

	// transform coordinate
	Eigen::Matrix<T,-1,-1,0,-1,-1> An ( 3, eimXw_.cols() ), Bn ( 3, eimXw_.cols() );
	for ( int nC = 0; nC < eimXw_.cols(); nC++ ){
		An.col ( nC ) = eimXw_.col ( nC ) - eivCentroidA;
		Bn.col ( nC ) = eimXc_.col ( nC ) - eivCentroidB;
	}

	//Compute the quaternions
	Eigen::Matrix<T,4,4> M; M.setZero();
	Eigen::Matrix<T,4,4> Ma, Mb;
	T forNormalisation = 0.f;
	for ( int nC = 0; nC < eimXw_.cols(); nC++ ){
		//pure imaginary Shortcuts
		Eigen::Matrix<T,4,1> a(0,0,0,0), b(0,0,0,0);
		a ( 1 ) = An ( 0, nC );
		a ( 2 ) = An ( 1, nC );
		a ( 3 ) = An ( 2, nC );
		b ( 1 ) = Bn ( 0, nC );
		b ( 2 ) = Bn ( 1, nC );
		b ( 3 ) = Bn ( 2, nC );
		forNormalisation += An.col( nC ).norm(); 
		forNormalisation += Bn.col( nC ).norm(); 
		//cross products
		Ma << a ( 0 ), -a ( 1 ), -a ( 2 ), -a ( 3 ),
			a ( 1 ),  a ( 0 ),  a ( 3 ), -a ( 2 ),
			a ( 2 ), -a ( 3 ),  a ( 0 ),  a ( 1 ),
			a ( 3 ),  a ( 2 ), -a ( 1 ),  a ( 0 );
		Mb << b ( 0 ), -b ( 1 ), -b ( 2 ), -b ( 3 ),
			b ( 1 ),  b ( 0 ), -b ( 3 ),  b ( 2 ),
			b ( 2 ),  b ( 3 ),  b ( 0 ), -b ( 1 ),
			b ( 3 ), -b ( 2 ),  b ( 1 ),  b ( 0 );
		//Add up
		M += Ma.transpose() * Mb;
	}

	forNormalisation /= ( An.cols() * 2 );
	//PRINT( forNormalisation );
	for ( int nC = 0; nC < eimNlw_.cols(); nC++ ){
		//pure imaginary Shortcuts
		Eigen::Matrix<T,4,1> a(0,0,0,0), b(0,0,0,0);
		a ( 1 ) = eimNlw_ ( 0, nC );
		a ( 2 ) = eimNlw_ ( 1, nC );
		a ( 3 ) = eimNlw_ ( 2, nC );
		a *= forNormalisation;
		b ( 1 ) = eimNlc_ ( 0, nC );
		b ( 2 ) = eimNlc_ ( 1, nC );
		b ( 3 ) = eimNlc_ ( 2, nC );
		b *= forNormalisation;
		//cross products
		Ma << a ( 0 ), -a ( 1 ), -a ( 2 ), -a ( 3 ),
			a ( 1 ),  a ( 0 ),  a ( 3 ), -a ( 2 ),
			a ( 2 ), -a ( 3 ),  a ( 0 ),  a ( 1 ),
			a ( 3 ),  a ( 2 ), -a ( 1 ),  a ( 0 );
		Mb << b ( 0 ), -b ( 1 ), -b ( 2 ), -b ( 3 ),
			b ( 1 ),  b ( 0 ), -b ( 3 ),  b ( 2 ),
			b ( 2 ),  b ( 3 ),  b ( 0 ), -b ( 1 ),
			b ( 3 ), -b ( 2 ),  b ( 1 ),  b ( 0 );
		//Add up
		M += Ma.transpose() * Mb;
	}
	float fWeightMD = 0;
	for ( int nC = 0; nC < eimMDw_.cols(); nC++ ){
		//pure imaginary Shortcuts
		Eigen::Matrix<T,4,1> a(0,0,0,0), b(0,0,0,0);
		a ( 1 ) = eimMDw_ ( 0, nC );
		a ( 2 ) = eimMDw_ ( 1, nC );
		a ( 3 ) = eimMDw_ ( 2, nC );
		a *= fWeightMD;
		b ( 1 ) = eimMDc_ ( 0, nC );
		b ( 2 ) = eimMDc_ ( 1, nC );
		b ( 3 ) = eimMDc_ ( 2, nC );
		b *= fWeightMD;
		//cross products
		Ma << a ( 0 ), -a ( 1 ), -a ( 2 ), -a ( 3 ),
			a ( 1 ),  a ( 0 ),  a ( 3 ), -a ( 2 ),
			a ( 2 ), -a ( 3 ),  a ( 0 ),  a ( 1 ),
			a ( 3 ),  a ( 2 ), -a ( 1 ),  a ( 0 );
		Mb << b ( 0 ), -b ( 1 ), -b ( 2 ), -b ( 3 ),
			b ( 1 ),  b ( 0 ), -b ( 3 ),  b ( 2 ),
			b ( 2 ),  b ( 3 ),  b ( 0 ), -b ( 1 ),
			b ( 3 ), -b ( 2 ),  b ( 1 ),  b ( 0 );
		//Add up
		M += Ma.transpose() * Mb;
	}

	Eigen::EigenSolver <Eigen::Matrix<T,4,4> > eigensolver ( M );
	Eigen::Matrix< std::complex< T >, 4, 1 > v = eigensolver.eigenvalues();

	//find the largest eigenvalue;
	float dLargest = -1000000;
	int n;

	for ( int i = 0; i < 4; i++ ) {
		if ( dLargest < v ( i ).real() ) {
			dLargest = v ( i ).real();
			n = i;
		}
	}

	//PRINT( dLargest );
	//PRINT( n );

	Eigen::Matrix<T,4,1> e;
	e <<eigensolver.eigenvectors().col ( n ) ( 0 ).real(),
		eigensolver.eigenvectors().col ( n ) ( 1 ).real(),
		eigensolver.eigenvectors().col ( n ) ( 2 ).real(),
		eigensolver.eigenvectors().col ( n ) ( 3 ).real();

	//PRINT( e );

	Eigen::Matrix<T,4,4>M1, M2, R;
	//Compute the rotation matrix
	M1 <<  e ( 0 ), -e ( 1 ), -e ( 2 ), -e ( 3 ),
		e ( 1 ),  e ( 0 ),  e ( 3 ), -e ( 2 ),
		e ( 2 ), -e ( 3 ),  e ( 0 ),  e ( 1 ),
		e ( 3 ),  e ( 2 ), -e ( 1 ),  e ( 0 );
	M2 <<  e ( 0 ), -e ( 1 ), -e ( 2 ), -e ( 3 ),
		e ( 1 ),  e ( 0 ), -e ( 3 ),  e ( 2 ),
		e ( 2 ),  e ( 3 ),  e ( 0 ), -e ( 1 ),
		e ( 3 ), -e ( 2 ),  e ( 1 ),  e ( 0 );
	R = M1.transpose() * M2;
	Eigen::Matrix< T, 3, 3> R_tmp;
	R_tmp = R.block ( 1, 1, 3, 3 );


	//Compute the final translation
	Eigen::Matrix< T, 3, 1> T_tmp;
	T_tmp = eivCentroidB - R_tmp * eivCentroidA;

	//Compute the residual error
	T dE = 0; // final energy
	Eigen::Matrix<T,3,1> eivE;

	//recollect inliers
	for ( int nC = 0; nC < eimXw_.cols(); nC++ ) {
		eivE = eimXc_.col ( nC ) - ( R_tmp * eimXw_.col ( nC ) + T_tmp );
		T dEN = eivE.norm();
		dE += dEN;
	}
	dE /= eimXw_.cols();

	T dA = 0;
	for ( int nC = 0; nC < eimNlw_.cols(); nC++ ) {
		//normal
		{
			T dEN = eimNlc_.col( nC ).dot( R_tmp * eimNlw_.col ( nC ) );
			dEN = dEN > 1.f ? 1.f : dEN;
			dEN = dEN <-1.f ?-1.f : dEN;
			dA += acos( dEN );
		}
		//main direction
		{
			T dEN = eimMDc_.col( nC ).dot( R_tmp * eimMDw_.col ( nC ) );
			dEN = dEN > 1.f ? 1.f : dEN;
			dEN = dEN <-1.f ?-1.f : dEN;
			dA += acos( dEN );
		}
	}
	dA /= (eimNlw_.cols()*2);

	( *pRw_ ) = R_tmp;
	( *pTw_ ) = T_tmp;

	Matrix<T,2,1> eivEA( dE, dA );
	return eivEA;
}

template< class T > /*Eigen::Matrix<float,-1,-1,0,-1,-1> = Eigen::MatrixXf*/
Eigen::Matrix<T,2,1> p1p_3d_nl_md ( const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimXw_, const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimNlw_, const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimMDw_, 
											  const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimXc_, const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimNlc_, const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimMDc_, 
											  Eigen::Matrix< T, 3, 3>* pRw_, Eigen::Matrix< T , 3, 1 >* pTw_ ){

	Eigen::Matrix< T , 3, 1 > Xw = eimXw_.col(0);
	Eigen::Matrix< T , 3, 1 > Xc = eimXc_.col(0);

	Eigen::Matrix< T , 3, 1 > Nlw = eimNlw_.col(0);
	Eigen::Matrix< T , 3, 1 > Nlc = eimNlc_.col(0);

	Eigen::Matrix< T , 3, 1 > Mdw = eimMDw_.col(0);
	Eigen::Matrix< T , 3, 1 > Mdc = eimMDc_.col(0);

	Eigen::Matrix< T , 3, 1 > nmw = Nlw.cross(Mdw);
	Eigen::Matrix< T , 3, 1 > nmc = Nlc.cross(Mdc);

	Eigen::Matrix< T , 3, 3 > Rc; 
	Rc.col(0) = Nlc;
	Rc.col(1) = Mdc;
	Rc.col(2) = nmc;

	Eigen::Matrix< T , 3, 3 > Rw; 
	Rw.col(0) = Nlw;
	Rw.col(1) = Mdw;
	Rw.col(2) = nmw;

	Eigen::Matrix< T , 3, 3 >  Rcw = Rw.transpose()*Rc;
	*pRw_ = Rcw;
	*pTw_ = Xc - Rcw*Xw;

	Matrix<T,2,1> eivEA( 0, 0 );
	return eivEA;
}



template< class T > /*Eigen::Matrix<float,-1,-1,0,-1,-1> = Eigen::MatrixXf*/
Eigen::Matrix<T,2,1> aoWithNormalRansac ( const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimXw_, const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimNlw_, 
														   const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimXc_, const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimNlc_, 
														   const T dist_thre_, const T angle_thre_, const int Iter,
														   Eigen::Matrix< T, 3, 3>* pR_, Eigen::Matrix< T , 3, 1 >* pT_, cv::Mat * pInliers_ ){
	// A is from World  B is Local coordinate system
	// eimB_ = R * eimA_ + T; //R and t is defined in world and transform a point in world to local
	// main references: http://www.mathworks.com/matlabcentral/fileexchange/22422-absolute-orientation
	using namespace Eigen;
	using namespace cv;
	//CHECK ( 	eimXw_.rows() == 3, " absoluteOrientation() requires the input matrix A_ is a 3 x N matrix. " );
	//CHECK ( 	eimXc_.rows() == 3, " absoluteOrientation() requires the input matrix B_ is a 3 x N matrix. " );
	//CHECK ( 	eimXw_.cols() == eimXc_.cols(), " absoluteOrientation() requires the columns of input matrix A_ and B_ are equal. " );

	RandomElements<int> re( (int)eimXw_.cols() );
	const int K = 2;
	vector<vector<int> > _v_v_inlier_idx;

	cv::Mat all_votes(1,Iter,CV_32FC1);
	for (int i=0; i < Iter; i++)	{
		//randomly select K candidates
		vector<int> selected_cols;
		re.run(K, &selected_cols);
		Matrix<T,-1,-1,0,-1,-1> eimX_world(3,K),eimNl_world(3,K),eimX_cam(3,K),eimNl_cam(3,K);
		for (int c=0; c<K; c++) {
			eimX_world.col( c )  = eimXw_.col( selected_cols[c] );
			eimX_cam.col( c )    = eimXc_.col( selected_cols[c] );
			eimNl_world.col( c ) = eimNlw_.col( selected_cols[c] );
			eimNl_cam.col( c )   = eimNlc_.col( selected_cols[c] );
		}
		//calc R&t		
		Eigen::Matrix< T, 3, 3> R_tmp;
		Eigen::Matrix< T, 3, 1> T_tmp;
		absoluteOrientationWithNormal<T>( eimX_world,eimNl_world,
										  eimX_cam, eimNl_cam, &R_tmp,&T_tmp );
		//collect votes
		int nVotes = 0;
		vector<int> inlier_idx;
		for (int c=0; c < eimXw_.cols(); c++  )	{
			Eigen::Matrix< T, 3, 1> eivE = eimXc_.col ( c ) - ( R_tmp * eimXw_.col ( c ) + T_tmp );
			T dCosA = eimNlc_.col( c ).dot( R_tmp * eimNlw_.col ( c ) );
			dCosA = dCosA > 1.f ? 1.f : dCosA;	dCosA = dCosA <-1.f ?-1.f : dCosA;
			T dA = acos( dCosA );
			if ( eivE.norm() < dist_thre_ && dA < angle_thre_ ){
				inlier_idx.push_back( c );
				nVotes ++;
			}
		}
		all_votes.at<int>( 0,i ) = nVotes;
		_v_v_inlier_idx.push_back( inlier_idx );
	}
	cv::Mat most;
	sortIdx(all_votes,most, CV_SORT_DESCENDING + CV_SORT_EVERY_ROW );
	//collect inliers
	////////////////////////////////////////////////////////////////
	int best_idx = most.at<int>(0,0);
	PRINT( all_votes.at<int>( 0,best_idx) );

	vector<int>& v_inlier_idx = _v_v_inlier_idx[best_idx];
	pInliers_->create(1,(int)v_inlier_idx.size(),CV_32SC1); 
	Matrix<T,-1,-1,0,-1,-1> eimX_world_selected( 3,v_inlier_idx.size() ),   eimNl_world_selected( 3,v_inlier_idx.size() ),
		                    eimX_cam_selected(3,v_inlier_idx.size()),       eimNl_cam_selected( 3,v_inlier_idx.size() );

	for ( int i=0; i< v_inlier_idx.size(); i++)
	{
		eimX_world_selected.col(i)  = eimXw_.col( v_inlier_idx[i] );
		eimNl_world_selected.col(i) = eimNlw_.col( v_inlier_idx[i] );
		eimX_cam_selected.col(i)  = eimXc_.col( v_inlier_idx[i] );
		eimNl_cam_selected.col(i) = eimNlc_.col( v_inlier_idx[i] );
		pInliers_->ptr<int>()[i] = v_inlier_idx[i];
	}

	Eigen::Matrix< T, 3, 3> R_tmp;
	Eigen::Matrix< T, 3, 1> T_tmp;
	Eigen::Matrix< T, 2, 1> dEA = absoluteOrientationWithNormal<T>( eimX_world_selected, eimNl_world_selected, eimX_cam_selected, eimNl_cam_selected, &R_tmp, &T_tmp);

	*pR_ = R_tmp;
	*pT_ = T_tmp;

	return dEA;
}

template< class T > /*Eigen::Matrix<float,-1,-1,0,-1,-1> = Eigen::MatrixXf*/
Eigen::Matrix<T,2,1> aoWithNormalWith2dConstraintRansac ( const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimXw_, const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimNlw_, 
														   const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimXc_, const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimNlc_, 
														   const T dist_thre_, const T angle_thre_, const T visual_angle_thre_, const int Iter, 
														   Eigen::Matrix< T, 3, 3>* pR_, Eigen::Matrix< T , 3, 1 >* pT_, cv::Mat * pInliers_ ){
	// A is from World  B is Local coordinate system
	// eimB_ = R * eimA_ + T; //R and t is defined in world and transform a point in world to local
	// main references: http://www.mathworks.com/matlabcentral/fileexchange/22422-absolute-orientation
	using namespace Eigen;
	using namespace cv;
	//CHECK ( 	eimXw_.rows() == 3, " absoluteOrientation() requires the input matrix A_ is a 3 x N matrix. " );
	//CHECK ( 	eimXc_.rows() == 3, " absoluteOrientation() requires the input matrix B_ is a 3 x N matrix. " );
	//CHECK ( 	eimXw_.cols() == eimXc_.cols(), " absoluteOrientation() requires the columns of input matrix A_ and B_ are equal. " );

	RandomElements<int> re( (int)eimXw_.cols() );
	const int K = 2;
	vector<vector<int> > _v_v_inlier_idx;

	//vector<vector<int> > _v_v_selected_cols;

	vector<Matrix3f> R_tmp; R_tmp.resize(Iter);//[Iter];
	vector<Vector3f> T_tmp; T_tmp.resize(Iter);//[Iter];

	//Matrix3f R_tmp[Iter];
	//Vector3f T_tmp[Iter];

	cv::Mat all_votes(1,Iter,CV_32FC1);

	//double t = (double)getTickCount() ;
	Matrix<T ,-1,-1,0,-1,-1> eimX_world(3,K),eimNl_world(3,K),eimX_cam(3,K),eimNl_cam(3,K);
	for (int i=0; i < Iter; i++)	{
		//randomly select K candidates
		vector<int> selected_cols;
		re.run(K, &selected_cols);
		for (int c=0; c<K; c++) {
			eimX_world.col( c )  = eimXw_.col( selected_cols[c] );
			eimX_cam.col( c )    = eimXc_.col( selected_cols[c] );
			eimNl_world.col( c ) = eimNlw_.col( selected_cols[c] );
			eimNl_cam.col( c )   = eimNlc_.col( selected_cols[c] );
		}
		//calc R&t		
		//Eigen::Matrix< T , 3, 3> R_tmp;
		//Eigen::Matrix< T , 3, 1> T_tmp;
		absoluteOrientationWithNormal<T>( eimX_world,eimNl_world,
										  eimX_cam, eimNl_cam, &(R_tmp[i]),&(T_tmp[i]) );
	}
	/*t = ((double)getTickCount() - t)/getTickFrequency();
	cout << "1: " << t << endl;
	t = (double)getTickCount();*/

	Eigen::Matrix< T , 3, 1> eivXc_tmp; 
	Eigen::Matrix< T , 3, 1> eivE;
	for (int i=0; i < Iter; i++)	{
		//collect votes
		int nVotes = 0;
		vector<int> inlier_idx;
		for (int c=0; c < eimXw_.cols(); c++  )	{
			eivXc_tmp = R_tmp[i] * eimXw_.col ( c ) + T_tmp[i];  
			//dist constraint
			eivE = eimXc_.col ( c ) - eivXc_tmp;
			//visual angle constraint
			T dCosVisualAngle = eimXc_.col ( c ).dot( eivXc_tmp )/eimXc_.col ( c ).norm()/eivXc_tmp.norm();
			dCosVisualAngle = dCosVisualAngle > 1.f ? 1.f : dCosVisualAngle;	dCosVisualAngle = dCosVisualAngle <-1.f ?-1.f : dCosVisualAngle;
			T dVisualAngle = acos( dCosVisualAngle );

			//normal constraint
			T dCosA = eimNlc_.col( c ).dot( R_tmp[i] * eimNlw_.col ( c ) );
			dCosA = dCosA > 1.f ? 1.f : dCosA;	dCosA = dCosA <-1.f ?-1.f : dCosA;
			T dA = acos( dCosA );
			if ( eivE.norm() < dist_thre_ && dA < angle_thre_ && dVisualAngle < visual_angle_thre_ ){
				inlier_idx.push_back( c );
				nVotes ++;
			}
		}
		all_votes.at<int>( 0,i ) = nVotes;
		_v_v_inlier_idx.push_back( inlier_idx );
	}

	/*t = ((double)getTickCount() - t)/getTickFrequency();
	cout << "2: " << t << endl;
	t = (double)getTickCount();*/

	cv::Mat most;
	sortIdx(all_votes,most, CV_SORT_DESCENDING + CV_SORT_EVERY_ROW );
	//collect inliers
	////////////////////////////////////////////////////////////////
	int best_idx = most.at<int>(0,0);
	//PRINT( all_votes.at<int>( 0,best_idx) );

	vector<int>& v_inlier_idx = _v_v_inlier_idx[best_idx];
	pInliers_->create(1,(int)v_inlier_idx.size(),CV_32SC1); 
	Matrix<T ,-1,-1,0,-1,-1> eimX_world_selected( 3,v_inlier_idx.size() ),   eimNl_world_selected( 3,v_inlier_idx.size() ),
		                    eimX_cam_selected(3,v_inlier_idx.size()),       eimNl_cam_selected( 3,v_inlier_idx.size() );

	for ( int i=0; i< v_inlier_idx.size(); i++)
	{
		eimX_world_selected.col(i)  = eimXw_.col( v_inlier_idx[i] );
		eimNl_world_selected.col(i) = eimNlw_.col( v_inlier_idx[i] );
		eimX_cam_selected.col(i)  = eimXc_.col( v_inlier_idx[i] );
		eimNl_cam_selected.col(i) = eimNlc_.col( v_inlier_idx[i] );
		pInliers_->ptr<int>()[i] = v_inlier_idx[i];
	}

	Eigen::Matrix< T , 3, 3> R_final;
	Eigen::Matrix< T , 3, 1> T_final;
	Eigen::Matrix< T , 2, 1> dEA = absoluteOrientationWithNormal<T>( eimX_world_selected, eimNl_world_selected, eimX_cam_selected, eimNl_cam_selected, &R_final, &T_final);

	*pR_ = R_final;
	*pT_ = T_final;

	return dEA;
}

template< class T > /*Eigen::Matrix<float,-1,-1,0,-1,-1> = Eigen::MatrixXf*/
Eigen::Matrix<T,2,1> aoWithNormaln2dConstraintRansac2( const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimXw_, const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimNlw_, 
																			   const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimXc_, const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimNlc_, 
																			   const T dist_thre_, const T angle_thre_, const T visual_angle_thre_, const int Iter_, 
																			   Eigen::Matrix< T , 3, 3>* pR_, Eigen::Matrix< T , 3, 1 >* pT_, cv::Mat * pInliers_ ){
	// A is from World  B is Local coordinate system
	// eimB_ = R * eimA_ + T; //R and t is defined in world and transform a point in world to local
	// main references: http://www.mathworks.com/matlabcentral/fileexchange/22422-absolute-orientation
	using namespace Eigen;
	using namespace cv;
	//CHECK ( 	eimXw_.rows() == 3, " absoluteOrientation() requires the input matrix A_ is a 3 x N matrix. " );
	//CHECK ( 	eimXc_.rows() == 3, " absoluteOrientation() requires the input matrix B_ is a 3 x N matrix. " );
	//CHECK ( 	eimXw_.cols() == eimXc_.cols(), " absoluteOrientation() requires the columns of input matrix A_ and B_ are equal. " );

	RandomElements<int> re( (int)eimXw_.cols() );
	const int K = 2;
	vector<vector<int> > _v_v_inlier_pt_idx;
	vector<vector<int> > _v_v_inlier_nl_idx;
	vector<vector<int> > _v_v_inlier_2d_idx;

	//vector<vector<int> > _v_v_selected_cols;

	vector<Matrix3f> R_tmp; R_tmp.resize(Iter_);//[Iter];
	vector<Vector3f> T_tmp; T_tmp.resize(Iter_);//[Iter];

	//Matrix3f R_tmp[Iter];
	//Vector3f T_tmp[Iter];

	cv::Mat all_votes(1,Iter_,CV_32FC1);
	cv::Mat all_votes_3d(1,Iter_,CV_32FC1);
	cv::Mat all_votes_nl(1,Iter_,CV_32FC1);
	cv::Mat all_votes_2d(1,Iter_,CV_32FC1);

	//double t = (double)getTickCount() ;
	Matrix<T ,-1,-1,0,-1,-1> eimX_world(3,K),eimNl_world(3,K),eimX_cam(3,K),eimNl_cam(3,K);
	for (int i=0; i < Iter_; i++)	{
		//randomly select K candidates
		vector<int> selected_cols;
		re.run(K, &selected_cols);
		for (int c=0; c<K; c++) {
			eimX_world.col( c )  = eimXw_.col( selected_cols[c] );
			eimX_cam.col( c )    = eimXc_.col( selected_cols[c] );
			eimNl_world.col( c ) = eimNlw_.col( selected_cols[c] );
			eimNl_cam.col( c )   = eimNlc_.col( selected_cols[c] );
		}
		//calc R&t		
		//Eigen::Matrix< T , 3, 3> R_tmp;
		//Eigen::Matrix< T , 3, 1> T_tmp;
		absoluteOrientationWithNormal<T>( eimX_world,eimNl_world,
											  eimX_cam,  eimNl_cam, &(R_tmp[i]),&(T_tmp[i]) );
	}


	/*t = ((double)getTickCount() - t)/getTickFrequency();
	cout << "1: " << t << endl;
	t = (double)getTickCount();*/

	Eigen::Matrix< T , 3, 1> eivXc_tmp; 
	Eigen::Matrix< T , 3, 1> eivE;
	for (int i=0; i < Iter_; i++)	{
		//collect votes
		int nVotes_3d = 0;
		int nVotes_nl = 0;
		int nVotes_2d = 0;
		vector<int> inlier_pt_idx;
		vector<int> inlier_nl_idx;
		vector<int> inlier_2d_idx;
		Vector3f C_tmp = - R_tmp[i].transpose() * T_tmp[i] ;
		for (int c=0; c < eimXw_.cols(); c++  )	{
			eivXc_tmp = R_tmp[i] * ( eimXw_.col ( c ) - C_tmp );  
			//dist constraint
			eivE = eimXc_.col ( c ) - eivXc_tmp;
			//visual angle constraint
			T dCosVisualAngle = eimXc_.col ( c ).dot( eivXc_tmp )/eimXc_.col ( c ).norm()/eivXc_tmp.norm();
			dCosVisualAngle = dCosVisualAngle > 1.f ? 1.f : dCosVisualAngle;	dCosVisualAngle = dCosVisualAngle <-1.f ?-1.f : dCosVisualAngle;
			T dVisualAngle = acos( dCosVisualAngle );

			//normal constraint
			T dCosA = eimNlc_.col( c ).dot( R_tmp[i] * eimNlw_.col ( c ) );
			dCosA = dCosA > 1.f ? 1.f : dCosA;	dCosA = dCosA <-1.f ?-1.f : dCosA;
			T dA = acos( dCosA );
			if ( eivE.norm() < dist_thre_ && dVisualAngle < visual_angle_thre_ ){
				inlier_pt_idx.push_back( c );
				nVotes_3d ++;
			}
			if ( dA < angle_thre_ && dVisualAngle < visual_angle_thre_ ){
				inlier_nl_idx.push_back( c );
				nVotes_nl ++;
			}
			if ( dVisualAngle < visual_angle_thre_ ){
				inlier_2d_idx.push_back( c );
				nVotes_2d ++;
			}
		}
		all_votes_3d.at<int>( 0,i ) = nVotes_3d;
		all_votes_nl.at<int>( 0,i ) = nVotes_nl;
		all_votes_2d.at<int>( 0,i ) = nVotes_2d;
		all_votes.at<int>( 0,i ) = nVotes_nl + nVotes_3d;
		_v_v_inlier_pt_idx.push_back( inlier_pt_idx );
		_v_v_inlier_nl_idx.push_back( inlier_nl_idx );
		_v_v_inlier_2d_idx.push_back( inlier_2d_idx );
	}

	/*t = ((double)getTickCount() - t)/getTickFrequency();
	cout << "2: " << t << endl;
	t = (double)getTickCount();*/

	cv::Mat most;
	sortIdx(all_votes, most, CV_SORT_DESCENDING + CV_SORT_EVERY_ROW );
	//collect inliers
	////////////////////////////////////////////////////////////////
	int ii = 0;
	int best_idx = most.at<int>(0,ii);
	while( !(all_votes_2d.at<int>( 0,best_idx ) >= 2 && all_votes_nl.at<int>( 0,best_idx ) >= 2 && all_votes_3d .at<int>( 0,best_idx ) >= 2 ) && ii< most.cols ){
		best_idx = most.at<int>(0,ii); ii++;
	}
	if( ii >= most.cols ){
		pR_->setZero(); 
		return Eigen::Matrix< T , 2, 1>( 1000.f, 1000.f );
	}
	//PRINT( all_votes.at<int>( 0,best_idx) );

	vector<int>& v_inlier_pt_idx = _v_v_inlier_pt_idx[best_idx];
	vector<int>& v_inlier_nl_idx = _v_v_inlier_nl_idx[best_idx];
	vector<int>& v_inlier_2d_idx = _v_v_inlier_2d_idx[best_idx];
	pInliers_->create(1,(int)v_inlier_pt_idx.size(),CV_32SC1); 
	Matrix<T ,-1,-1,0,-1,-1> eimX_world_selected( 3,v_inlier_pt_idx.size() ),   eimNl_world_selected( 3,v_inlier_nl_idx.size() ),
		                         eimX_cam_selected(3,v_inlier_pt_idx.size() ),       eimNl_cam_selected( 3,v_inlier_nl_idx.size() );

	for ( int i=0; i< v_inlier_pt_idx.size(); i++)
	{
		eimX_world_selected.col(i)  = eimXw_.col( v_inlier_pt_idx[i] );
		eimX_cam_selected.col(i)  = eimXc_.col( v_inlier_pt_idx[i] );
		pInliers_->ptr<int>()[i] = v_inlier_pt_idx[i];
	}

	for ( int i=0; i< v_inlier_nl_idx.size(); i++)
	{
		eimNl_world_selected.col(i) = eimNlw_.col( v_inlier_nl_idx[i] );
		eimNl_cam_selected.col(i) = eimNlc_.col( v_inlier_nl_idx[i] );
	}

	
	/*for ( int i=0; i< v_inlier_2d_idx.size(); i++)
	{
		int idx = i + v_inlier_nl_idx.size();


		eimNl_world_selected.col(idx) = eimNlw_.col( v_inlier_2d_idx[i] );
		eimNl_cam_selected.col(idx) = eimNlc_.col( v_inlier_2d_idx[i] );
	}*/

	Eigen::Matrix< T , 3, 3> R_final;
	Eigen::Matrix< T , 3, 1> T_final;
	Eigen::Matrix< T , 2, 1> dEA = absoluteOrientationWithNormal<T>( eimX_world_selected, eimNl_world_selected, eimX_cam_selected, eimNl_cam_selected, &R_final, &T_final);

	*pR_ = R_final;
	*pT_ = T_final;

	return dEA;
}

template< class T > /*Eigen::Matrix<float,-1,-1,0,-1,-1> = Eigen::MatrixXf*/
Eigen::Matrix<T,2,1> aoWithNormalMainDirectionWith2dConstraintRansac ( const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimXw_, const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimNlw_, const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimMDw_, 
														   const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimXc_, const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimNlc_, const Eigen::Matrix<T,-1,-1,0,-1,-1>& eimMDc_,  
														   const T dist_thre_, const T angle_thre_, const T visual_angle_thre_, const int Iter,
														   Eigen::Matrix< T , 3, 3>* pR_, Eigen::Matrix< T , 3, 1 >* pT_, cv::Mat * pInliers_ ){
	// A is from World  B is Local coordinate system
	// eimB_ = R * eimA_ + T; //R and t is defined in world and transform a point in world to local
	// main references: http://www.mathworks.com/matlabcentral/fileexchange/22422-absolute-orientation
	using namespace Eigen;
	using namespace cv;
	//CHECK ( eimXw_.rows() == 3, " absoluteOrientation() requires the input matrix A_ is a 3 x N matrix. " );
	//CHECK ( eimXc_.rows() == 3, " absoluteOrientation() requires the input matrix B_ is a 3 x N matrix. " );
	//CHECK ( eimXw_.cols() == eimXc_.cols(), " absoluteOrientation() requires the columns of input matrix A_ and B_ are equal. " );

	T md_thre = angle_thre_*20;

	RandomElements<int> re( (int)eimXw_.cols() );
	const int K = 2;
	vector<vector<int> > _v_v_inlier_idx;

	//vector<vector<int> > _v_v_selected_cols;

	vector<Matrix3f> R_tmp; R_tmp.resize(Iter);//[Iter];
	vector<Vector3f> T_tmp; T_tmp.resize(Iter);//[Iter];

	cv::Mat all_votes(1,Iter,CV_32FC1);

	//double t = (double)getTickCount() ;
	Matrix<T ,-1,-1,0,-1,-1> eimX_world(3,K),eimNl_world(3,K),eimMD_world(3,K),eimX_cam(3,K),eimNl_cam(3,K),eimMD_cam(3,K);
	for (int i=0; i < Iter; i++)	{
		//randomly select K candidates
		vector<int> selected_cols;
		re.run(K, &selected_cols);
		for (int c=0; c<K; c++) {
			eimX_world.col( c )  = eimXw_.col( selected_cols[c] );
			eimX_cam.col( c )    = eimXc_.col( selected_cols[c] );
			eimNl_world.col( c ) = eimNlw_.col( selected_cols[c] );
			eimNl_cam.col( c )   = eimNlc_.col( selected_cols[c] );
			eimMD_world.col( c ) = eimMDw_.col( selected_cols[c] );
			eimMD_cam.col( c )   = eimMDc_.col( selected_cols[c] );
		}
		//calc R&t		
		//Eigen::Matrix< T , 3, 3> R_tmp;
		//Eigen::Matrix< T , 3, 1> T_tmp;
		//p1p_3d_nl_md<T>( eimX_world,eimNl_world,eimMD_world,
		//					 eimX_cam, eimNl_cam, eimMD_cam, &(R_tmp[i]),&(T_tmp[i]) );
		absoluteOrientationWithNormalnMainDirection<T>( eimX_world, eimNl_world, eimMD_world,
														eimX_cam, eimNl_cam, eimMD_cam, &(R_tmp[i]), &(T_tmp[i]));
	}
	/*t = ((double)getTickCount() - t)/getTickFrequency();
	cout << "1: " << t << endl;
	t = (double)getTickCount();*/

	Eigen::Matrix< T , 3, 1> eivXc_tmp; 
	Eigen::Matrix< T , 3, 1> eivE;
	for (int i=0; i < Iter; i++)	{
		//collect votes
		int nVotes = 0;
		vector<int> inlier_idx;
		for (int c=0; c < eimXw_.cols(); c++  )	{
			eivXc_tmp = R_tmp[i] * eimXw_.col ( c ) + T_tmp[i];  
			//dist constraint
			eivE = eimXc_.col ( c ) - eivXc_tmp;
			T dVisualAngle, dNormalAngle, dMDAngle;
			//visual angle constraint
			{
				T dCosA = eimXc_.col ( c ).dot( eivXc_tmp )/eimXc_.col ( c ).norm()/eivXc_tmp.norm();
				dCosA = dCosA > 1.f ? 1.f : dCosA;	dCosA = dCosA <-1.f ?-1.f : dCosA;
				dVisualAngle = acos( dCosA );
			}
			//normal constraint
			{
				T dCosA = eimNlc_.col( c ).dot( R_tmp[i] * eimNlw_.col ( c ) );
				dCosA = dCosA > 1.f ? 1.f : dCosA;	dCosA = dCosA <-1.f ?-1.f : dCosA;
				dNormalAngle = acos( dCosA );
			}
			//main direction constraint
			{
				T dCosA = eimMDc_.col( c ).dot( R_tmp[i] * eimMDw_.col ( c ) );
				dCosA = dCosA > 1.f ? 1.f : dCosA;	dCosA = dCosA <-1.f ?-1.f : dCosA;
				dMDAngle = acos( dCosA );
			}
			if ( eivE.norm() < dist_thre_ && dNormalAngle < angle_thre_ && dMDAngle < md_thre  && dVisualAngle < visual_angle_thre_ ){
				inlier_idx.push_back( c );
				nVotes ++;
			}
		}
		all_votes.at<int>( 0,i ) = nVotes;
		_v_v_inlier_idx.push_back( inlier_idx );
	}

	/*t = ((double)getTickCount() - t)/getTickFrequency();
	cout << "2: " << t << endl;
	t = (double)getTickCount();*/

	cv::Mat most;
	sortIdx(all_votes,most, CV_SORT_DESCENDING + CV_SORT_EVERY_ROW );
	//collect inliers
	////////////////////////////////////////////////////////////////
	int best_idx = most.at<int>(0,0);
	//PRINT( all_votes.at<int>( 0,best_idx) );

	vector<int>& v_inlier_idx = _v_v_inlier_idx[best_idx];
	pInliers_->create(1,(int)v_inlier_idx.size(),CV_32SC1); 
	Matrix<T ,-1,-1,0,-1,-1> eimX_world_selected( 3,v_inlier_idx.size() ),   eimNl_world_selected( 3,v_inlier_idx.size() ),  eimMD_world_selected( 3,v_inlier_idx.size() ),
		                    eimX_cam_selected(3,v_inlier_idx.size()),       eimNl_cam_selected( 3,v_inlier_idx.size() ),  eimMD_cam_selected( 3,v_inlier_idx.size() );

	for ( int i=0; i< v_inlier_idx.size(); i++)
	{
		eimX_world_selected.col(i)  = eimXw_.col( v_inlier_idx[i] );
		eimNl_world_selected.col(i) = eimNlw_.col( v_inlier_idx[i] );
		eimMD_world_selected.col(i) = eimMDw_.col( v_inlier_idx[i] );
		eimX_cam_selected.col(i)  = eimXc_.col( v_inlier_idx[i] );
		eimNl_cam_selected.col(i) = eimNlc_.col( v_inlier_idx[i] );
		eimMD_cam_selected.col(i) = eimMDc_.col( v_inlier_idx[i] );
		pInliers_->ptr<int>()[i] = v_inlier_idx[i];
	}

	Eigen::Matrix< T , 3, 3> R_final;
	Eigen::Matrix< T , 3, 1> T_final;
	Eigen::Matrix< T , 2, 1> dEA = absoluteOrientationWithNormalnMainDirection<T>( eimX_world_selected, eimNl_world_selected,eimMD_world_selected, eimX_cam_selected, eimNl_cam_selected,eimMD_cam_selected, &R_final, &T_final);

	*pR_ = R_final;
	*pT_ = T_final;

	return dEA;
}

template< class T, int ROW, int COL >
T matNormL1 ( const Eigen::Matrix< T, ROW, COL >& eimMat1_, const Eigen::Matrix< T, ROW, COL >& eimMat2_ )
{
	Eigen::Matrix< T, ROW, COL > eimTmp = eimMat1_ - eimMat2_;
	Eigen::Matrix< T, ROW, COL > eimAbs = eimTmp.cwiseAbs();
	return (T) eimAbs.sum();
}

template< class T >
void setSkew( T x_, T y_, T z_, Eigen::Matrix< T, 3,3 >* peimMat_){
	*peimMat_ << 0, -z_, y_, z_, 0, -x_, -y_, x_, 0 ;
}

template< class T >
void setRotMatrixUsingExponentialMap( T x_, T y_, T z_, Eigen::Matrix< T, 3,3 >* peimR_ ){
	//http://opencv.itseez.com/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=rodrigues#void Rodrigues(InputArray src, OutputArray dst, OutputArray jacobian)
	T theta = sqrt( x_*x_ + y_*y_ + z_*z_ );
	if(	theta < std::numeric_limits<T>::epsilon() ){
		*peimR_ = Eigen::Matrix< T, 3,3 >::Identity();
		return;
	}
	T sinTheta = sin(theta);
	T cosTheta = cos(theta);
	Eigen::Matrix< T, 3,3 > eimSkew; 
	setSkew< T >(x_/theta,y_/theta,z_/theta,&eimSkew);
	*peimR_ = Eigen::Matrix< T, 3,3 >::Identity() + eimSkew*sinTheta + eimSkew*eimSkew*(1-cosTheta);
}

}//utility
}//btl
#endif
