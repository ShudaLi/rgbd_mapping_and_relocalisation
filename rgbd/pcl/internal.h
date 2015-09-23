/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef PCL_KINFU_INTERNAL_HPP_
#define PCL_KINFU_INTERNAL_HPP_

//#include "../cv/common.hpp"
#include <opencv2/core/cuda/common.hpp>

namespace pcl
{
  namespace device
  {
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	  typedef float4 PointType;
    /** \brief Camera intrinsics structure
      */ 
    struct Intr
    {
      float fx, fy, cx, cy;
	  __device__ __host__ Intr() {}
	  __device__ __host__ Intr(float fx_, float fy_, float cx_, float cy_) : fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}

      __device__ __host__ Intr operator()(int level_index) const
      { 
        int div = 1 << level_index; 
        return (Intr (fx / div, fy / div, cx / div, cy / div));
      }
    };

    /** \brief 3x3 Matrix for device code
      */ 
    struct Mat33
    {
      float3 data[3];
    };

	struct matrix3f{
		float3 r[3];
	};

	struct matrix3d{
		double3 r[3];
	};

	struct matrix3_cmf{
		float3 c[3];
	};

	struct matrix3_cmd{
		double3 c[3];
	};

    /** \brief Light source collection
      */ 
    struct LightSource
    {
      float3 pos[1];
      int number;
    };

	template<class D, class Matx> D& device_cast (Matx& matx)
	{
		return (* reinterpret_cast<D*>(matx.data ()));
	}

  }//device
}//pcl

#endif /* PCL_KINFU_INTERNAL_HPP_ */
