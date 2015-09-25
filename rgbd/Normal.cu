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
//
//You agree to acknowledge use of the Software in any reports or publications of
//results obtained with the Software and make reference to the following publication :
//Li, Shuda, &Calway, Andrew(2015).RGBD Relocalisation Using Pairwise Geometry
//and Concise Key Point Sets.In Intl Conf.Robotics and Automation.

#define EXPORT
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#endif

#include <opencv2/cudaarithm.hpp>
#include <math_constants.h>
#include <opencv2/core/cuda/common.hpp>
#include "pcl/limits.hpp"
#include "pcl/device.hpp"
#include "pcl/vector_math.hpp"
#include "Normal.cuh"

namespace btl{ namespace device
{
using namespace pcl::device;
using namespace cv;
using namespace cv::cuda;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////


}//device
}//btl