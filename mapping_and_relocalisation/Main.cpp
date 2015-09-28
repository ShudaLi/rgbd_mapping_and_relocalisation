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


//display kinect depth in real-time
#define INFO
#define TIMER
#include <GL/glew.h>
#include <GL/freeglut.h>
//#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <string>
#include <vector>

#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "Utility.hpp"

//camera calibration from a sequence of images
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <OpenNI.h>

#include "Kinect.h"
#include "EigenUtil.hpp"
#include "GLUtil.hpp"
#include <map>
#include "Camera.h"
#include "RGBDFrame.h"
#include "VideoSourceKinect.hpp"
#include "pcl/internal.h"
#include "CubicGrids.h"
#include "KinfuTracker.h"

#include <QGLViewer/qglviewer.h>
#include "Data4Viewer.h"
#include "DataLive.h"
#include "MultiViewer.h"
#include <qapplication.h>
#include <qsplitter.h>

using namespace std;

int main(int argc, char** argv)
{
  // Read command lines arguments.
  QApplication application(argc,argv);

  // Create Splitters
  QSplitter *hSplit  = new QSplitter(Qt::Vertical);
  QSplitter *vSplit1 = new QSplitter(hSplit);
  QSplitter *vSplit2 = new QSplitter(hSplit);

  hSplit->resize(1280,960);

  CDataLive::tp_shared_ptr _pData( new CDataLive() );

  // Instantiate the viewers.
  CMultiViewer global_view(string("global_view"),_pData, vSplit1, NULL);
  CMultiViewer camera_view(string("rgb_view"),_pData, vSplit1, &global_view);
  CMultiViewer rgb_view	(string("camera_view"),   _pData, vSplit2, &global_view);
  CMultiViewer depth_view (string("depth_view"), _pData, vSplit2, &global_view);

#if QT_VERSION < 0x040000
  // Set the viewer as the application main widget.
  application.setMainWidget(&viewer);
#else
  hSplit->setWindowTitle("Kinect Multi-view");
#endif

  try{
	  // Make the viewer window visible on screen.
	  hSplit->show();
	  // Run main loop.
	  return application.exec();
  }
  catch ( btl::utility::CError& e )  {
	  if ( std::string const* mi = boost::get_error_info< btl::utility::CErrorInfo > ( e ) ) {
		  std::cerr << "Error Info: " << *mi << std::endl;
	  }
  }
  catch ( std::runtime_error& e ){
	  PRINTSTR( e.what() );
  }
}
