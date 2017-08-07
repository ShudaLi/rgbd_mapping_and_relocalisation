
//display kinect depth in real-time
#define _USE_MATH_DEFINES
#define INFO
#define TIMER
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <memory>
#include <map>
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
#include <sophus/se3.hpp>
#include "EigenUtil.hpp"
#include "GLUtil.hpp"
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
  CMultiViewer camera_view(string("rgb_view"),_pData, vSplit1, (const QGLWidget*)&global_view);
  CMultiViewer rgb_view	(string("camera_view"),   _pData, vSplit2, (const QGLWidget*)&global_view);
  CMultiViewer depth_view (string("depth_view"), _pData, vSplit2, (const QGLWidget*)&global_view);

#if QT_VERSION < 0x040000
  // Set the viewer as the application main widget.
  application.setMainWidget(&viewer);
#else
  hSplit->setWindowTitle("Kinect Multi-view");
#endif

	// Make the viewer window visible on screen.
	hSplit->show();
	// Run main loop.
	return application.exec();
}
