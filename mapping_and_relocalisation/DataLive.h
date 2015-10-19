#ifndef _KINECT_LIVEVIEWER_QGLV_APP_H_
#define _KINECT_LIVEVIEWER_QGLV_APP_H_

using namespace btl::gl_util;
using namespace btl::kinect;
using namespace btl::geometry;
using namespace std;
using namespace Eigen;

class CDataLive: public CData4Viewer
{
public:
	typedef boost::shared_ptr<CDataLive> tp_shared_ptr;

	CDataLive();
	virtual ~CDataLive(){ ; }
	virtual void loadFromYml();
	virtual void reset();
	virtual void updatePF();

	std::string _oniFileName; // the openni file 
};


#endif