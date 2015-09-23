//Copyright(c) 2015 by Shuda Li[lishuda1980@gmail.com]
//
//Mapping and Relocalisation is licensed under the GPLv3 license.
//Details can be found in the following.
//
//For using the code or comparing to it in your research, you are
//expected to cite :
//Li, Shuda, &Calway, (2015) Andrew.RGBD Relocalisation Using
//Pairwise Geometry and Concise Key Point Sets.
//In Intl.Conf.on Robotics and Automatiro(ICRA) 2015.
//
//Permission is hereby granted, free of charge, to any person
//obtaining a copy of this software and associated documentation
//files(the "Software"), to deal in the Software without
//restriction, including without limitation the rights to use,
//copy, modify, merge, publish, distribute, sublicense, and / or sell
//copies of the Software, and to permit persons to whom the
//Software is furnished to do so, subject to the following
//conditions :
//
//The above copyright notice and this permission notice shall be
//included in all copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
//EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
//OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
//NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
//HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
//WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
//OTHER DEALINGS IN THE SOFTWARE.


#ifndef BTL_KINECT
#define BTL_KINECT


namespace btl{
namespace kinect{

#define KINECT_WIDTH 640
#define KINECT_WIDTH_L1 320
#define KINECT_WIDTH_L2 160
#define KINECT_WIDTH_L3 80
#define KINECT_WIDTH_L4 40
#define KINECT_WIDTH_L5 20
#define KINECT_WIDTH_LARGE 1280

#define KINECT_HEIGHT 480
#define KINECT_HEIGHT_L1 240
#define KINECT_HEIGHT_L2 120
#define KINECT_HEIGHT_L3 60
#define KINECT_HEIGHT_L4 30
#define KINECT_HEIGHT_L5 15
#define KINECT_HEIGHT_LARGE 1024

#define KINECT_WxH 307200
#define KINECT_WxH_L1 76800 //320*240
#define KINECT_WxH_L2 19200 //160*120
#define KINECT_WxH_L3 4800  // 80*60
#define KINECT_WxH_L4 1200  // 40*30
#define KINECT_WxH_L5 300  // 20*15
#define	KINECT_WxH_LARGE 1310720 //1280*1024

#define KINECT_WxHx3 921600
#define KINECT_WxHx3_L1 230400 
#define KINECT_WxHx3_L2 57600
#define KINECT_WxHx3_L3 19200
#define KINECT_WxHx3_L4 4800

static unsigned int __aDepthWxH[7] = {KINECT_WxH,   KINECT_WxH_L1,   KINECT_WxH_L2,   KINECT_WxH_L3,    KINECT_WxH_L4,   KINECT_WxH_L5,   KINECT_WxH   };
static unsigned short __aDepthW[7] = {KINECT_WIDTH, KINECT_WIDTH_L1, KINECT_WIDTH_L2, KINECT_WIDTH_L3,  KINECT_WIDTH_L4, KINECT_WIDTH_L5, KINECT_WIDTH };
static unsigned short __aDepthH[7] = {KINECT_HEIGHT,KINECT_HEIGHT_L1,KINECT_HEIGHT_L2,KINECT_HEIGHT_L3, KINECT_HEIGHT_L4,KINECT_HEIGHT_L5,KINECT_HEIGHT};

static unsigned int __aRGBWxH[7] = {KINECT_WxH,   KINECT_WxH_L1,   KINECT_WxH_L2,   KINECT_WxH_L3,    KINECT_WxH_L4,   KINECT_WxH_L5,   KINECT_WxH_LARGE   };
static unsigned short __aRGBW[7] = {KINECT_WIDTH, KINECT_WIDTH_L1, KINECT_WIDTH_L2, KINECT_WIDTH_L3,  KINECT_WIDTH_L4, KINECT_WIDTH_L5, KINECT_WIDTH_LARGE };
static unsigned short __aRGBH[7] = {KINECT_HEIGHT,KINECT_HEIGHT_L1,KINECT_HEIGHT_L2,KINECT_HEIGHT_L3, KINECT_HEIGHT_L4,KINECT_HEIGHT_L5,KINECT_HEIGHT_LARGE};


}//kinect
}//btl

namespace btl{
	enum{ SURF=0, ORB, BRISKC, FREAK, BINBOOST }; // features
	enum{ Frm_2_Frm_ICP=0, Vol_2_Frm_ICP, Combined_ICP };
	enum{ Relocalisation_Only=0, Tracking_n_Mapping, Mapping_Using_GT, Tracking_NonStation };//stage
}

#endif
