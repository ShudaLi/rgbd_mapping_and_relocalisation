cmake_minimum_required (VERSION 2.8)
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0020 NEW)
endif()

project (MappingAndRelocalisation)

find_package(CUDA)
include(FindCUDA)

find_package( OpenCV REQUIRED PATHS COMPONENTS core imgcodecs features2d  videoio highgui video calib3d cudaarithm )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(OpenGL REQUIRED)
find_package(Qt5OpenGL REQUIRED)
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5Xml REQUIRED)

QT5_WRAP_CPP(HEADERS_MOC ${HEADERS})

include_directories ( $ENV{QGLVIEWER_INC} )
include_directories ( $ENV{EIGEN_INC} )
include_directories ( $ENV{OPENNI2_INC} )
include_directories ( $ENV{FREEGLUT_INC} )
include_directories ( $ENV{GLEW_INC} )
include_directories ( ${PROJECT_SOURCE_DIR}/../rgbd )
include_directories ( ${PROJECT_SOURCE_DIR}/../common/ )
include_directories ( ${PROJECT_SOURCE_DIR}/../external_libraries/ )

link_directories ( $ENV{QGLVIEWER_LIB} )
link_directories ( $ENV{OPENNI2_LIB} )
link_directories ( $ENV{FREEGLUT_LIB} )
link_directories ( $ENV{GLEW_LIB} )
	
	
if(WIN32)
	set(RGBDLIB optimized Rgbd debug Rgbdd)
	set(QGLVIEWER  optimized QGLViewer2  debug QGLViewerd2 )
	set(ZLIB optimized zlib debug zlibd )	
	set(EXTRA_LIBS ${RGBDLIB} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} OpenNI2
		freeglut GLU32  glew32 GLU32 opengl32 ${QGLVIEWER} ${QT_LIBRARIES} ${OpenCV_LIBS})
elseif(UNIX)
	set(RGBDLIB Rgbd)
	set(FREEGLUTLIB ${GLUT_glut_LIBRARY})
	set(GLEWLIB ${GLEW_LIBRARY})
	set(GLLIB ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY})
	set(OPENNI2LIB ${OPENNI2_LIBRARY})
	set(ZLIB ${ZLIB_LIBRARY})
	set(QGLVIEWER  QGLViewer-qt5 )
	set(EXTRA_LIBS ${RGBDLIB} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} OpenNI2
        ${FREEGLUTLIB} ${GLLIB} ${GLEWLIB} ${QGLVIEWER} ${QT_LIBRARIES} ${OpenCV_LIBS})
endif()

cuda_add_executable(MappingAndRelocalisation Main.cpp  MultiViewer.cpp Data4Viewer.h Data4Viewer.cpp DataLive.cpp MultiViewer.h DataLive.h)
target_link_libraries( MappingAndRelocalisation ${EXTRA_LIBS} ${ZLIB} )
qt5_use_modules(MappingAndRelocalisation OpenGL Gui Widgets Xml)
install( TARGETS MappingAndRelocalisation DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/../" )