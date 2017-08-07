cmake_minimum_required (VERSION 2.8)
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0020 NEW)
endif()

project (MappingAndRelocalisation)
include(${PROJECT_SOURCE_DIR}/../cmake/UseOpenNI.cmake)
include(${PROJECT_SOURCE_DIR}/../cmake/UseGLEW.cmake)
include(${PROJECT_SOURCE_DIR}/../cmake/UseQGLViewer.cmake)

find_package(CUDA)
include(FindCUDA)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( /usr/local/lib )
find_package(OpenGL REQUIRED)
find_package(Qt5OpenGL REQUIRED)
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5Xml REQUIRED)

QT5_WRAP_CPP(HEADERS_MOC ${HEADERS})
set(CUDA_NVCC_FLAGS; -arch=sm_30) #-arch=sm_35) #-arch=sm_21)
include_directories ( $ENV{QGLVIEWER_INC} )
#include_directories ( $ENV{GLEW_INC} )
include_directories ( ${PROJECT_SOURCE_DIR}/../rgbd )
include_directories ( ${PROJECT_SOURCE_DIR}/../common/ )
include_directories ( ${PROJECT_SOURCE_DIR}/../external_libraries/)
include_directories ( ${PROJECT_SOURCE_DIR}/../external_libraries/sophus/include/)
include_directories ( ${PROJECT_SOURCE_DIR}/../external_libraries/nifticlib/niftilib/ )

link_directories ( $ENV{QGLVIEWER_LIB} )
#link_directories ( $ENV{GLEW_LIB} )
	
if(WIN32)
	set(QGLVIEWER optimized ${QGLVIEWER_LIBRARY} debug ${QGLVIEWER_LIBRARY_DEBUG} )
	set(EXTRA_LIBS ${RGBDLIB} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} 
		GLU32 GLU32 opengl32 ${QT_LIBRARIES} ${OpenCV_LIBS})
elseif(UNIX)
	FIND_LIBRARY(GLUT_LIBRARY glut HINTS "/usr/lib/x86_64-linux-gnu")
	FIND_PATH(GLUT_INCLUDE_DIR glut.h HINTS "/usr/include/GL")
	#set(GLEWLIB ${GLEW_LIBRARY})
	set(GLLIB ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY})
	set(OPENNI2LIB ${OPENNI2_LIBRARY})
	set(QGLVIEWER  ${QGLVIEWER_LIBRARY} )
	set(EXTRA_LIBS ${RGBDLIB} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} 
        ${GLLIB} ${GLEWLIB} ${QGLVIEWER} ${QT_LIBRARIES} ${OpenCV_LIBS} ${GLUT_LIBRARY} 	      ${OPENGL_LIBRARY} -lGLEW -lstdc++fs)
endif()

cuda_add_executable(MappingAndRelocalisation Main.cpp  MultiViewer.cpp Data4Viewer.h Data4Viewer.cpp DataLive.cpp MultiViewer.h DataLive.h)
target_link_libraries( MappingAndRelocalisation ${EXTRA_LIBS} ${ZLIB} Rgbd zlib niftiio 
${OPENNI_LIBRARY} ${GLEW_LIBRARY} ${QGLVIEWER} ${OpenCV_LIBS})
qt5_use_modules(MappingAndRelocalisation OpenGL Gui Widgets Xml)
