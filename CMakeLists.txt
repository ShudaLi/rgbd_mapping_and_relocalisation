CMAKE_MINIMUM_REQUIRED (VERSION 2.8)

IF(COMMAND cmake_policy)
  cmake_policy(SET CMP0043 NEW)
  cmake_policy(SET CMP0020 NEW)
ENDIF()

PROJECT (RgbdMappingAndRelocalisation)

find_package( OpenCV REQUIRED PATHS $ENV{OPENCV_NEW} NO_DEFAULT_PATH COMPONENTS core imgcodecs features2d  videoio highgui video calib3d cudaarithm )

find_package( CUDA )
include(FindCUDA)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif()

if(UNIX)
    IF( NOT CMAKE_BUILD_TYPE )
   		SET( CMAKE_BUILD_TYPE Release )
	ENDIF()
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
endif()

set(CUDA_NVCC_FLAGS; -arch=sm_30 )#cmake

#find_package(Qt5OpenGL REQUIRED PATHS $ENV{QtOpenGL})
#find_package(Qt5Core REQUIRED  PATHS $ENV{QtCore})
#find_package(Qt5Gui REQUIRED  PATHS $ENV{QtGui})
#find_package(Qt5Xml REQUIRED  PATHS $ENV{QtXml})
include_directories ( ${PROJECT_SOURCE_DIR}/external_libraries/sophus/include/ )
include_directories ( ${PROJECT_SOURCE_DIR}/external_libraries/ )
include_directories ( ${PROJECT_SOURCE_DIR}/external_libraries/zlib/ )
#Notice that the headers only include the headers that needs to be processed by moc.
#QT5_WRAP_CPP(HEADERS_MOC ${HEADERS})
ADD_SUBDIRECTORY(external_libraries)
ADD_SUBDIRECTORY(rgbd)
ADD_SUBDIRECTORY(mapping_and_relocalisation)
	



