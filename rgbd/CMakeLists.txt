CMAKE_MINIMUM_REQUIRED (VERSION 2.8)

find_package( OpenCV REQUIRED 
PATHS "$ENV{OPENCV_NEW}" "C:/Program Files (x86)/OpenCV/Lib" 
COMPONENTS core imgcodecs features2d  videoio highgui video calib3d cudaarithm 
cudaimgproc cudawarping cudafeatures2d )

include(${PROJECT_SOURCE_DIR}/cmake/UseOpenNI.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/UseGLEW.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/UseQGLViewer.cmake)

include_directories( ${OpenCV_INCLUDE_DIRS} )

if( UNIX )
	message(STATUS "platform: Linux")

	find_package(OpenGL REQUIRED)
	include_directories(${OpenGL_INCLUDE_DIRS})
	link_directories(${OpenGL_LIBRARY_DIRS})
	add_definitions(${OpenGL_DEFINITIONS})

	find_package(GLEW REQUIRED)
elseif( WIN32 )
	message(STATUS "platform: Windows")
endif()

include_directories ( ${PROJECT_SOURCE_DIR}/common )
include_directories ( ${PROJECT_SOURCE_DIR}/rgbd )
include_directories ( ${PROJECT_SOURCE_DIR}/external_libraries/ )
include_directories ( ${PROJECT_SOURCE_DIR}/external_libraries/sophus/include )
include_directories ( ${PROJECT_SOURCE_DIR}/external_libraries/nifticlib/niftilib/ )
include_directories ( ${PROJECT_SOURCE_DIR}/external_libraries/nifticlib/znzlib/ )

#find source files
file(GLOB OTHER_HEADERS "${PARENT_DIR}/common/[a-zA-Z]*.h" "${PARENT_DIR}/common/[a-zA-Z]*.hpp"   )
message(STATUS "other header names: ${OTHER_HEADERS}")

file(GLOB RGBD_HEADERS "[a-zA-Z]*.h" "[a-zA-Z]*.hpp"   )
message(STATUS "header names: ${RGBD_HEADERS}")

file(GLOB RGBD_SOURCES "[a-zA-Z]*.cpp")
message(STATUS "source names: ${RGBD_SOURCES}")

file(GLOB RGBD_CUDA_HEADERS "[a-zA-Z]*.cuh")
message(STATUS "cuda header names: ${RGBD_CUDA_HEADERS}")

file(GLOB RGBD_CUDA_SOURCES "[a-zA-Z]*.cu")
message(STATUS "cuda source names: ${RGBD_CUDA_SOURCES}")

file(GLOB KINFU_HEADERS "kinfu/[a-zA-Z]*.h" "[a-zA-Z]*.hpp"   )
message(STATUS "header names: ${KINFU_HEADERS}")

file(GLOB KINFU_SOURCES "kinfu/[a-zA-Z]*.cpp")
message(STATUS "source names: ${KINFU_SOURCES}")

file(GLOB KINFU_CUDA_HEADERS "kinfu/[a-zA-Z]*.cuh")
message(STATUS "cuda header names: ${KINFU_CUDA_HEADERS}")

file(GLOB KINFU_CUDA_SOURCES "kinfu/[a-zA-Z]*.cu")
message(STATUS "cuda source names: ${KINFU_CUDA_SOURCES}")

cuda_add_library( Rgbd ${RGBD_HEADERS} ${OTHER_HEADERS} ${RGBD_SOURCES} ${RGBD_CUDA_HEADERS} ${RGBD_CUDA_SOURCES} ${KINFU_HEADERS} ${KINFU_SOURCES} ${KINFU_CUDA_HEADERS} ${KINFU_CUDA_SOURCES})