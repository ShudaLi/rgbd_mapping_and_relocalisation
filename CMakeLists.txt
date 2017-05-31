cmake_minimum_required (VERSION 2.8)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0043 NEW)
  cmake_policy(SET CMP0020 NEW)
endif()

project (Btl)

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

set(CUDA_NVCC_FLAGS; -arch=sm_30 )#cmak

find_package(Qt5OpenGL REQUIRED PATHS $ENV{QtOpenGL})
find_package(Qt5Core REQUIRED  PATHS $ENV{QtCore})
find_package(Qt5Gui REQUIRED  PATHS $ENV{QtGui})
find_package(Qt5Xml REQUIRED  PATHS $ENV{QtXml})

#Notice that the headers only include the headers that needs to be processed by moc.
QT5_WRAP_CPP(HEADERS_MOC ${HEADERS})


if( UNIX )
	find_package(Eigen3 REQUIRED)
	include_directories(${EIGEN3_INCLUDE_DIR})

	message(STATUS "platform: Linux")

	find_package(OpenGL REQUIRED)
	include_directories(${OpenGL_INCLUDE_DIRS})
	link_directories(${OpenGL_LIBRARY_DIRS})
	add_definitions(${OpenGL_DEFINITIONS})

	find_package(GLUT REQUIRED)
	include_directories(${GLUT_INCLUDE_DIRS})
	link_directories(${GLUT_LIBRARY_DIRS})
	add_definitions(${GLUT_DEFINITIONS})

	find_package(GLEW REQUIRED)
	find_package(ZLIB REQUIRED)

	include_directories ( $ENV{SOPHUS_INC} )
	include_directories ( $ENV{OPENNI2_INC} )
	include_directories ( $ENV{NIFTI_INC} )
	include_directories ( $ENV{NIFTI_INC}/nifti )

	include_directories ( ${PROJECT_SOURCE_DIR}/common )
	include_directories ( ${PROJECT_SOURCE_DIR}/rgbd )

	link_directories ( $ENV{ZLIB_LIB} )
	link_directories ( $ENV{NIFTI_LIB} )
	link_directories ( $ENV{OPENNI2_LIB} )
	link_directories ( $ENV{BOOST_LIB} )
	link_directories ( $ENV{FREEGLUT_LIB} )
	link_directories ( $ENV{GLEW_LIB} )
	link_directories ( ${PROJECT_SOURCE_DIR}/lib )
elseif( WIN32 )
	message(STATUS "platform: Windows")

	include_directories ( $ENV{OPENCV_NEW}/include )
	include_directories ( $ENV{EIGEN_INC} )
	include_directories ( $ENV{SOPHUS_INC} )
	include_directories ( $ENV{OPENNI2_INC} )
    include_directories ( $ENV{BOOST_INC} )
    include_directories ( $ENV{FREEGLUT_INC} )
    include_directories ( $ENV{GLEW_INC} )
	include_directories ( $ENV{NIFTI_INC} )
	include_directories ( $ENV{NIFTI_INC}/nifti )

	include_directories ( ${PROJECT_SOURCE_DIR}/common )
	include_directories ( ${PROJECT_SOURCE_DIR}/rgbd )

	link_directories ( $ENV{ZLIB_LIB} )
	link_directories ( $ENV{NIFTI_LIB} )
	link_directories ( $ENV{OPENNI2_LIB} )
	link_directories ( $ENV{BOOST_LIB} )
	link_directories ( $ENV{FREEGLUT_LIB} )
	link_directories ( $ENV{GLEW_LIB} )
	link_directories ( ${PROJECT_SOURCE_DIR}/lib )
endif()

set(BUILD_SHARED_LIBS  TRUE)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include("mapping_and_relocalisation/CMakeLists.txt")

#find rgbd

	file(GLOB OTHER_HEADERS "${PARENT_DIR}/common/[a-zA-Z]*.h" "${PARENT_DIR}/common/[a-zA-Z]*.hpp"   )
	message(STATUS "other header names: ${OTHER_HEADERS}")
	
	file(GLOB RGBD_HEADERS "rgbd/[a-zA-Z]*.h" "rgbd/[a-zA-Z]*.hpp"   )
	message(STATUS "header names: ${RGBD_HEADERS}")
	
	file(GLOB RGBD_SOURCES "rgbd/[a-zA-Z]*.cpp")
	message(STATUS "source names: ${RGBD_SOURCES}")
	
	file(GLOB RGBD_CUDA_HEADERS "rgbd/[a-zA-Z]*.cuh")
	message(STATUS "cuda header names: ${RGBD_CUDA_HEADERS}")

	file(GLOB RGBD_CUDA_SOURCES "rgbd/[a-zA-Z]*.cu")
	message(STATUS "cuda source names: ${RGBD_CUDA_SOURCES}")
	
	file(GLOB KINFU_HEADERS "rgbd/kinfu/[a-zA-Z]*.h" "rgbd/[a-zA-Z]*.hpp"   )
	message(STATUS "header names: ${KINFU_HEADERS}")
	
	file(GLOB KINFU_SOURCES "rgbd/kinfu/[a-zA-Z]*.cpp")
	message(STATUS "source names: ${KINFU_SOURCES}")
	
	file(GLOB KINFU_CUDA_HEADERS "rgbd/kinfu/[a-zA-Z]*.cuh")
	message(STATUS "cuda header names: ${KINFU_CUDA_HEADERS}")

	file(GLOB KINFU_CUDA_SOURCES "rgbd/kinfu/[a-zA-Z]*.cu")
	message(STATUS "cuda source names: ${KINFU_CUDA_SOURCES}")
	
	cuda_add_library( Rgbd SHARED ${RGBD_HEADERS} ${OTHER_HEADERS} ${RGBD_SOURCES} ${RGBD_CUDA_HEADERS} ${RGBD_CUDA_SOURCES} ${KINFU_HEADERS} ${KINFU_SOURCES} ${KINFU_CUDA_HEADERS} ${KINFU_CUDA_SOURCES})
if(MSVC)
	set(NIFTIIOLIB optimized niftiio debug niftiiod )	
	set(ZNZLIB optimized znz debug znzd )	
	set(ZLIB optimized zlib debug zlibd )	
	set(FREEGLUTLIB freeglut)
	set(GLEWLIB glew32)
	set(GLLIB GLU32 opengl32)
	set(OPENNI2LIB OpenNI2)
elseif(UNIX)
	set(FREEGLUTLIB ${GLUT_glut_LIBRARY})
	set(GLEWLIB ${GLEW_LIBRARY})
	set(GLLIB ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY})
	set(OPENNI2LIB OpenNI2)
	set(ZLIB ${ZLIB_LIBRARY})
	set(NIFTIIOLIB niftiio )
endif()

	target_link_libraries( Rgbd ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY} ${Boost_DATE_TIME_LIBRARY} ${Boost_SERIALIZATION_LIBRARY} ${Boost_THREAD_LIBRARY} ${Boost_CHRONO_LIBRARY}
	opencv_core opencv_features2d opencv_xfeatures2d opencv_imgproc opencv_cudaimgproc opencv_cudawarping opencv_cudafeatures2d opencv_cudaarithm
    ${FREEGLUTLIB} ${GLLIB} ${GLEWLIB} ${CUDALIB} ${OPENNI2LIB} ${NIFTIIOLIB} ${ZNZLIB} ${ZLIB} ) 

if(MSVC)
      	set_target_properties(Rgbd PROPERTIES
      		DEBUG_POSTFIX "${LIB_TARGET_SUFFIX}d"
      		RELEASE_POSTFIX "${LIB_TARGET_SUFFIX}"
      		MINSIZEREL_POSTFIX "${LIB_TARGET_SUFFIX}"
      		RELWITHDEBINFO_POSTFIX "${LIB_TARGET_SUFFIX}"
      	)
endif()

install( TARGETS Rgbd        DESTINATION "${PROJECT_SOURCE_DIR}/lib" )



