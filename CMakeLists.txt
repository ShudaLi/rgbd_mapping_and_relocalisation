CMAKE_MINIMUM_REQUIRED (VERSION 2.8)

IF(COMMAND cmake_policy)
  cmake_policy(SET CMP0043 NEW)
  cmake_policy(SET CMP0020 NEW)
ENDIF()

PROJECT (RgbdMappingAndRelocalisation)

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

ADD_SUBDIRECTORY(external_libraries)
ADD_SUBDIRECTORY(rgbd)
ADD_SUBDIRECTORY(mapping_and_relocalisation)
	



