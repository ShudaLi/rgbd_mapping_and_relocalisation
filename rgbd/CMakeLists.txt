CMAKE_MINIMUM_REQUIRED (VERSION 2.8)

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

	include_directories ( $ENV{OPENNI2_INC} )
	include_directories ( $ENV{NIFTI_INC} )
	include_directories ( $ENV{NIFTI_INC}/nifti )

	include_directories ( ${PROJECT_SOURCE_DIR}/common )
	include_directories ( ${PROJECT_SOURCE_DIR}/rgbd )

	link_directories ( $ENV{NIFTI_LIB} )
	link_directories ( $ENV{OPENNI2_LIB} )

	link_directories ( $ENV{FREEGLUT_LIB} )
	link_directories ( $ENV{GLEW_LIB} )
	link_directories ( ${PROJECT_SOURCE_DIR}/lib )
elseif( WIN32 )
	message(STATUS "platform: Windows")

	include_directories ( $ENV{OPENCV_NEW}/include )
	include_directories ( $ENV{OPENNI2_INC} )
    include_directories ( $ENV{FREEGLUT_INC} )
    include_directories ( $ENV{GLEW_INC} )
	include_directories ( $ENV{NIFTI_INC} )
	include_directories ( $ENV{NIFTI_INC}/nifti )

	include_directories ( ${PROJECT_SOURCE_DIR}/common )
	include_directories ( ${PROJECT_SOURCE_DIR}/rgbd )

	link_directories ( $ENV{NIFTI_LIB} )
	link_directories ( $ENV{OPENNI2_LIB} )
	link_directories ( $ENV{FREEGLUT_LIB} )
	link_directories ( $ENV{GLEW_LIB} )
	link_directories ( ${PROJECT_SOURCE_DIR}/lib )
endif()


set(BUILD_SHARED_LIBS  TRUE)

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

target_link_libraries( Rgbd 
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
install( TARGETS Rgbd DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/../" )




