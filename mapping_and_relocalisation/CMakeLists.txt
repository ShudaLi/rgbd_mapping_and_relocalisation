cmake_minimum_required (VERSION 2.8)
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0020 NEW)
endif()

project (MappingAndRelocalisation)

find_package(OpenGL REQUIRED)
find_package(Qt5OpenGL REQUIRED)
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5Xml REQUIRED)
FIND_PACKAGE(Boost COMPONENTS system filesystem date_time chrono)
IF (Boost_FOUND)
    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
    ADD_DEFINITIONS( "-DHAS_BOOST" )
ENDIF()

QT5_WRAP_CPP(HEADERS_MOC ${HEADERS})


    include_directories ( $ENV{QGLVIEWER_INC} )
	include_directories ( $ENV{EIGEN_INC} )
	include_directories ( $ENV{OPENNI2_INC} )
    include_directories ( $ENV{BOOST_INC} ) 
    include_directories ( $ENV{FREEGLUT_INC} )
    include_directories ( $ENV{GLEW_INC} )
	include_directories ( $ENV{CUDA_INC_PATH} )
	include_directories ( ${PROJECT_SOURCE_DIR}/rgbd )

    link_directories ( $ENV{QGLVIEWER_LIB} )
	link_directories ( $ENV{OPENNI2_LIB} )
	link_directories ( $ENV{BOOST_LIB} )
	link_directories ( $ENV{FREEGLUT_LIB} )
	link_directories ( $ENV{GLEW_LIB} )
	link_directories ( ${PROJECT_SOURCE_DIR}/lib )
	link_directories ( $ENV{ZLIB_LIB} )
	
	
if(WIN32)
        set(RGBDLIB optimized Rgbd debug Rgbdd)
        set(QGLVIEWER  optimized QGLViewer2  debug QGLViewerd2 )
		set(ZLIB optimized zlib debug zlibd )	
		set(EXTRA_LIBS ${RGBDLIB} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} OpenNI2
			${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY} ${Boost_DATETIME_LIBRARY} 
			freeglut GLU32  glew32 GLU32 opengl32 ${QGLVIEWER} ${QT_LIBRARIES} ${OpenCV_LIBS})
elseif(UNIX)
	include_directories ( ${PROJECT_SOURCE_DIR}/rgbd )

	link_directories ( ${PROJECT_SOURCE_DIR}/lib )
	set(RGBDLIB Rgbd)
	set(FREEGLUTLIB ${GLUT_glut_LIBRARY})
	set(GLEWLIB ${GLEW_LIBRARY})
	set(GLLIB ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY})
	set(OPENNI2LIB ${OPENNI2_LIBRARY})
	set(ZLIB ${ZLIB_LIBRARY})
	set(QGLVIEWER  QGLViewer-qt5 )
	set(EXTRA_LIBS ${RGBDLIB} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} OpenNI2
		${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SERIALIZATION_LIBRARY} 
        ${FREEGLUTLIB} ${GLLIB} ${GLEWLIB} ${QGLVIEWER} ${QT_LIBRARIES} ${OpenCV_LIBS})
endif()

add_executable(MappingAndRelocalisation mapping_and_relocalisation/Main.cpp  mapping_and_relocalisation/MultiViewer.cpp mapping_and_relocalisation/Data4Viewer.h mapping_and_relocalisation/Data4Viewer.cpp mapping_and_relocalisation/DataLive.cpp mapping_and_relocalisation/MultiViewer.h mapping_and_relocalisation/DataLive.h)
target_link_libraries( MappingAndRelocalisation ${EXTRA_LIBS} ${ZLIB} )
qt5_use_modules(MappingAndRelocalisation OpenGL Gui Widgets Xml)
