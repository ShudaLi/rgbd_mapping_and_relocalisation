###################
# UseOpenNI.cmake #
###################

IF(MSVC_IDE)
	FIND_PATH(GLEW_ROOT LICENSE HINTS "C:/all_libs/glew")
ELSEIF("${CMAKE_SYSTEM}" MATCHES "Linux")
	#FIND_PATH(OPENNI_ROOT LICENSE HINTS ~/Software/OpenNI2)
ELSE()
	MESSAGE(FATAL_ERROR "OpenNI not currently set up to work on this platform.")
ENDIF()

FIND_PATH(QGLVIEWER_INC "QGLViewer/qglviewer.h" HINTS "${QGLVIEWER_ROOT}/QGLViewer/")
FIND_LIBRARY(QGLVIEWER_LIBRARY QGLViewerd2 HINTS "${QGLVIEWER_ROOT}/QGLViewer/")

INCLUDE_DIRECTORIES(${GLEW_INC})
  
