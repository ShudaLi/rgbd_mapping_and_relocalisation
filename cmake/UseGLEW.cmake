###################
# UseOpenNI.cmake #
###################


IF(MSVC_IDE)
	FIND_PATH(GLEW_ROOT LICENSE HINTS "C:/all_libs/glew")
ELSEIF("${CMAKE_SYSTEM}" MATCHES "Linux")
	#FIND_PATH(OPENNI_ROOT LICENSE HINTS ~/Software/OpenNI2)
ELSE()
	MESSAGE(FATAL_ERROR "Glew not currently set up to work on this platform.")
ENDIF()

FIND_PATH(GLEW_INC "GL/glew.h" HINTS "${GLEW_ROOT}/include/")
FIND_LIBRARY(GLEW_LIBRARY glew32 HINTS "${GLEW_ROOT}/bin/Release/x64/" "${GLEW_ROOT}/lib/Release/x64/" )

INCLUDE_DIRECTORIES(${GLEW_INC})
  
