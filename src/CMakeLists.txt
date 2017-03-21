cmake_minimum_required (VERSION 3.1)
project( RealtimeSegment )

set( VERSION 0.1.0 )

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin" )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib" )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib" )

### Compilation configuration ###
if( MSVC )
    ### Enable parallel compilation for Visual Studio
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /bigobj" )
    # set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w" )
else()
    #### Libigl requires a modern C++ compiler that supports c++11
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
    # set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w" )
endif()

### Third party Find?.CMake
set( CMAKE_MODULE_PATH  "${CMAKE_CURRENT_SOURCE_DIR}/../cmake" )


### Set enviroments variables
set( RTSEGMENT               "${PROJECT_NAME}"                  )
set( RTSEGMENT_ROOT          "${PROJECT_SOURCE_DIR}"            )
set( RTSEGMENT_INCLUDE_DIRS  "${PROJECT_SOURCE_DIR}/../include" )
set( RTSEGMENT_DEFINITIONS   "" )
set( RTSEGMENT_EXTRA_LIBS    "" )

### Third party find manualy
set( KINECT2_INCLUDE_DIRS "KINECT2_INCLUDE_DIRS NOT FOUND" CACHE PATH "Include dirs of Kinect2" )
set( KINECT_INCLUDE_DIRS  "KINECT_INCLUDE_DIRS NOT FOUND"  CACHE PATH "Include dirs of Kinect"  )
set( KINECT2_LIBS "KINECT2_LIBS NOT FOUND" CACHE FILEPATH "Libs path of Kinect2" )
set( KINECT_LIBS  "KINECT_LIBS NOT FOUND"  CACHE FILEPATH "Libs path of Kinect"  )

option( USE_REALSENSE "Realsense"   OFF )
option( USE_KINECT2   "Kinect2.0"   ON  )
option( USE_KINECT1   "Kinect1.8"   OFF )

### Third party dependency
find_package( OpenCV  REQUIRED )
message( STATUS "Using OpenCV-${OpenCV_VERSION}" )

option( WITH_EIGEN    "Eigen3"      OFF )
if( ${WITH_EIGEN} )
find_package( Eigen3  REQUIRED )
message( STATUS "Using Eigen3-${EIGEN3_VERSION}" )
list( APPEND RTSEGMENT_INCLUDE_DIRS  "${EIGEN3_INCLUDE_DIR}" )
endif()

set( REALSENSE_INCLUDE_DIRS  "REALSENSE_INCLUDE_DIRS NOT FOUND" CACHE PATH     "Include dirs of Realsense"      )
set( REALSENSE_LIBS_DEBUG    "REALSENSE_LIBS_DEBUG NOT FOUND"   CACHE FILEPATH "Libs path of Realsense DEBUG"   )
set( REALSENSE_LIBS_RELEASE  "REALSENSE_LIBS_RELEASE NOT FOUND" CACHE FILEPATH "Libs path of Realsense RELEASE" )
set( REALSENSE_LIBS $<$<CONFIG:Debug>:${REALSENSE_LIBS_DEBUG}>$<$<CONFIG:Release>:${REALSENSE_LIBS_RELEASE}>$<$<CONFIG:MinSizeRel>:${REALSENSE_LIBS_DEBUG}>$<$<CONFIG:RelWithDebInfo>:${REALSENSE_LIBS_RELEASE}>)

set( GLFW_INCLUDE_DIRS  "GLFW_INCLUDE_DIRS NOT FOUND" CACHE PATH     "Include dirs of glfw"      )
set( GLEW_INCLUDE_DIRS  "GLEW_INCLUDE_DIRS NOT FOUND" CACHE PATH     "Include dirs of glew"      )
set( GLM_INCLUDE_DIRS   "GLM_INCLUDE_DIRS NOT FOUND" CACHE PATH     "Include dirs of glm"      )



if( ${USE_KINECT2} )
    list( APPEND RTSEGMENT_INCLUDE_DIRS  "${KINECT2_INCLUDE_DIRS}" )
    list( APPEND RTSEGMENT_EXTRA_LIBS    "${KINECT2_LIBS}"         )
elseif( ${USE_KINECT1} )
    list( APPEND RTSEGMENT_INCLUDE_DIRS  "${KINECT_INCLUDE_DIRS}" )
    list( APPEND RTSEGMENT_EXTRA_LIBS    "${KINECT_LIBS}"         )
elseif( ${USE_REALSENSE} )
    list( APPEND RTSEGMENT_INCLUDE_DIRS  "${REALSENSE_INCLUDE_DIRS}" )
    list( APPEND RTSEGMENT_EXTRA_LIBS    "${REALSENSE_LIBS}"         )
endif()

list( APPEND RTSEGMENT_INCLUDE_DIRS      "${OpenCV_INCLUDE_DIRS}" )
list( APPEND RTSEGMENT_EXTRA_LIBS        "${OpenCV_LIBS}"         )


configure_file( ${PROJECT_SOURCE_DIR}/cmake/types.h.in ${PROJECT_SOURCE_DIR}/../include/types.h )
# Build source files
file( GLOB RTSEGMENT_SOURCE_FILES *.cpp *.h )

include_directories( ${RTSEGMENT_INCLUDE_DIRS} )
add_definitions(     ${RTSEGMENT_DEFINITIONS}  )
# add_library( ${PROJECT_NAME} STATIC ${RTSEGMENT_SOURCE_FILES} )
add_executable( ${PROJECT_NAME} ${RTSEGMENT_SOURCE_FILES}      )
target_link_libraries( ${PROJECT_NAME} ${RTSEGMENT_EXTRA_LIBS} )
