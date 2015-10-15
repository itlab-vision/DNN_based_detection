CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(PROJECT_NAME detector)
project(${PROJECT_NAME})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

FILE(GLOB hdrs "*.h*" "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h*")
FILE(GLOB srcs "*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

#----------------------------------------------------------------------------------#
option(WITH_CAFFE "Link with Caffe library." OFF)
option(WITH_CPU_ONLY "Caffe will be used in CPU or GPU mode" OFF)
option(WITH_LUA "Link with Lua to support Torch-based classifier." OFF)
#----------------------------------------------------------------------------------#
# if NOT WITH_CAFFE then exclude classifier implementation based on Caffe library
if (NOT WITH_CAFFE)
    list(REMOVE_ITEM hdrs "${CMAKE_CURRENT_SOURCE_DIR}/include/caffe_classifier.hpp")
    list(REMOVE_ITEM srcs "${CMAKE_CURRENT_SOURCE_DIR}/src/caffe_classifier.cpp")
endif()
# if NOT WITH_LUA then exclude classifier implementation based on Torch library
if (NOT WITH_LUA)
    list(REMOVE_ITEM hdrs "${CMAKE_CURRENT_SOURCE_DIR}/include/torch_classifier.hpp")
    list(REMOVE_ITEM srcs "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_classifier.cpp")
endif()
#----------------------------------------------------------------------------------#

add_library(${PROJECT_NAME} STATIC ${srcs} ${hdrs})

if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall")
endif()

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

find_package(OpenMP QUIET)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#----------------------------------------------------------------------------------#
if (WITH_CAFFE)
    find_package(Caffe REQUIRED)
    # Seems that after successfull Caffe search CAFFE_FOUND variable is not defined.
    if (Caffe_INCLUDE_DIRS)
        include_directories(${Caffe_INCLUDE_DIRS})
        target_link_libraries(${PROJECT_NAME} ${Caffe_LIBRARIES})
    endif()
endif()

if (WITH_CPU_ONLY)
   message(STATUS "Caffe in CPU_ONLY mode.")
   add_definitions(-DCPU_ONLY)
endif()

if (WITH_LUA)
    find_package(Lua REQUIRED)
    if (LUA_FOUND)
        include_directories(${LUA_INCLUDE_DIR})
        target_link_libraries(${PROJECT_NAME} ${LUA_LIBRARIES})
    endif()
endif()
#----------------------------------------------------------------------------------#


add_subdirectory(samples)
#----------------------------------------------------------------------------------#
message(STATUS ${PROJECT_NAME})
message(STATUS "-------------------------------------------------------")
message(STATUS "Caffe support     ${WITH_CAFFE}")
message(STATUS "      CPY_ONLY:   ${WITH_CPU_ONLY}")
message(STATUS "Lua support       ${WITH_LUA}")
message(STATUS "-------------------------------------------------------")
