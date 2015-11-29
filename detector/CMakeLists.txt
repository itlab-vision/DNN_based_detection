CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(PROJECT_NAME detector)
project(${PROJECT_NAME})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

FILE(GLOB hdrs "*.h*" "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h*")
FILE(GLOB srcs "*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

#----------------------------------------------------------------------------------#
option(WITH_CAFFE "Link with Caffe library." OFF)
option(WITH_CPU_ONLY "Caffe will be used in CPU or GPU mode" OFF)
option(WITH_TORCH "Link with Lua to support Torch-based classifier." OFF)
option(WITH_MPI "Link MPI libraries." OFF)
option(PAR_PYRAMID "Parallel by the pyramid of resized images." OFF)
option(PAR_SET_IMAGES "Parallel by initial set of images." OFF)
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
    add_definitions(-DHAVE_CAFFE)
endif()

if (WITH_CPU_ONLY)
   message(STATUS "Caffe in CPU_ONLY mode.")
   add_definitions(-DCPU_ONLY)
endif()

set(CMAKE_SCRIPT_DIR ${CMAKE_SOURCE_DIR}/CMakeScripts)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SCRIPT_DIR})

if (WITH_TORCH)
    find_package(Lua REQUIRED)
    if (LUA_FOUND)
        include_directories(${LUA_INCLUDE_DIR})
        target_link_libraries(${PROJECT_NAME} ${LUA_LIBRARIES})
    endif()
    find_package(Torch REQUIRED)
    if (TORCH_FOUND)
        include_directories(${TORCH_INCLUDE_DIR})
        target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
    endif()
    add_definitions(-DHAVE_TORCH)
endif()

if (WITH_MPI)
    find_package(MPI REQUIRED)
    if (MPI_FOUND)
        include_directories(${MPI_CXX_INCLUDE_PATH})
        target_link_libraries(${PROJECT_NAME} ${MPI_CXX_LIBRARIES})
        if (MPI_CXX_COMPILE_FLAGS)
            set_target_properties(${PROJECT_NAME} PROPERTIES
                COMPILE_FLAGS ${MPI_CXX_COMPILE_FLAGS})
        endif()
        if (MPI_CXX_LINK_FLAGS)
            set_target_properties(${PROJECT_NAME} PROPERTIES
                LINK_FLAGS ${MPI_CXX_LINK_FLAGS})
        endif()
    endif()
    add_definitions(-DHAVE_MPI)
endif()

if (PAR_PYRAMID)
    add_definitions(-DPAR_PYRAMID)
endif()
if (PAR_SET_IMAGES)
    add_definitions(-DPAR_SET_IMAGES)
endif()
#----------------------------------------------------------------------------------#

add_subdirectory(samples)
add_subdirectory(gtest)
add_subdirectory(test)

#----------------------------------------------------------------------------------#
message(STATUS ${PROJECT_NAME})
message(STATUS "-------------------------------------------------------")
message(STATUS "Caffe support     ${WITH_CAFFE}")
message(STATUS "   CPU_ONLY:      ${WITH_CPU_ONLY}")
message(STATUS "Torch support     ${WITH_TORCH}")
message(STATUS "MPI support       ${WITH_MPI}")
message(STATUS "   PAR_PYRAMID:   ${PAR_PYRAMID}")
message(STATUS "   PAR_SET_IMAGES:${PAR_SET_IMAGES}")
message(STATUS "-------------------------------------------------------")
