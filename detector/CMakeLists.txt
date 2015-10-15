CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(PROJECT_NAME detector)
project(${PROJECT_NAME})

option(WITH_CPU_ONLY "Caffe will be used in CPU or GPU mode" OFF)
if (WITH_CPU_ONLY)
   message(STATUS "Caffe in CPU_ONLY mode.")
   add_definitions(-DCPU_ONLY)
endif()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
FILE(GLOB hdrs "*.h*" "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h*")
FILE(GLOB srcs "*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
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

set(HAVE_CAFFE OFF)
find_package(Caffe REQUIRED)
# Seems that after successfull Caffe search CAFFE_FOUND variable is not defined.
if (Caffe_INCLUDE_DIRS)
    include_directories(${Caffe_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${Caffe_LIBRARIES})
    set(HAVE_CAFFE ON)
    # target_compile_definitions is not supported in cmake 2.8.7. :(
    # target_compile_definitions(${PROJECT_NAME} PRIVATE HAVE_CAFFE)
    set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY COMPILE_DEFINITIONS "HAVE_CAFFE")
endif()

set(HAVE_LUA OFF)
find_package(Lua QUIET)
if (LUA_FOUND)
    include_directories(${LUA_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} ${LUA_LIBRARIES})
    set(HAVE_LUA ON)
    # target_compile_definitions(${PROJECT_NAME} PRIVATE HAVE_LUA)
    set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY COMPILE_DEFINITIONS "HAVE_LUA")
endif()

add_subdirectory(samples)

message(STATUS ${PROJECT_NAME})
message(STATUS "-------------------------------------------------------")
message(STATUS "Caffe support     ${HAVE_CAFFE}")
message(STATUS "      CPY_ONLY:   ${WITH_CPU_ONLY}")
message(STATUS "Lua support       ${HAVE_LUA}")
message(STATUS "-------------------------------------------------------")
