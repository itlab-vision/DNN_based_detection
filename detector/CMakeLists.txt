CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(PROJECT_NAME detector)
project(${PROJECT_NAME})

if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(OpenCV REQUIRED)

find_package(Caffe REQUIRED)

set(Lua_LIBS
    "/usr/local/lib/libluajit.so"
    "/usr/local/lib/libluaT.so"
    "/usr/local/lib/libTH.so"
    "/usr/local/lib/lua/5.1/libtorch.so"
    "/usr/local/lib/lua/5.1/libnn.so")
set(Lua_INCLUDE "/usr/local/include")
include_directories(Lua_INCLUDE)

set(FACEDETECTOR_LIBRARY ${PROJECT_NAME})
set(FACEDETECTOR_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories(${FACEDETECTOR_INCLUDE} ${Caffe_INCLUDE_DIRS})

set(LIBRARY_DEPS ${OpenCV_LIBS} ${Caffe_LIBS})
set(LUA_LIBRARY ${Lua_LIBS})

add_subdirectory(src)
add_subdirectory(samples)
