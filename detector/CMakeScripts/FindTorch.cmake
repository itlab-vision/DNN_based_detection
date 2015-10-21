# - Try to find Torch
#
# The following variables are optionally searched for defaults
#  TORCH_ROOT_DIR:            Base directory where all TORCH components are found
#
# The following are set after configuration is done:
#  TORCH_FOUND
#  TORCH_INCLUDE_DIR
#  TORCH_LIBRARIES

include(FindPackageHandleStandardArgs)

set(TORCH_ROOT_DIR
	$ENV{PATH}
	$ENV{LD_LIBRARY_PATH}	
)

find_path(TORCH_INCLUDE_DIR_LUA lua.hpp PATHS ${TORCH_ROOT_DIR})
find_path(TORCH_INCLUDE_DIR_LUAT luaT.h PATHS ${TORCH_ROOT_DIR})
find_path(TORCH_INCLUDE_DIR_TH TH/TH.h PATHS ${TORCH_ROOT_DIR})

find_library(TORCH_LIBRARY_LUAT luaT
	PATHS ${TORCH_ROOT_DIR} PATH_SUFFIXES lib)
find_library(TORCH_LIBRARY_LUATH TH
	PATHS ${TORCH_ROOT_DIR} PATH_SUFFIXES lib)
find_library(TORCH_LIBRARY_LUAJIT luajit
	PATHS ${TORCH_ROOT_DIR} PATH_SUFFIXES lib)
find_library(TORCH_LIBRARY torch
	PATHS ${TORCH_ROOT_DIR} PATH_SUFFIXES lib lua/5.1)
find_library(TORCH_LIBRARY_NN nn
	PATHS ${TORCH_ROOT_DIR} PATH_SUFFIXES lib lua/5.1)

set(TORCH_INCLUDE_DIR
	${TORCH_INCLUDE_DIR_LUA}
	${TORCH_INCLUDE_DIR_LUAT}
	${TORCH_INCLUDE_DIR_TH})
set(TORCH_LIBRARIES
	${TORCH_LIBRARY_LUAT}
	${TORCH_LIBRARY_LUATH}
	${TORCH_LIBRARY}
	${TORCH_LIBRARY_NN}
	${TORCH_LIBRARY_LUAJIT})

find_package_handle_standard_args(TORCH DEFAULT_MSG
	TORCH_INCLUDE_DIR TORCH_LIBRARIES)

if (TORCH_FOUND)
	set(TORCH_INCLUDE_DIR ${TORCH_INCLUDE_DIR})
	set(TORCH_LIBRARIES ${TORCH_LIBRARIES})
	message(STATUS "Torch found in" ${TORCH_INCLUDE_DIR} ${TORCH_LIBRARIES})
else()
	message(FATAL_ERROR "Torch was not found.")
endif()
