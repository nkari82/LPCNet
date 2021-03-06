cmake_minimum_required(VERSION 3.10.2)

project(lpcnet)

cmake_host_system_information(RESULT _os QUERY OS_NAME)
cmake_host_system_information(RESULT _platform QUERY OS_PLATFORM)

message(STATUS "OS: " ${_os})
message(STATUS "Platform: " ${_platform})

option(BUILD_STATIC "Build shared library" ON)
option(BUILD_SHARED "Build static library" OFF)
option(HAVE_GOMP "Have gomp" OFF)

if( MSVC )
 set (CMAKE_C_FLAGS "-D__AVX__")
 set (CMAKE_CXX_FLAGS "/std:c++17 /GR -D__AVX__")
else()
 set (CMAKE_C_FLAGS "-O3 -g -mavx -mfma")
 set (CMAKE_CXX_FLAGS "-std=c++17 -O3 -g -mavx -mfma")
endif()

if( MSVC )
  add_compile_options( "/EHsc" )
  add_compile_options( "/sdl-" )
endif()

add_subdirectory(./thirdparty/cppglob)
list(APPEND EXTRA_LIBS cppglob_static)
list(APPEND EXTRA_INCLUDES "./thirdparty/cppglob/include")
  
add_subdirectory(./thirdparty/sox)
list(APPEND EXTRA_LIBS libsox lpc10)
list(APPEND EXTRA_INCLUDES "./thirdparty/sox/src")

list(APPEND EXTRA_INCLUDES "./thirdparty/cxxopts/include")
  
add_library(lpcnet STATIC
	src/common.c
	src/freq.c
	src/kiss_fft.c
	src/celt_lpc.c
	src/pitch.c
	src/lpcnet.c
	src/lpcnet_dec.c
	src/lpcnet_enc.c
	src/nnet.c
	src/ceps_codebooks.c)
target_include_directories(lpcnet PUBLIC ./include PRIVATE ./src)

add_executable (dump_data src/dump_data.cpp)
target_include_directories(dump_data PRIVATE ./include ./src ${EXTRA_INCLUDES})
target_link_libraries (dump_data LINK_PUBLIC lpcnet ${EXTRA_LIBS})
if(!MSVC)
target_link_libraries (dump_data LINK_PUBLIC lpcnet m)
endif()

add_executable (test_lpcnet src/test_lpcnet.cpp)
target_include_directories(test_lpcnet PRIVATE ./include ./src)
target_link_libraries (test_lpcnet LINK_PUBLIC lpcnet)
if(!MSVC)
target_link_libraries (test_lpcnet LINK_PUBLIC lpcnet m)
endif()
