project(lpcnet)
cmake_minimum_required(VERSION 3.10.2)

cmake_host_system_information(RESULT _os QUERY OS_NAME)
cmake_host_system_information(RESULT _platform QUERY OS_PLATFORM)

message(STATUS "OS: " ${_os})
message(STATUS "Platform: " ${_platform})

if(MSVC)
set (CMAKE_C_FLAGS "-D__AVX__ -D__AVX2__")
set (CMAKE_CXX_FLAGS "/std:c++17 /EHsc -D__AVX__ -D__AVX2__")
else()
set (CMAKE_C_FLAGS "-O3 -g -mavx2 -mfma")
set (CMAKE_CXX_FLAGS "-std=c++17 -O3 -g -mavx2 -mfma")
endif()

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
	src/nnet_data.c
	src/ceps_codebooks.c)
target_include_directories(lpcnet PUBLIC ./include PRIVATE ./src)
target_compile_definitions(lpcnet PRIVATE LPCNet)

add_executable (dump_data src/dump_data.cpp)
target_include_directories(dump_data PRIVATE ./include ./src)
target_link_libraries (dump_data LINK_PUBLIC lpcnet)
if(!MSVC)
target_link_libraries (dump_data LINK_PUBLIC lpcnet m)
endif()

add_executable (test_lpcnet src/test_lpcnet.cpp)
target_include_directories(test_lpcnet PRIVATE ./include ./src)
target_link_libraries (test_lpcnet LINK_PUBLIC lpcnet)
if(!MSVC)
target_link_libraries (test_lpcnet LINK_PUBLIC lpcnet m)
endif()
