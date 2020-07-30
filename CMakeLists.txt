project(lpcnet)
cmake_minimum_required(VERSION 3.10.2)

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


target_include_directories(lpcnet PRIVATE include src)

set (CMAKE_C_FLAGS "-D__AVX__ -D__AVX2__")
set (CMAKE_CXX_FLAGS "-D__AVX__ -D__AVX2__")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_definitions(-D__AVX__)
target_compile_definitions(lpcnet PRIVATE LPCNet)


add_executable (dump src/dump_data.c)
target_include_directories(dump PRIVATE include src)
target_link_libraries (dump LINK_PUBLIC lpcnet)

add_executable (test src/test_lpcnet.c)
target_include_directories(test PRIVATE include src)
target_link_libraries (test LINK_PUBLIC lpcnet)
