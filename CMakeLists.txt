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


target_include_directories(lpcnet 
PRIVATE include src)

set (CMAKE_C_FLAGS "-DTACOTRON2 -D__AVX__")
set (CMAKE_CXX_FLAGS "-DTACOTRON2 -D__AVX__")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_definitions(-D__AVX__)

target_compile_definitions(lpcnet PRIVATE TACOTRON2)
