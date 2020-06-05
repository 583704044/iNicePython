cmake_minimum_required(VERSION 3.12)
project(nlm_tester)

set(CMAKE_CXX_STANDARD 98)

include_directories(include)
include_directories(include/R_ext)
link_directories(lib)


#add_library(nlm_simple SHARED nlm_simple.cpp uncmin.cpp FunctionInfo.cpp nlm_tester.cpp nlm_tester.h)

add_executable(nlm_tester nlm_tester.cpp nlm_simple.cpp
        uncmin.cpp FunctionInfo.cpp
        nlm_tester.cpp nlm_tester.h nmath_slim.h
        fmax2.c fmin2.c nmath_slim.h)

#target_link_libraries(nlm_tester "libRblas.so")
#target_link_libraries(nlm_tester "libRlapack.so")