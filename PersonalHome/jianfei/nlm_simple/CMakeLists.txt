cmake_minimum_required(VERSION 3.12)
project(nlm_tester)

set(CMAKE_CXX_STANDARD 11)

include_directories(include)
include_directories(include/R_ext)

#add_library(nlm_simple SHARED nlm_simple.cpp uncmin.cpp FunctionInfo.cpp nlm_tester.cpp nlm_tester.h)

#SET(OBJS
#        ${CMAKE_CURRENT_SOURCE_DIR}/exelib/dtrsl.o
#        )
#
#SET_SOURCE_FILES_PROPERTIES(
#        ${OBJS}
#        PROPERTIES
#        EXTERNAL_OBJECT true
#        GENERATED true
#)

add_library(nlm_simple SHARED
        nlm_simple.cpp
        FunctionInfo.cpp
        uncmin.cpp
        fmax2.cpp fmin2.cpp
        sharedlib/dtrsl.o
        sharedlib/ddot.o
        sharedlib/dscal.o
        sharedlib/dnrm2.o
        sharedlib/daxpy.o)

#add_executable(nlm_tester
#        nlm_tester.cpp nlm_simple.cpp
#        FunctionInfo.cpp
#        uncmin.cpp
#        fmax2.cpp fmin2.cpp
#        ./exelib/dtrsl.o
#        ./exelib/ddot.o
#        ./exelib/dscal.o
#        ./exelib/dnrm2.o
#        ./exelib/daxpy.o)

#link_directories(/usr/exelib/libblas)
#TARGET_LINK_LIBRARIES(nlm_tester libblas.so)

#target_link_libraries(nlm_tester "libRblas.so")
#target_link_libraries(nlm_tester "libRlapack.so")

