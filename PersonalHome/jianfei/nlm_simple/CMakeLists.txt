cmake_minimum_required(VERSION 3.12)
project(nlm_simple)

set(CMAKE_CXX_STANDARD 98)

include_directories(include)
include_directories(include/R_ext)

add_library(nlm_simple SHARED nlm_simple.cpp uncmin.cpp FunctionInfo.cpp)
