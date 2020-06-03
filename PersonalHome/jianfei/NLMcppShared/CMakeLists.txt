cmake_minimum_required(VERSION 3.12)
project(NLMcppShared)

set(CMAKE_CXX_STANDARD 98)

add_library(NLMcppShared SHARED library.cpp library.h)