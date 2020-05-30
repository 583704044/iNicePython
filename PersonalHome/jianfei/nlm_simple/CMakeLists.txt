cmake_minimum_required(VERSION 3.12)
project(nlm_simple)

set(CMAKE_CXX_STANDARD 98)

add_library(nlm_simple SHARED nlm_simple.cpp nlm_simple.h)