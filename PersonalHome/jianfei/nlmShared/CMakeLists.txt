cmake_minimum_required(VERSION 3.12)
project(nlmShared)

set(CMAKE_CXX_STANDARD 11)

include_directories(include)
include_directories(include/R_ext)

add_library(nlmShared SHARED r_nlm.c
        include/config.h
        include/Defn.h
        include/R_ext/Applic.h
        include/R_ext/RS.h
        statsR.h
        stats.h)
