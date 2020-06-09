CC = g++
CFLAGS = -I./include -I./include/R_ext
CFLAGS = -Wall


nlm_tester: nlm_tester.o nlm_simple.o FunctionInfo.o uncmin.o
     $(CC) -o nlm_tester.o nlm_simple.o FunctionInfo.o uncmin.o

nlm_tester.o: nlm_tester.cpp 
